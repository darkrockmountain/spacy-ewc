from thinc.api import Model
from spacy.training import Example
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from thinc.api import get_current_ops
import spacy.util as spacy_utils
from typing import List, Optional, cast, Union, get_args
import logging
from .vector_dict import VectorDict


logger = logging.getLogger(__name__)


class EWC:
    """
    The EWC (Elastic Weight Consolidation) class implements the EWC algorithm to prevent catastrophic forgetting
    in neural networks during sequential learning tasks. It captures the model's parameters after training on the first task,
    computes the Fisher Information Matrix to estimate the importance of each parameter, and applies a penalty to the loss function
    during subsequent training to retain important parameters.

    References:
    - Kirkpatrick, J., Pascanu, R., Rabinowitz, N. C., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017).
      Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521-3526.
      (https://arxiv.org/abs/1612.00796)
    """

    model_type = Union[Language, TrainablePipe]

    # ===== Initialization and Setup =====
    def __init__(
        self,
        pipe: model_type,
        data: List[Example],
        *,
        lambda_: float = 1000.0,
        pipe_name: Optional[str] = None,
    ):
        """
        Initialize the EWC instance by capturing the model's parameters after training on the first task,
        computing the Fisher Information Matrix (FIM), and setting up the pipeline.

        Parameters:
        - pipe (Union[Language, TrainablePipe]): The spaCy Language model or a TrainablePipe component.
        - data (List[Example]): The list of training examples used to compute the Fisher Information Matrix.
        - lambda_ (float): The regularization strength that scales the EWC penalty (default is 1000.0).
        - pipe_name (Optional[str]): The name of the pipe component if `pipe` is a Language model (default is 'ner').

        The initialization performs the following steps:
        1. Validates the type of the provided pipe.
        2. Retrieves the TrainablePipe component from the Language model if necessary.
        3. Captures the initial model parameters (θ*), which will serve as a reference for important parameters.
        4. Computes the Fisher Information Matrix (F), which estimates the importance of each parameter based on its
           contribution to the loss function.

        Mathematical Formulation:
        - θ*: The parameters of the model after training on the first task.
        - F: The Fisher Information Matrix, where each diagonal element F_i estimates the importance of parameter θ_i.
        - λ (lambda_): The regularization strength controlling the trade-off between retaining old knowledge and learning new information.

        Reference:
        - The Fisher Information Matrix is computed as the expected value of the squared gradients of the loss function
          with respect to the model parameters, evaluated on the data from the first task.

        Raises:
        - ValueError: If the pipe is not an instance of Language or TrainablePipe.
        - ValueError: If initial model parameters (θ*) are not set.
        - ValueError: If the Fisher Information Matrix (F) is not computed.
        """
        logger.info("Initializing EWC instance.")

        # Ensure the provided pipe is either a Language model or TrainablePipe
        allowed_classes = get_args(EWC.model_type)
        if not isinstance(pipe, allowed_classes):
            allowed_class_names = [cls.__name__ for cls in allowed_classes]
            raise ValueError(
                f"pipe param can only be an instance of one of: {allowed_class_names}"
            )

        self.pipe: TrainablePipe = pipe

        # If the pipe is a Language model, retrieve the named component
        if isinstance(self.pipe, Language):
            if not pipe_name:
                pipe_name = "ner"
            self.pipe = self.pipe.get_pipe(pipe_name)

        # Set the regularization strength
        self.lambda_ = lambda_

        # Capture parameters after training on the first task
        self.theta_star: VectorDict = self._capture_current_parameters(copy=True)
        logger.debug("Captured initial model parameters (theta_star).")

        # Ensure theta_star has been set correctly
        if not self.theta_star:
            raise ValueError("Initial model parameters are not set.")

        # Compute the Fisher Information Matrix based on the provided training
        # data
        self.fisher_matrix: VectorDict = self._compute_fisher_matrix(data)
        logger.debug("Computed Fisher Information Matrix.")

        # Ensure the Fisher Information Matrix is computed
        if not self.fisher_matrix:
            raise ValueError("Fisher Information Matrix has not been computed.")

    def _validate_initialization(self, function_name: str = None):
        """
        Validate that the Fisher Information Matrix and the initial parameters θ* are initialized.

        Parameters:
        - function_name (str): The name of the function calling this validation (optional).

        Raises:
        - ValueError: If the Fisher Information Matrix (F) has not been computed.
        - ValueError: If the initial model parameters (θ*) are not set.

        This method ensures that the necessary components for EWC computations are available before proceeding
        with penalty calculations or gradient updates.
        """
        if not self.fisher_matrix:
            raise ValueError(
                "Fisher Information Matrix has not been computed."
                + (
                    f" Ensure `self.fisher_matrix` has been initialized with start gradients before calling `{function_name}()`."
                    if function_name
                    else ""
                )
            )
        if not self.theta_star:
            raise ValueError(
                "Initial model parameters are not set."
                + (
                    f" Ensure `self.theta_star` has been initialized with start parameters before calling `{function_name}()`."
                    if function_name
                    else ""
                )
            )

    def set_lambda(self, new_lambda: float):
        """
        Update the regularization strength lambda_.

        Parameters:
        - new_lambda (float): The new value for lambda_.
        """
        logger.info(f"Updating lambda_ from {self.lambda_} to {new_lambda}.")
        self.lambda_ = new_lambda

    # ===== Parameter Management =====

    def _capture_current_parameters(self, copy=False) -> VectorDict:
        """
        Retrieve the current model parameters θ, with an option to copy or reference them.

        Parameters:
        - copy (bool): If True, returns a copy of the parameters; otherwise, returns references.

        Returns:
        - current_params (VectorDict): A dictionary of current model parameters.

        The parameters are stored in a VectorDict with keys formatted as "{layer.name}_{param_name}".

        This method is used to capture the model's parameters at various stages, such as after training on
        a task, to compare against θ* during penalty computation.
        """
        logger.info("Retrieving current model parameters.")
        current_params = VectorDict()
        ner_model: Model = self.pipe.model
        for layer in ner_model.layers:
            for name in layer.param_names:
                # Conditionally copy or keep reference based on the 'copy'
                # parameter
                try:
                    if copy:
                        current_params[f"{layer.name}_{name}"] = layer.get_param(
                            name
                        ).copy()
                    else:
                        current_params[f"{layer.name}_{name}"] = layer.get_param(name)
                except Exception as e:
                    logger.warning(
                        f"Failed to retrieve parameter '{name}' for copying: {str(e)}"
                    )
        return current_params

    # ===== Fisher Information Matrix Calculation =====

    def _compute_fisher_matrix(self, examples: List[Example]) -> VectorDict:
        """
        Compute the Fisher Information Matrix (F) for the model based on the training examples.

        Parameters:
        - examples (List[Example]): The list of training examples from the first task.

        Returns:
        - fisher_matrix (VectorDict): A dictionary representing the Fisher Information Matrix.

        The Fisher Information Matrix estimates the importance of each parameter θ_i by calculating
        the expected value of the squared gradients of the loss function L with respect to θ_i.

        Mathematical Formulation:
        - For each parameter θ_i, compute:
          F_i = E_x[(∂L(x; θ)/∂θ_i)^2]

        - Where E_x denotes the expectation over the data samples x.



        Implementation Details:
        - The method performs forward and backward passes over the training data to accumulate gradients.
        - Gradients are squared and summed to approximate the Fisher Information Matrix.
        - The accumulated gradients are averaged over the number of batches to obtain the final FIM.

        Reference:
        - The use of the diagonal approximation of the Fisher Information Matrix as in the original EWC paper:
          Kirkpatrick et al., 2017.

        Raises:
        - ValueError: If no batches yield positive loss, indicating the Fisher Information Matrix cannot be computed.
        """
        logger.info("Computing Fisher Information Matrix.")

        # Prepare the model operations
        ops = get_current_ops()

        # Set up data batching
        batches = spacy_utils.minibatch(
            examples, size=spacy_utils.compounding(4.0, 32.0, 1.001)
        )

        # Initialize an empty Fisher Information Matrix
        fisher_matrix = VectorDict()
        num_batches = 0

        for batch in batches:
            # Track the loss
            losses = {}

            # Perform forward and backward passes to compute gradients
            self.pipe.update(batch, losses=losses, sgd=None)

            # If no NER loss is computed, skip this batch
            if "ner" not in losses or losses["ner"] <= 0:
                logger.warning("Skipping batch with no or zero loss.")
                continue

            for layer in cast(Model, self.pipe.model).layers:
                # Retrieve gradient information for each parameter
                for (_, name), (_, grad) in layer.get_gradients().items():
                    if name not in layer.param_names:
                        continue
                    # Square the gradient and add to the Fisher Information
                    # Matrix
                    grad = ops.asarray(grad).copy() ** 2
                    key = f"{layer.name}_{name}"
                    try:
                        if key in fisher_matrix:
                            fisher_matrix[key] += grad
                        else:
                            fisher_matrix[key] = grad
                    except ValueError as e:
                        logger.error(f"Error updating Fisher Matrix for {key}: {e}")
                        continue

            num_batches += 1

        if num_batches == 0:
            raise ValueError(
                "No batches yielded positive loss; Fisher Information Matrix not computed."
            )

        # Average the matrix over the batches
        for name in fisher_matrix:
            fisher_matrix[name] /= num_batches
            logger.debug(f"Fisher Matrix value for {name}: {fisher_matrix[name]}")
        return fisher_matrix

    # ===== Penalty Computations =====

    def compute_ewc_penalty(self) -> float:
        """
        Calculate the EWC penalty term Ω(θ) for the loss function, based on parameter importance.

        Returns:
        - ewc_penalty (float): The scalar value of the EWC penalty.

        The EWC penalty encourages the model to retain important parameters learned from previous tasks.

        Mathematical Formulation:
        - Ω(θ) = (1/2) * Σ_i F_i * (θ_i - θ_i^*)^2
        - Where:
          - θ_i: Current parameter value.
          - θ_i^*: Parameter value after training on the first task.
          - F_i: Fisher Information for parameter θ_i.

        The penalty term is added to the loss function to penalize deviations from θ_i^* proportional to F_i.

        Reference:
        - Equation (3) in Kirkpatrick et al., 2017.

        Raises:
        - ValueError: If the Fisher Information Matrix or initial parameters are not initialized.
        """
        self._validate_initialization("compute_ewc_penalty")
        logger.info("Calculating loss penalty.")

        ewc_penalty = 0.0
        current_params = self._capture_current_parameters()

        for key in self.theta_star.keys():
            current_param = current_params[key]
            theta_star_param = self.theta_star[key]
            fisher_param = self.fisher_matrix[key]

            # Compute the penalty if shapes match
            if current_param.shape == theta_star_param.shape == fisher_param.shape:
                # Compute (θ_i - θ_i^*)^2
                param_diff_squared = (current_param - theta_star_param) ** 2
                # Compute F_i * (θ_i - θ_i^*)^2
                penalty_contrib = (fisher_param * param_diff_squared).sum()
                ewc_penalty += penalty_contrib
                logger.debug(f"Penalty contribution for {key}: {penalty_contrib}")

        # Multiply by 0.5 as per the formula
        ewc_penalty *= 0.5

        return float(ewc_penalty)

    def compute_gradient_penalty(self):
        """
        Calculate the gradient penalty to be applied to current parameters based on the Fisher Information Matrix.

        Returns:
        - ewc_penalty_gradients (VectorDict): A dictionary of gradient penalties for each parameter.

        The gradient penalty is computed as the derivative of the EWC penalty term with respect to θ_i.

        Mathematical Formulation:
        - ∂Ω(θ)/∂θ_i = F_i * (θ_i - θ_i^*)
        - This gradient penalty is added to the parameter gradients during optimization.

        Reference:
        - The gradient of the EWC penalty as used in the optimization process in Kirkpatrick et al., 2017.

        Raises:
        - ValueError: If the Fisher Information Matrix or initial parameters are not initialized.
        """
        self._validate_initialization("compute_gradient_penalty")
        logger.info("Calculating gradient penalty.")

        ewc_penalty_gradients = VectorDict()
        current_params = self._capture_current_parameters()

        for key in self.theta_star.keys():
            current_param = current_params[key]
            theta_star_param = self.theta_star[key]
            fisher_param = self.fisher_matrix[key]

            # Calculate the EWC gradient penalty
            # ∂Ω(θ)/∂θ_i = F_i * (θ_i - θ_i^*)
            ewc_penalty = fisher_param * (current_param - theta_star_param)
            ewc_penalty_gradients[key] = ewc_penalty
            logger.debug(f"Gradient penalty for {key}: {ewc_penalty}")

        return ewc_penalty_gradients

    # ===== Gradient Application =====
    def apply_ewc_penalty_to_gradients(self):
        """
        Apply the EWC penalty directly to the model's gradients during training.

        This method modifies the gradients of the model's parameters in-place by adding the scaled EWC gradient penalty.

        Mathematical Formulation:
        - For each parameter θ_i, update the gradient g_i as:
            g_i ← g_i + λ * F_i * (θ_i - θ_i^*)
        - Where:
            - g_i: Original gradient of parameter θ_i.
            - λ: Regularization strength (lambda_).
            - F_i: Fisher Information for parameter θ_i.
            - θ_i^*: Parameter value after training on the first task.

        Reference:
        - The application of the EWC gradient penalty during optimization as per Kirkpatrick et al., 2017.

        Raises:
        - ValueError: If the Fisher Information Matrix or initial parameters are not initialized.
        - ValueError: If there are mismatches in parameter shapes or data types.
        """

        self._validate_initialization("apply_ewc_penalty_to_gradients")
        logger.info(f"Applying EWC penalty to gradients with lambda={self.lambda_}.")

        ner_model: Model = self.pipe.model
        current_params = self._capture_current_parameters()

        for layer in ner_model.layers:
            for (_, name), (_, grad) in layer.get_gradients().items():
                if name not in layer.param_names:
                    continue
                key_name = f"{layer.name}_{name}"

                # Ensure key presence and shape compatibility
                if (
                    key_name not in current_params
                    or key_name not in self.theta_star
                    or key_name not in self.fisher_matrix
                ):
                    raise ValueError(f"Invalid key_name found '{key_name}'.")

                theta_current = current_params[key_name]
                theta_star_param = self.theta_star[key_name]
                fisher_param = self.fisher_matrix[key_name]

                if (
                    theta_current.shape != theta_star_param.shape
                    or theta_current.shape != fisher_param.shape
                    or theta_current.shape != grad.shape
                ):
                    logger.info(f"Shape mismatch for {key_name}, skipping.")
                    continue

                if (
                    theta_current.dtype != theta_star_param.dtype
                    or theta_current.dtype != fisher_param.dtype
                    or theta_current.dtype != grad.dtype
                ):
                    logger.info(f"Dtype mismatch for {key_name}, skipping.")
                    continue

                # Calculate and apply the EWC penalty to the gradient
                # Update gradient: g_i ← g_i + λ * F_i * (θ_i - θ_i^*)
                ewc_penalty = fisher_param * (theta_current - theta_star_param)
                grad += self.lambda_ * ewc_penalty
                logger.debug(f"Applied penalty for {key_name}: {ewc_penalty}")

    # ===== Loss Calculation =====

    def ewc_loss(self, task_loss):
        """
        Calculate the total EWC loss by combining the task-specific loss with the EWC penalty term.

        Parameters:
        - task_loss (float): The loss value from the current task.
        - lambda_ (float): The regularization strength that scales the EWC penalty (default is 1000).

        Returns:
        - ewc_adjusted_loss (float): The total loss adjusted with the EWC penalty.

        Mathematical Formulation:
        - L_total = L_task + (λ / 2) * Ω(θ)
        - Where:
          - L_task: Loss from the current task.
          - Ω(θ) = Σ_i F_i * (θ_i - θ_i^*)^2

        Reference:
        - The total loss function including the EWC penalty as per Kirkpatrick et al., 2017.

        """
        logger.info("Calculating EWC-adjusted loss.")
        ewc_penalty = self.compute_ewc_penalty()
        ewc_adjusted_loss = task_loss + (self.lambda_ * ewc_penalty)
        logger.debug(f"Computed EWC loss: {ewc_adjusted_loss}")
        return ewc_adjusted_loss
