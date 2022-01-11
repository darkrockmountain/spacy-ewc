from thinc.api import Model
from spacy.training import Example
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from thinc.api import get_current_ops
import spacy.util as spacy_utils
from typing import List, Tuple, Dict, Optional, Any, cast, Union, get_args, Callable
from spacy_ewc.vector_dict import VectorDict
import logging

logger = logging.getLogger(__name__)


class EWC:

    model_type = Union[Language, TrainablePipe]

    # ===== Initialization and Setup =====
    def __init__(self, pipe: model_type, data: List[Example], *, pipe_name: Optional[str] = None):
        """
        Initialize the EWC class by capturing the model's parameters after training
        on the first task, computing the Fisher Information Matrix, and setting up the pipeline.
        """
        logger.info("Initializing EWC instance.")

        # Ensure the provided pipe is either a Language model or TrainablePipe
        allowed_classes = get_args(EWC.model_type)
        if not isinstance(pipe, allowed_classes):
            allowed_class_names = [cls.__name__ for cls in allowed_classes]
            raise ValueError(
                f"pipe param can only be an instance of one of: {
                    allowed_class_names}"
            )

        self.pipe: TrainablePipe = pipe

        # If the pipe is a Language model, retrieve the named component
        if isinstance(self.pipe, Language):
            if not pipe_name:
                pipe_name = "ner"
            self.pipe = self.pipe.get_pipe(pipe_name)

        # Capture parameters after training on the first task
        self.theta_star: VectorDict = self._capture_current_parameters(
            copy=True)
        logger.debug("Captured initial model parameters (theta_star).")

        # Ensure theta_star has been set correctly
        if not self.theta_star:
            raise ValueError("Initial model parameters are not set.")

        # Compute the Fisher Information Matrix based on the provided training data
        self.fisher_matrix: VectorDict = self._compute_fisher_matrix(data)
        logger.debug("Computed Fisher Information Matrix.")

        # Ensure the Fisher Information Matrix is computed
        if not self.fisher_matrix:
            raise ValueError(
                "Fisher Information Matrix has not been computed.")

    def _validate_initialization(self, function_name: str = None):
        """
        Check that the Fisher Information Matrix and theta_star parameters are initialized.
        """
        if not self.fisher_matrix:
            raise ValueError("Fisher Information Matrix has not been computed." +
                             (f" Ensure `self.fisher_matrix` has been initialized with start gradients before calling `{function_name}()`." if function_name else ""))
        if not self.theta_star:
            raise ValueError("Initial model parameters are not set." +
                             (f" Ensure `self.theta_star` has been initialized with start parameters before calling `{function_name}()`." if function_name else ""))

    # ===== Parameter Management =====

    def _capture_current_parameters(self, copy=False) -> VectorDict:
        """
        Retrieve the current model parameters, with an option to copy or reference them.
        """
        logger.info("Retrieving current model parameters.")
        current_params = VectorDict()
        ner_model: Model = self.pipe.model
        for layer in ner_model.walk():
            for name in layer.param_names:
                # Conditionally copy or keep reference based on the 'copy' parameter
                try:
                    if copy:
                        current_params[f"{layer.name}_{
                            name}"] = layer.get_param(name).copy()
                    else:
                        current_params[f"{layer.name}_{
                            name}"] = layer.get_param(name)
                except Exception as e:
                    logger.warning(f"Failed to retrieve parameter '{
                                   name}' for copying: {str(e)}")
        return current_params

    # ===== Fisher Information Matrix Calculation =====

    def _compute_fisher_matrix(self, examples: List[Example]) -> VectorDict:
        """
        Compute the Fisher Information Matrix for the model based on the training examples.
        This matrix estimates parameter importance for knowledge retention.
        """
        logger.info("Computing Fisher Information Matrix.")

        # Prepare the model operations
        ops = get_current_ops()

        # Set up data batching
        batches = spacy_utils.minibatch(
            examples,
            size=spacy_utils.compounding(4.0, 32.0, 1.001)
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
            if 'ner' not in losses or losses['ner'] <= 0:
                logger.warning("Skipping batch with no or zero loss.")
                continue

            for layer in cast(Model, self.pipe.model).walk():
                # Retrieve gradient information for each parameter
                for (_, name), (_, grad) in layer.get_gradients().items():
                    if name not in layer.param_names:
                        continue

                    # Square the gradient and add to the Fisher Information Matrix
                    grad = ops.asarray(grad).copy() ** 2
                    try:
                        fisher_matrix[f"{layer.name}_{
                            name}"] = fisher_matrix.get(name, 0) + grad
                    except ValueError as e:
                        logger.error(
                            f"Error updating Fisher Matrix for {name}: {e}")
                        continue

            num_batches += 1

        if num_batches == 0:
            raise ValueError(
                "No batches yielded positive loss; Fisher Information Matrix not computed.")

        # Average the matrix over the batches
        for name in fisher_matrix:
            fisher_matrix[name] /= num_batches
            logger.debug(f"Fisher Matrix value for {
                         name}: {fisher_matrix[name]}")

        return fisher_matrix

    # ===== Penalty Computations =====

    def compute_ewc_penalty(self) -> float:
        """
        Calculate the EWC penalty term for the loss function, based on parameter importance.
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
                penalty_contrib = (
                    fisher_param * (current_param - theta_star_param) ** 2).sum()
                ewc_penalty += penalty_contrib
                logger.debug(f"Penalty contribution for {
                             key}: {penalty_contrib}")

        return float(ewc_penalty)

    def compute_gradient_penalty(self):
        """
        Calculate the gradient penalty to be applied to current parameters based on the Fisher Information Matrix.
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
            ewc_penalty = fisher_param * \
                (current_param.copy() - theta_star_param)
            ewc_penalty_gradients[key] = ewc_penalty
            logger.debug(f"Gradient penalty for {key}: {ewc_penalty}")

        return ewc_penalty_gradients

    # ===== Gradient Application =====
    def apply_ewc_penalty_to_gradients(self, lambda_=1000):
        """
        Apply the EWC penalty directly to the model's gradients.
        """
        self._validate_initialization("apply_ewc_penalty_to_gradients")
        logger.info(
            f"Applying EWC penalty to gradients with lambda={lambda_}.")

        ner_model = self.pipe.model
        current_params = self._capture_current_parameters()

        for layer in ner_model.walk():
            for (_, name), (_, grad) in layer.get_gradients().items():
                if name not in layer.param_names:
                    continue
                key_name = f"{layer.name}_{name}"

                # Ensure key presence and shape compatibility
                if key_name not in current_params or key_name not in self.theta_star or key_name not in self.fisher_matrix:
                    raise ValueError(f"Invalid key_name found '{key_name}'.")

                theta_current = current_params[key_name]
                theta_star_param = self.theta_star[key_name]
                fisher_param = self.fisher_matrix[key_name]

                if theta_current.shape != theta_star_param.shape or theta_current.shape != fisher_param.shape or theta_current.shape != grad.shape:
                    logger.info(f"Shape mismatch for {key_name}, skipping.")
                    continue

                if theta_current.dtype != theta_star_param.dtype != fisher_param.dtype != grad.dtype:
                    logger.info(f"Dtype mismatch for {key_name}, skipping.")
                    continue

                # Calculate and apply the EWC penalty to the gradient
                ewc_penalty = fisher_param * (theta_current - theta_star_param)
                grad += (lambda_ * ewc_penalty)
                logger.debug(f"Applied penalty for {key_name}: {ewc_penalty}")

    # ===== Loss Calculation =====
    def ewc_loss(self, task_loss, lambda_=1000):
        """
        Calculate the total EWC loss by combining the task loss with the EWC penalty term.
        """
        logger.info("Calculating EWC-adjusted loss.")
        ewc_adjusted_loss = task_loss + \
            (lambda_ * 0.5 * self.compute_ewc_penalty())
        logger.debug(f"Computed EWC loss: {ewc_adjusted_loss}")
        return ewc_adjusted_loss
