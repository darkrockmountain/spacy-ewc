from thinc.api import Model
from spacy.training import Example
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from thinc.api import get_current_ops
import spacy.util as spacy_utils
from typing import List, Tuple, Dict, Optional, Any, cast, Union, get_args, Callable


class EWC:

    model_type = Union[Language, TrainablePipe]

    def __init__(self,  pipe: model_type, data: List[Example], *,
                 pipe_name: Optional[str] = None):

        allowed_classes = get_args(EWC.model_type)
        if not isinstance(pipe, allowed_classes):
            allowed_class_names = [cls.__name__ for cls in allowed_classes]
            raise ValueError(
                f"pipe param can only be an instance of one of: {
                    allowed_class_names}"
            )

        self.pipe: TrainablePipe = pipe

        if isinstance(self.pipe, Language):
            if not pipe_name:
                pipe_name = "ner"
            self.pipe = self.pipe.get_pipe(pipe_name)

        # To store the Fisher Information Matrix after computing it with the initial task
        self.fisher_matrix = None

        # Capture parameters after training on the first task (copy True we need to keep it)
        self.theta_star = self.get_current_params(copy=True)

        # Calculate Fisher Information Matrix on the first task
        self.fisher_matrix = self._compute_fisher_matrix(
            data)

    def _compute_fisher_matrix(self, examples: List[Example]):

        # Prepare the model operations
        ops = get_current_ops()

        # Set up DataLoader for batching
        batches = spacy_utils.minibatch(
            # Create Examples for each instance in the batch
            # Ensure each data item contains valid NER examples
            examples,
            size=spacy_utils.compounding(4.0, 32.0, 1.001)
        )

        # Initialize an empty Fisher Information Matrix
        fisher_matrix = {}

        num_batches = 0

        for batch in batches:

            # Track the loss to ensure it’s being computed
            losses = {}

            # Perform forward and backward passes to compute gradients (no parameter update)
            # Use the pipe ner update without optimizer to compute gradients without updating model parameters
            self.pipe.update(batch, losses=losses, sgd=None)

            # Check if loss was computed, otherwise skip this batch
            if 'ner' not in losses or losses['ner'] <= 0:
                continue

            for layer in cast(Model, self.pipe.model).walk():
                # Retrieve the gradient for this parameter
                for (_, name), (_, grad) in layer.get_gradients().items():
                    if not name in layer.param_names:
                        # there is no need to store gradient nodes that are not going to be calculated.
                        continue
                    # Convert to array for easier manipulation
                    grad = ops.asarray(grad).copy() ** 2
                    try:
                        fisher_matrix[f"{layer.name}_{name}"] = fisher_matrix.get(
                            name, 0) + grad
                    except ValueError as e:
                        continue

            num_batches += 1
        # Average Fisher Information Matrix over all batches
        for name in fisher_matrix:
            fisher_matrix[name] /= num_batches

        return fisher_matrix

    # Get the current params of the model.
    def get_current_params(self, copy=False):
        current_params = {}
        ner_model: Model = self.pipe.model
        for layer in ner_model.walk():
            for name in layer.param_names:
                # Conditionally copy or keep reference based on the 'copy' parameter
                if copy:
                    current_params[f"{layer.name}_{
                        name}"] = layer.get_param(name).copy()
                else:
                    current_params[f"{layer.name}_{
                        name}"] = layer.get_param(name)
        return current_params

    def loss_penalty(self) -> float:
        # Initialize the penalty to zero
        ewc_penalty = 0.0
        current_params = self.get_current_params()

        for key in self.theta_star.keys():
            current_param = current_params[key]
            theta_star_param = self.theta_star[key]
            fisher_param = self.fisher_matrix[key]
            if current_param.shape == theta_star_param.shape == fisher_param.shape:
                ewc_penalty += (fisher_param *
                                (current_param - theta_star_param) ** 2).sum()

        return float(ewc_penalty)

    def gradient_penalty(self):
        ewc_penalty_gradients = {}

        current_params = self.get_current_params()

        for key in self.theta_star.keys():
            # Get current model parameters
            current_param = current_params[key]
            theta_star_param = self.theta_star[key]
            fisher_param = self.fisher_matrix[key]
            # Compute the EWC penalty term for each parameter
            ewc_penalty = fisher_param * \
                (current_param.copy() - theta_star_param)
            # modify the models pointer gradient
            ewc_penalty_gradients[key] = ewc_penalty

        return ewc_penalty_gradients

    def apply_ewc_penalty_to_gradients(self, lambda_=1000):
        ner_model = self.pipe.model
        current_params = self.get_current_params()

        for layer in ner_model.walk():
            # Retrieve the gradient for this parameter
            for (_, name), (_, grad) in layer.get_gradients().items():
                if not name in layer.param_names:
                    # there is no need to access gradient nodes that are not going to be calculated.
                    continue
                key_name = f"{layer.name}_{name}"
                if key_name not in current_params or key_name not in self.theta_star or key_name not in self.fisher_matrix:
                    raise ValueError(
                        f"Invalid key_name found '{key_name}': "
                        f"theta_current key names {
                            current_params.keys()}, "
                        f"theta_star_param names {self.theta_star.keys()}, "
                        f"fisher_param names {self.fisher_matrix.keys()}, "
                    )

                theta_current = current_params[key_name]
                theta_star_param = self.theta_star[key_name]
                fisher_param = self.fisher_matrix[key_name]

                # Validation Check: Ensure shapes and types are compatible
                if (theta_current.shape != theta_star_param.shape or
                    theta_current.shape != fisher_param.shape or
                        theta_current.shape != grad.shape):
                    continue
                if not (theta_current.dtype == theta_star_param.dtype == fisher_param.dtype == grad.dtype):
                    continue
                # Calculate the EWC penalty for this parameter
                ewc_penalty = fisher_param * (theta_current - theta_star_param)

                # Add EWC penalty directly to the gradient (in-place modification)
                grad += (lambda_ * ewc_penalty)

    def ewc_loss(self, task_loss, lambda_=1000):
        return task_loss + (lambda_ * 0.5*self.loss_penalty())
