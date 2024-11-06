from thinc.api import Model, Optimizer
from spacy.training import Example
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from typing import List, Optional, Callable, get_args
import logging
from spacy_ewc.ewc import EWC


# Set up logging
logger = logging.getLogger(__name__)


class EWCModelWrapper(Model):
    """
    A model wrapper that integrates Elastic Weight Consolidation (EWC) penalty
    during model updates. This wrapper intercepts the `finish_update` method to
    apply EWC penalty to the gradients before completing the model's update.

    Attributes:
        _wrapped_model (Model): The underlying model to be wrapped.
        apply_ewc_penalty_to_gradients (Callable): A function to apply the EWC penalty
            to the model's gradients.
    """

    def __init__(self, model: Model, apply_ewc_penalty_to_gradients: Callable):
        """
        Initialize the _EWCModelWrapper with a model and the EWC gradient penalty function.

        Args:
            model (Model): The model to wrap, usually a Thinc model in a SpaCy pipeline.
            apply_ewc_penalty_to_gradients (Callable): The function to apply EWC penalties
                to gradients during updates.
        """
        self._wrapped_model: Model = model
        self.apply_ewc_penalty_to_gradients: Callable = apply_ewc_penalty_to_gradients
        logger.info("Initialized _EWCModelWrapper with EWC penalty function.")

    def finish_update(self, optimizer: Optimizer) -> None:
        """
        Override the model's `finish_update` method to apply the EWC penalty before
        the optimizer updates parameters based on current gradients.

        Args:
            optimizer (Optimizer): The optimizer used to update model parameters.
        """
        logger.debug("Applying EWC penalty before model update.")
        # Apply EWC penalty to gradients before finishing the update
        self.apply_ewc_penalty_to_gradients()
        logger.debug("EWC penalty applied to gradients.")
        # Complete the model's update with the optimizer
        return self._wrapped_model.finish_update(optimizer=optimizer)

    def __getattribute__(self, name):
        """
        Override attribute access to prioritize `_EWCModelWrapper`'s `finish_update`
        while delegating other attributes to `_wrapped_model`.

        Args:
            name (str): The attribute name to access.

        Returns:
            Any: The attribute value from either `_EWCModelWrapper` or `_wrapped_model`.
        """
        if name == "finish_update":
            # Ensure `finish_update` always uses `_EWCModelWrapper`'s version
            return object.__getattribute__(self, "finish_update")

        # Attempt to retrieve the attribute; fallback to `_wrapped_model` if
        # needed
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # If not found in wrapper, access it from `_wrapped_model`
            _wrapped_model = super().__getattribute__("_wrapped_model")
            attr = getattr(_wrapped_model, name)
            logger.debug(f"Attribute '{name}' accessed from _wrapped_model.")
            return attr


def create_ewc_pipe(
    pipe: EWC.model_type,
    data: List[Example],
    *,
    lambda_: float = 1000.0,
    pipe_name: Optional[str] = None,
) -> TrainablePipe:
    """
    Initialize a SpaCy trainable pipeline component (e.g., `ner`) with EWC for continual learning.
    This function sets up EWC to apply penalties to the model's gradients during training updates.

    Args:
        pipe (EWC.model_type): The SpaCy pipeline component or `Language` model.
        data (List[Example]): Training data examples used to calculate the
                              Fisher Information Matrix for EWC.
        lambda_ (float): The regularization strength that scales the EWC penalty (default is 1000.0).
        pipe_name (Optional[str]): The name of the component if `pipe` is a `Language` model.

    Returns:
        TrainablePipe: The SpaCy pipeline component with EWC penalty integration.
    """
    # Validate that `pipe` is either a `Language` model or `TrainablePipe`
    # component
    allowed_classes = get_args(EWC.model_type)
    if not isinstance(pipe, allowed_classes):
        allowed_class_names = [cls.__name__ for cls in allowed_classes]
        logger.error("Invalid pipe type provided. Must be Language or TrainablePipe.")
        raise ValueError(
            f"pipe param can only be an instance of one of: {allowed_class_names}"
        )

    # Extract the specified pipeline component if `pipe` is a Language model
    wrapped_pipe: TrainablePipe = pipe
    if isinstance(wrapped_pipe, Language):
        if not pipe_name:
            pipe_name = "ner"  # Default to the 'ner' component if none is specified
        wrapped_pipe = wrapped_pipe.get_pipe(pipe_name)
        logger.info(f"Extracted '{pipe_name}' component from Language model.")

    # Initialize EWC with the pipeline component and training data
    ewc = EWC(wrapped_pipe, data, lambda_=lambda_)
    logger.info("Initialized EWC with training data.")

    # Wrap the component's model with _EWCModelWrapper to apply EWC penalties
    wrapped_pipe.model = EWCModelWrapper(
        wrapped_pipe.model, ewc.apply_ewc_penalty_to_gradients
    )
    logger.info("Wrapped model with _EWCModelWrapper for EWC penalty integration.")

    return wrapped_pipe
