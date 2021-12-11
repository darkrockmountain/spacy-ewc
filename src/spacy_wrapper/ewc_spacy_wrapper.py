from thinc.model import Model
import spacy
from thinc.api import Model
from thinc.types import FloatsXd
from spacy.training import Example, validate_examples
from typing import List, Tuple, Dict, Optional, Any, cast, Union, get_args, Callable
from tqdm import tqdm
import logging
from thinc.api import get_current_ops
import numpy as np
import random
from thinc.api import Optimizer, set_dropout_rate
from spacy_ewc import EWC


class _EWCModelWrapper(Model):

    def __init__(self, model: Model, apply_ewc_penalty_to_gradients: Callable):
        self._wrapped_model: Model = model
        self.apply_ewc_penalty_to_gradients: Callable = apply_ewc_penalty_to_gradients

    def finish_update(self, optimizer: Optimizer) -> None:
        """Update parameters with current gradients. The optimizer is called
        with each parameter and gradient of the model.
        """
        self.apply_ewc_penalty_to_gradients()

        return self._wrapped_model.finish_update(optimizer=optimizer)

    def __getattribute__(self, name):
        if name == "finish_update":
            # Redirect `finish_update` to _EWCModelWrapper's version
            return object.__getattribute__(self, "finish_update")

        # For other attributes, default to _wrapped_model or self
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # If not found in the wrapper, try _wrapped_model
            _wrapped_model = super().__getattribute__("_wrapped_model")
            return getattr(_wrapped_model, name)


class EWCPipeWrapper(spacy.pipeline.TrainablePipe):

    def __init__(self, pipe: EWC.model_type, data: List[Example], *,
                 pipe_name: Optional[str] = None):

        allowed_classes = get_args(EWC.model_type)
        if not isinstance(pipe, allowed_classes):
            allowed_class_names = [cls.__name__ for cls in allowed_classes]
            raise ValueError(
                f"pipe param can only be an instance of one of: {
                    allowed_class_names}"
            )

        self._wrapped_pipe: spacy.pipeline.TrainablePipe = pipe
        if isinstance(self._wrapped_pipe, spacy.Language):
            if not pipe_name:
                pipe_name = "ner"
            self._wrapped_pipe = self._wrapped_pipe.get_pipe(pipe_name)

        self.ewc = EWC(self._wrapped_pipe, data)

        self._wrapped_pipe.model = _EWCModelWrapper(
            self._wrapped_pipe.model, self.ewc.apply_ewc_penalty_to_gradients)
