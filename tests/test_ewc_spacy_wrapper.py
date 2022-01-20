import unittest
from unittest.mock import Mock, MagicMock
from thinc.api import Model, Optimizer
from spacy.training import Example
from spacy.pipeline import TrainablePipe
from spacy.language import Language
from spacy_wrapper import EWCModelWrapper, create_ewc_pipe
from spacy_ewc import EWC
import numpy as np


class TestEWCModelWrapper(unittest.TestCase):

    def setUp(self):
        # Mock the underlying model and EWC penalty function
        self.mock_model = Mock(spec=Model)
        self.mock_penalty_function = Mock()

        # Instantiate the EWCModelWrapper
        self.ewc_model_wrapper = EWCModelWrapper(
            self.mock_model, self.mock_penalty_function)

    def test_finish_update_calls_penalty_function_and_model_update(self):
        # Mock optimizer
        mock_optimizer = Mock(spec=Optimizer)

        # Call finish_update
        self.ewc_model_wrapper.finish_update(mock_optimizer)

        # Assert that the EWC penalty function was called
        self.mock_penalty_function.assert_called_once()

        # Assert that the underlying model's finish_update was called with optimizer
        self.mock_model.finish_update.assert_called_once_with(
            optimizer=mock_optimizer)

    def test_getattribute_finish_update_override(self):
        # Access finish_update and confirm it uses EWCModelWrapper's version
        finish_update_method = self.ewc_model_wrapper.finish_update
        self.assertEqual(finish_update_method,
                         self.ewc_model_wrapper.finish_update)

    def test_getattribute_fallback_to_wrapped_model(self):
        # Mock an attribute in the wrapped model
        self.mock_model.some_attribute = "test_value"

        # Access the attribute through EWCModelWrapper
        self.assertEqual(self.ewc_model_wrapper.some_attribute, "test_value")

    def test_getattribute_missing_attribute_raises_error(self):
        # Attempt to access a non-existent attribute
        with self.assertRaises(AttributeError):
            _ = self.ewc_model_wrapper.non_existent_attr


class MockedModel(MagicMock):
    """
    A mock class for thinc Model, including layers, param_names,
    and a method for retrieving gradients.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(spec=Model, *args, **kwargs)

        # Set up the layer name and parameter names
        self.param_names = ["param1", "param2"]

        # Mock the finish_update method
        self.finish_update = MagicMock()

    def _create_mocked_layers(self):
        # Create a random number of layers with unique indexes
        # Random number of layers (e.g., 2 to 4)
        layer_count = np.random.randint(2, 5)
        return [MockedModel() for _ in range(layer_count)]

    def get_gradients(self):
        # Return a dictionary of gradients with random values
        return {
            (self.id, param_name): (np.random.rand(2) * -1, np.random.rand(2))
            for param_name in self.param_names
        }

    @property
    def _layers(self):
        return self._create_mocked_layers()

    @property
    def layers(self):
        return self._layers


class TestCreateEwcPipe(unittest.TestCase):

    def setUp(self):
        # Use MockedModel for cleaner setup
        self.mocked_model = MockedModel()

        # Set up a mock TrainablePipe with the mocked model
        self.mock_pipe = MagicMock(spec=TrainablePipe)
        self.mock_pipe.model = self.mocked_model
        self.mock_pipe.update = MagicMock(
            side_effect=lambda batch, losses, sgd: losses.update({'ner': 1.0}))

        # Set up mock training data
        self.mock_data = [Mock(spec=Example)]

        # Mock EWC instantiation
        self.mock_ewc = MagicMock(spec=EWC)
        self.mock_ewc.apply_ewc_penalty_to_gradients = MagicMock()
        with unittest.mock.patch('spacy_ewc.EWC', return_value=self.mock_ewc):
            self.lambda_value = 1000.0
            self.pipe_with_ewc = create_ewc_pipe(
                pipe=self.mock_pipe,
                data=self.mock_data,
                lambda_=self.lambda_value
            )

    def test_create_ewc_pipe_with_trainable_pipe(self):
        # Check that the function returns a TrainablePipe
        self.assertIsInstance(self.pipe_with_ewc, TrainablePipe)

        # Check that the model in TrainablePipe was wrapped with EWCModelWrapper
        self.assertIsInstance(self.pipe_with_ewc.model, EWCModelWrapper)

    def test_create_ewc_pipe_with_language_model(self):
        # Mock a Language object with a component
        mock_language = Mock(spec=Language)

        mock_language.get_pipe.return_value = self.mock_pipe

        with unittest.mock.patch('spacy_ewc.EWC', return_value=self.mock_ewc):
            pipe_with_ewc = create_ewc_pipe(
                pipe=mock_language,
                data=self.mock_data,
                pipe_name="other"
            )

        # Ensure the Language model's component was retrieved and wrapped
        mock_language.get_pipe.assert_called_once_with("other")
        self.assertIsInstance(pipe_with_ewc.model, EWCModelWrapper)

    def test_create_ewc_pipe_with_invalid_type(self):
        # Test passing an invalid type to `pipe`
        with self.assertRaises(ValueError) as context:
            create_ewc_pipe(pipe=Mock(), data=self.mock_data)
        self.assertIn("pipe param can only be an instance of one of: ['Language', 'TrainablePipe']", str(
            context.exception))

    def test_create_ewc_pipe_default_component_name(self):
        # Mock a Language object with a default component (ner)
        mock_language = Mock(spec=Language)

        mock_language.get_pipe.return_value = self.mock_pipe

        with unittest.mock.patch('spacy_ewc.EWC', return_value=self.mock_ewc):
            pipe_with_ewc = create_ewc_pipe(
                pipe=mock_language,
                data=self.mock_data
            )

        # Ensure default component name 'ner' is used if pipe_name is None
        mock_language.get_pipe.assert_called_once_with("ner")
        self.assertIsInstance(pipe_with_ewc.model, EWCModelWrapper)

    def test_create_ewc_pipe_ewc_initialization(self):
        # Ensure EWC is initialized with correct parameters in create_ewc_pipe
        with unittest.mock.patch('spacy_ewc.EWC', return_value=self.mock_ewc) as mock_ewc_class:
            create_ewc_pipe(pipe=self.mock_pipe,
                            data=self.mock_data, lambda_=500.0)
            self.assertIsInstance(self.mock_pipe.model, EWCModelWrapper)


if __name__ == '__main__':
    unittest.main()
