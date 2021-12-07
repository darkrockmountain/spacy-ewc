import unittest
from spacy.lang.en import English
from spacy_ewc import EWC


class TestEWC(unittest.TestCase):

    def setUp(self):
        # Initialize the nlp pipeline
        self.nlp = English()
        self.nlp.add_pipe("ner")

        # Mock training data for the initial task
        self.train_data = [
            ("This is a sample sentence", {"entities": [(10, 16, "LABEL")]}),
            ("Another example sentence", {"entities": [(8, 15, "LABEL")]}),
        ]

        # Create an instance of the EWC class
        self.ewc = EWC(self.nlp, self.train_data, re_train_model=True)

    def test_initial_fisher_matrix_not_none(self):
        # Ensure that the Fisher matrix is computed after training on initial task
        self.assertIsNotNone(
            self.ewc.fisher_matrix, "Fisher matrix should not be None after initialization.")

    def test_theta_star_initialization(self):
        # Verify theta_star is initialized correctly after initial training
        self.assertIsInstance(self.ewc.theta_star, dict,
                              "theta_star should be a dictionary.")
        self.assertGreater(len(self.ewc.theta_star), 0,
                           "theta_star should contain parameters.")

    def test_capture_current_params(self):
        # Test if get_current_params method correctly captures model parameters
        current_params = self.ewc.get_current_params()
        self.assertIsInstance(
            current_params, dict, "get_current_params should return a dictionary.")
        self.assertGreater(len(current_params), 0,
                           "Captured current_params should contain parameters.")

    def test_fisher_matrix_computation(self):
        # Test if the Fisher matrix is correctly computed
        fisher_matrix = self.ewc._compute_fisher_matrix(self.train_data)
        self.assertIsInstance(fisher_matrix, dict,
                              "Fisher matrix should be a dictionary.")
        self.assertGreater(len(fisher_matrix), 0,
                           "Fisher matrix should contain computed values.")

    def test_loss_penalty_calculation(self):
        # Test the loss penalty calculation function
        penalty = self.ewc.loss_penalty()
        self.assertIsInstance(
            penalty, float, "loss_penalty should return a float.")
        self.assertGreaterEqual(
            penalty, 0.0, "loss_penalty should be non-negative.")

    def test_ewc_loss_calculation(self):
        # Test the EWC loss function
        mock_task_loss = 0.5
        ewc_loss = self.ewc.ewc_loss(mock_task_loss, lambda_=1000)
        self.assertIsInstance(
            ewc_loss, float, "ewc_loss should return a float.")
        self.assertGreaterEqual(
            ewc_loss, mock_task_loss, "EWC loss should be at least as large as task loss.")



    def tearDown(self):
        del self.ewc
        del self.nlp


if __name__ == '__main__':
    unittest.main()
