from thinc.api import Model
from spacy.training import Example
from thinc.api import get_current_ops


class EWC:
    def __init__(self, nlp,
                 data,
                 re_train_model: bool = False,
                 ):

        self.nlp = nlp
        # To store the Fisher Information Matrix after computing it with the initial task
        self.fisher_matrix = None

        if re_train_model:
            self.__train_initial_model(data)

        # Capture parameters after training on the first task
        self.theta_star = self.get_current_params()

        # Calculate Fisher Information Matrix on the first task
        self.fisher_matrix = self._compute_fisher_matrix(
            data)

    def __train_initial_model(self, train_data):
        # Step 1: Train on Initial Task and Capture Theta Star
        optimizer = self.nlp.initialize()

        examples = [Example.from_dict(self.nlp.make_doc(
            text), annotations) for text, annotations in train_data]

        self.nlp.get_pipe("ner").update(examples, sgd=optimizer)

    def _compute_fisher_matrix(self, data):
        # Prepare the model operations
        ops = get_current_ops()

        # Initialize an empty Fisher Information Matrix
        fisher_matrix = {}

        
        ner = self.nlp.get_pipe("ner")

        # Use the pipe ner update without optimizer to compute gradients without updating model parameters
        examples = [Example.from_dict(self.nlp.make_doc(
            text), annotations) for text, annotations in data]

        ner.update(examples, sgd=None)

        for layer in ner.model.walk():
            # Retrieve the gradient for this parameter
            for (_, name), (_, grad) in layer.get_gradients().items():
                if not name in layer.param_names:
                    # there is no need to store gradient nodes that are not going to be calculated.
                    continue
                # Convert to array for easier manipulation
                grad = ops.asarray(grad).copy() ** 2
         
                fisher_matrix[f"{layer.name}_{name}"] = fisher_matrix.get(
                    name, 0) + grad


        return fisher_matrix

    # Get the current params of the model.
    def get_current_params(self):
        current_params = {}
        ner_model: Model = self.nlp.get_pipe("ner").model
        for layer in ner_model.walk():
            for name in layer.param_names:
                current_params[f"{layer.name}_{name}"] = layer.get_param(
                    name).copy()
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

    def ewc_loss(self, task_loss, lambda_=1000):
        return task_loss + (lambda_ * 0.5*self.loss_penalty())
