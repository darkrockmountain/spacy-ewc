import unittest
import spacy
from spacy.training import Example
from data_examples.original_spacy_labels import original_spacy_labels
from data_examples.training_data import training_data
from spacy_wrapper.ewc_spacy_wrapper import EWCPipeWrapper
from utils.extract_labels import extract_labels
from thinc.api import Adam


class TestEWCIntegrationWithSpacyWrapper(unittest.TestCase):

    def setUp(self):
        # Initialize the NLP pipeline with an NER component
        self.nlp = spacy.load("en_core_web_sm")

        # Add initial entity labels
        ner = self.nlp.get_pipe("ner") if self.nlp.has_pipe(
            "ner") else self.nlp.add_pipe("ner")

        self.training_data = training_data
        self.original_spacy_labels = original_spacy_labels

        self.original_labels = extract_labels(self.original_spacy_labels)
        self.training_labels = extract_labels(self.training_data)

        for label in self.training_labels:
            ner.add_label(label)

        self.test_sentence = "Elon Musk founded Space-X in 2002 as the CEO and lead engineer, investing approximately $100 million of his own money into the company, which was initially based in El Segundo, California, before moving to Hawthorne, California."

        # Initialize the EWC instance with retraining

        ner = EWCPipeWrapper(ner, [Example.from_dict(self.nlp.make_doc(
            text), annotations) for text, annotations in self.original_spacy_labels])

    def test_train_nlp_with_ewc_integration(self):
        # Define a dictionary to hold the losses

        original_ner_doc = self.nlp(self.test_sentence)

        sgd = None
        for e in range(10):

            # Create minibatches using spaCy's utility function
            batches = spacy.util.minibatch(
                [
                    Example.from_dict(self.nlp.make_doc(text), ann)
                    for text, ann in self.training_data
                ],
                size=spacy.util.compounding(4.0, 32.0, 1.001)
            )
            for batch in batches:
                losses = {}
                # Train the NLP model with EWC applied
                self.nlp.update(examples=batch, sgd=sgd, losses=losses)

        # Ensure losses were recorded for NER training
        self.assertIn(
            "ner", losses, "Losses should include 'ner' key after training.")
        self.assertGreater(
            losses["ner"], 0, "NER losses should be greater than zero after training.")

        ewc_ner_doc = self.nlp(self.test_sentence)
        self.assertEqual(original_ner_doc.text, ewc_ner_doc.text)


        original_ner_labels = set(
            [ent.label_ for ent in original_ner_doc.ents])
        ewc_ner_labels = set([ent.label_ for ent in ewc_ner_doc.ents])

        self.assertNotEqual(original_ner_labels, ewc_ner_labels)

        not_changed_labels = ewc_ner_labels.difference(self.training_labels)

        self.assertEqual(ewc_ner_labels.intersection(
            self.training_labels), self.training_labels)

        self.assertGreater(len(not_changed_labels), 0)
        self.assertLess(len(not_changed_labels), len(original_ner_labels))
        self.assertLess(len(not_changed_labels), len(ewc_ner_labels))

    def tearDown(self):
        del self.nlp


if __name__ == "__main__":
    unittest.main()
