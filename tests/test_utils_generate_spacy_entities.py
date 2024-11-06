import unittest
import spacy
from spacy_ewc.utils.generate_spacy_entities import generate_spacy_entities


class TestGenerateSpacyEntities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load a small spaCy model for testing
        cls.nlp = spacy.blank("en")

        # Manually add entity ruler to mock NER behavior for testing purposes
        ruler = cls.nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "PERSON", "pattern": "John"},
            {"label": "ORG", "pattern": "Microsoft"},
            {"label": "GPE", "pattern": "San Francisco"},
            {"label": "DATE", "pattern": "yesterday"},
        ]
        ruler.add_patterns(patterns)

    def test_basic_functionality(self):
        sentences = [
            "John bought a new car yesterday.",
            "Microsoft announced new software in San Francisco.",
        ]
        expected_output = [
            (
                "John bought a new car yesterday.",
                {"entities": [(0, 4, "PERSON"), (22, 31, "DATE")]},
            ),
            (
                "Microsoft announced new software in San Francisco.",
                {"entities": [(0, 9, "ORG"), (36, 49, "GPE")]},
            ),
        ]
        result = generate_spacy_entities(sentences, self.nlp)
        self.assertEqual(result, expected_output)

    def test_empty_sentences_list(self):
        sentences = []
        expected_output = []
        result = generate_spacy_entities(sentences, self.nlp)
        self.assertEqual(result, expected_output)

    def test_no_entities_in_sentence(self):
        sentences = ["There are no entities here."]
        expected_output = [("There are no entities here.", {"entities": []})]
        result = generate_spacy_entities(sentences, self.nlp)
        self.assertEqual(result, expected_output)

    def test_sentence_with_overlapping_entities(self):
        # Testing with an entity pattern that might overlap (e.g., both "John"
        # and "John Smith" as separate entities)
        sentences = ["John Smith went to San Francisco."]
        # Expected to catch "John" as PERSON and "San Francisco" as GPE based
        # on the test setup
        expected_output = [
            (
                "John Smith went to San Francisco.",
                {"entities": [(0, 4, "PERSON"), (19, 32, "GPE")]},
            )
        ]
        result = generate_spacy_entities(sentences, self.nlp)
        self.assertEqual(result, expected_output)

    def test_sentence_with_special_characters(self):
        sentences = ["John bought @ new car yesterday!"]
        expected_output = [
            (
                "John bought @ new car yesterday!",
                {"entities": [(0, 4, "PERSON"), (22, 31, "DATE")]},
            )
        ]
        result = generate_spacy_entities(sentences, self.nlp)
        self.assertEqual(result, expected_output)

    def test_mixed_entity_types(self):
        sentences = ["John visited Microsoft headquarters in San Francisco yesterday."]
        expected_output = [
            (
                "John visited Microsoft headquarters in San Francisco yesterday.",
                {
                    "entities": [
                        (0, 4, "PERSON"),
                        (13, 22, "ORG"),
                        (39, 52, "GPE"),
                        (53, 62, "DATE"),
                    ]
                },
            )
        ]
        result = generate_spacy_entities(sentences, self.nlp)
        self.assertEqual(result, expected_output)

    def test_non_string_input(self):
        sentences = [123, None, "John went to San Francisco."]
        # Only valid string sentences should return entities, others should
        # either be ignored or raise an error
        with self.assertRaises(Exception):
            generate_spacy_entities(sentences, self.nlp)


if __name__ == "__main__":
    unittest.main()
