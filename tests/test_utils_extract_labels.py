import unittest
from spacy_ewc.utils.extract_labels import extract_labels


class TestExtractLabels(unittest.TestCase):
    def test_basic_functionality(self):
        # Test with multiple sentences and multiple labels
        data = [
            (
                "John bought a new car yesterday.",
                {"entities": [(0, 4, "PERSON"), (22, 31, "DATE")]},
            ),
            (
                "Microsoft announced its new software in San Francisco.",
                {"entities": [(0, 9, "ORG"), (40, 53, "GPE")]},
            ),
        ]
        expected_labels = {"PERSON", "DATE", "ORG", "GPE"}
        result = extract_labels(data)
        self.assertEqual(result, expected_labels)

    def test_empty_data(self):
        # Test with empty list
        data = []
        expected_labels = set()
        result = extract_labels(data)
        self.assertEqual(result, expected_labels)

    def test_no_entities_key(self):
        # Test when "entities" key is missing in some entries
        data = [
            ("A sentence without entities.", {}),
            ("Another sentence without entities.", {"entities": []}),
        ]
        expected_labels = set()
        result = extract_labels(data)
        self.assertEqual(result, expected_labels)

    def test_empty_entities_list(self):
        # Test when "entities" is an empty list
        data = [("A sentence with empty entities list.", {"entities": []})]
        expected_labels = set()
        result = extract_labels(data)
        self.assertEqual(result, expected_labels)

    def test_multiple_occurrences_of_same_label(self):
        # Test when the same label appears multiple times in different entities
        data = [
            (
                "Alice met Bob in New York.",
                {
                    "entities": [
                        (0, 5, "PERSON"),
                        (9, 12, "PERSON"),
                        (16, 24, "GPE"),
                    ]
                },
            ),
            (
                "Bob moved to New York.",
                {"entities": [(0, 3, "PERSON"), (14, 22, "GPE")]},
            ),
        ]
        expected_labels = {"PERSON", "GPE"}
        result = extract_labels(data)
        self.assertEqual(result, expected_labels)

    def test_mixed_labels(self):
        # Test with mixed and nested entities
        data = [
            (
                "A complex sentence.",
                {"entities": [(0, 1, "LOC"), (2, 5, "TIME"), (6, 10, "MISC")]},
            ),
            (
                "Another with overlapping labels.",
                {"entities": [(1, 4, "ORG"), (5, 8, "PRODUCT")]},
            ),
        ]
        expected_labels = {"LOC", "TIME", "MISC", "ORG", "PRODUCT"}
        result = extract_labels(data)
        self.assertEqual(result, expected_labels)

    def test_single_entity(self):
        # Test with only one entity in the list
        data = [("This has only one entity.", {"entities": [(5, 9, "EVENT")]})]
        expected_labels = {"EVENT"}
        result = extract_labels(data)
        self.assertEqual(result, expected_labels)

    def test_unexpected_data_format(self):
        # Test with unexpected data format (not a tuple)
        data = [{"text": "This format is unexpected.", "entities": [(0, 4, "ORG")]}]
        with self.assertRaises(Exception):
            extract_labels(data)


if __name__ == "__main__":
    unittest.main()
