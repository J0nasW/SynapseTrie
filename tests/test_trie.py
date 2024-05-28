import unittest
# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Only for local testing
from SynapseTrie import WordTrie

class TestWordTrie(unittest.TestCase):
    def setUp(self):
        # Create a new trie before each test
        self.trie = WordTrie(weights=True, word_filter=True, text_filter=True)

    def test_add_single_word(self):
        # Test adding a single word
        self.trie.add("hello", value=1, weight=0.5)
        self.assertIn('#', self.trie.root['hello'])

    def test_search_single_word(self):
        # Ensure that searching for an added word returns correct results
        self.trie.add("world", value=2, weight=1.0)
        result = self.trie.search("world", return_nodes=True)
        self.assertEqual(result[0], ('world', 2, 1.0))

    def test_remove_single_word(self):
        # Test removing an added word
        self.trie.add("remove", value=3, weight=0.5)
        self.trie.remove_by_string("remove")
        result = self.trie.search("remove")
        self.assertEqual(len(result), 0)

    def test_weight_required_on_add_with_weights(self):
        # Ensure that adding a word without a weight raises an error when weights are enabled
        with self.assertRaises(ValueError):
            self.trie.add("test", value=4)

    def test_search_nonexistent_word(self):
        # Search for a word that does not exist
        result = self.trie.search("nonexistent")
        self.assertEqual(len(result), 0)

    def test_case_insensitivity_and_filtering(self):
        # Test case insensitivity and filtering
        self.trie.add("Hallo!", value=5, weight=1.5)
        result = self.trie.search("hallo", return_nodes=True)
        self.assertEqual(result[0], ('hallo', 5, 1.5))

    def test_return_nodes_and_meta_data(self):
        # Test return of meta data and node details
        self.trie.add("test case", value=6, weight=2.0)
        result, meta = self.trie.search("test case", return_nodes=True, return_meta=True)
        self.assertEqual(result[0], ('test case', 6, 2.0))
        self.assertEqual(meta, {'match_length': 1, 'match_ratio': 0.5})

    def test_json_save_load(self):
        # Test saving to and loading from a JSON file
        import tempfile
        import os
        self.trie.add("json", value=7, weight=2.5)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.trie.to_json(temp_file.name)
        self.trie.from_json(temp_file.name)
        result = self.trie.search("json", return_nodes=True)
        self.assertEqual(result[0], ('json', 7, 2.5))
        os.unlink(temp_file.name)  # Clean up the temp file

if __name__ == '__main__':
    unittest.main(verbosity=2)