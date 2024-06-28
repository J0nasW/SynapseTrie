import unittest
from scipy.sparse import csr_matrix
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Only for local testing
from SynapseTrie import WordTrie

class TestWordTrie(unittest.TestCase):
    def setUp(self):
        # Create a new trie before each test
        self.trie = WordTrie(word_filter=True, text_filter=True)
        
    # Add method tests
    def test_add_single_word(self):
        """Test adding a single word."""
        _RESERVED_KEY = "#"
        
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        # Retrieve the ID assigned to 'hello'
        added_word_id = None
        for id, path in self.trie.id_to_node_path.items():
            reconstructed_word = ''.join(path)
            if reconstructed_word == "hello":
                added_word_id = id
                break
        
        self.assertIsNotNone(added_word_id, "Word 'hello' was not added correctly.")
        # Verify the node at the end of this path has the correct payload and weight
        node = self.trie.root
        for char in self.trie.id_to_node_path[added_word_id]:
            node = node[char]
        
        self.assertIn(_RESERVED_KEY, node)
        self.assertEqual(node[_RESERVED_KEY]['payload'], {"info": "greeting"})
        self.assertEqual(node[_RESERVED_KEY]['weight'], 1.0)

    def test_add_duplicate_word(self):
        """Test adding a duplicate word throws an exception."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        with self.assertRaises(ValueError):
            self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
            
    def test_add_multiple_words(self):
        """Test adding multiple words as a list."""
        self.trie.add_bulk(["hello", "world"], weight_list=[1.0, 2.0], payload_list=[{"info": "greeting"}, {"info": "planet"}])
        # Retrieve the ID assigned to 'hello'
        added_word_id = None
        for id, path in self.trie.id_to_node_path.items():
            reconstructed_word = ''.join(path)
            if reconstructed_word == "hello":
                added_word_id = id
                break
            
        self.assertIsNotNone(added_word_id, "Word 'hello' was not added correctly.")
        # Verify the node at the end of this path has the correct payload and weight
        node = self.trie.root
        for char in self.trie.id_to_node_path[added_word_id]:
            node = node[char]
            
        self.assertIn('#', node)
        self.assertEqual(node['#']['payload'], {"info": "greeting"})
        self.assertEqual(node['#']['weight'], 1.0)
        
        # Retrieve the ID assigned to 'world'
        added_word_id = None
        for id, path in self.trie.id_to_node_path.items():
            reconstructed_word = ''.join(path)
            if reconstructed_word == "world":
                added_word_id = id
                break
            
        self.assertIsNotNone(added_word_id, "Word 'world' was not added correctly.")
        # Verify the node at the end of this path has the correct payload and weight
        node = self.trie.root
        for char in self.trie.id_to_node_path[added_word_id]:
            node = node[char]
            
        self.assertIn('#', node)
        self.assertEqual(node['#']['payload'], {"info": "planet"})
        self.assertEqual(node['#']['weight'], 2.0)
        
    def test_add_multiple_words_no_weights_or_payloads(self):
        """Test adding multiple words without weights or payloads."""
        self.trie.add_bulk(["hello", "world"])
        # Retrieve the ID assigned to 'hello'
        added_word_id = None
        for id, path in self.trie.id_to_node_path.items():
            reconstructed_word = ''.join(path)
            if reconstructed_word == "hello":
                added_word_id = id
                break
            
        self.assertIsNotNone(added_word_id, "Word 'hello' was not added correctly.")
        

    # Remove method tests
    def test_remove_by_string(self):
        """Test removing a word by string."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        self.trie.remove("hello")
        self.assertFalse('hello' in self.trie.id_to_node_path.values())

    def test_remove_by_id(self):
        """Test removing a word by ID."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        phrase_id = list(self.trie.id_to_node_path.keys())[0]
        self.trie.remove(phrase_id)
        self.assertFalse(phrase_id in self.trie.id_to_node_path)

    def test_remove_nonexistent(self):
        """Test removing a non-existent entry does not raise an error."""
        self.trie.remove("world")  # Should not raise

    # Get info tests
    def test_get_info_by_string(self):
        """Test fetching info by string."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        result = self.trie.get_info("hello")
        self.assertEqual(result['payload'], {"info": "greeting"})

    def test_get_info_by_id(self):
        """Test fetching info by ID."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        phrase_id = list(self.trie.id_to_node_path.keys())[0]
        result = self.trie.get_info(phrase_id)
        self.assertEqual(result['payload'], {"info": "greeting"})

    def test_get_info_nonexistent(self):
        """Test fetching info for a non-existent word or ID returns None."""
        result = self.trie.get_info("nonexistent")
        self.assertIsNone(result)
        result = self.trie.get_info(999)
        self.assertIsNone(result)
        
    # Search method tests
    def test_simple_search(self):
        """Test simple search for a single word."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        result = self.trie.search("hello", return_type='word')
        self.assertIn("hello", result[0])  # Result is now a list

    def test_search_return_types(self):
        """Test different return types for search results."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        # Test for 'word' return type
        result = self.trie.search("hello", return_type='word')
        self.assertEqual(result, ["hello"])
        # Test for 'id' return type
        result_id = self.trie.search("hello", return_type='id')
        phrase_id = list(self.trie.id_to_node_path.keys())[0]
        self.assertEqual(result_id, [phrase_id])
        # Test for 'payload' return type
        result_payload = self.trie.search("hello", return_type='payload')
        self.assertEqual(result_payload, [{"info": "greeting"}])

    def test_search_with_meta_information(self):
        """Test search function with return of meta information."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        self.trie.add("world", weight=2.0, payload={"info": "planet"})
        result, meta = self.trie.search("hello world", return_meta=True, return_type='word')
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertEqual(meta['match_length'], 2)
        self.assertGreater(meta['match_ratio'], 0)
        self.assertGreater(meta['mean_weight'], 0)

    def test_search_nonexistent_word(self):
        """Test searching for a word not in the trie."""
        result = self.trie.search("nonexistent")
        self.assertEqual(result, [])  # Should return an empty list for nonexistent words

    def test_search_list_of_strings(self):
        """Test searching for a list of strings."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        self.trie.add("world", weight=2.0, payload={"info": "planet"})
        texts = ["hello", "world", "unknown"]
        results = self.trie.search(texts)
        self.assertEqual(len(results), 2)  # Expecting three results: for 'hello', 'world', and not for 'unknown'
        
    def test_get_feature_names(self):
        """Test the get_feature_names function."""
        self.trie.add("hello", weight=1.0, payload={"info": "greeting"})
        self.trie.add("world", weight=2.0, payload={"info": "planet"})
        self.trie.add("goodbye", weight=3.0, payload={"info": "farewell"})
        expected = ['hello', 'world', 'goodbye']
        self.assertEqual(self.trie.get_feature_names(), expected)
        
    def test_build_phrase_document_matrix(self):
        """Test the build_phrase_document_matrix function."""
        self.trie.add("test phrase", weight=1.0, payload={"info": "greeting"})
        self.trie.add("ultimative test", weight=2.0, payload={"info": "farewell"})
        self.trie.add("awesome", weight=3.0, payload={"info": "planet"})
        documents = ['here we have a test phrase', 'that is awesome', 'this ultimative test yields a test phrase', '']
        matrix = self.trie.build_phrase_document_matrix(documents)
        print(f"Matrix:\n{matrix.toarray()}")
        expected = csr_matrix([[1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0]])
        self.assertTrue((matrix != expected).nnz == 0)
        
    def test_length(self):
        """Test the length function."""
        self.trie.add("test phrase", weight=1.0, payload={"info": "greeting"})
        self.trie.add("ultimative test", weight=2.0, payload={"info": "farewell"})
        self.trie.add("awesome", weight=3.0, payload={"info": "planet"})
        self.assertEqual(self.trie.length(), 3)

if __name__ == '__main__':
    unittest.main(verbosity=2)