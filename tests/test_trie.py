import unittest
from synapsetrie import WordTrie

class TestWordTrie(unittest.TestCase):
    def test_addition(self):
        trie = WordTrie()
        trie.add("hello", 1)
        self.assertTrue(trie.search("hello")[0][0] == "hello")

    def test_word_trie(self):
        # Test initialization
        trie = WordTrie()
        self.assertIsInstance(trie, WordTrie)

        # Test adding words
        trie.add("hello", 1)
        trie.add("world", 2)
        self.assertTrue(trie.search("hello")[0][0] == "hello")
        self.assertTrue(trie.search("world")[0][0] == "world")

        # Test searching for non-existent words
        self.assertFalse(trie.search("nonexistent"))

        # Test adding and searching for words with special characters
        trie.add("hello-world", 3)
        self.assertTrue(trie.search("hello-world")[0][0] == "hello-world")

        # Test adding and searching for words with uppercase letters
        trie.add("Hello", 4)
        self.assertTrue(trie.search("Hello")[0][0] == "Hello")

        # Test searching for words case-insensitively
        self.assertTrue(trie.search("HELLO")[0][0] == "Hello")

        # Test removing words
        trie.remove("Hello")
        self.assertFalse(trie.search("Hello"))

if __name__ == "__main__":
    unittest.main()