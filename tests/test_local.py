import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Only for local testing
from SynapseTrie import WordTrie

trie = WordTrie(word_filter=True, text_filter=True)

trie.add("hello", weight=1.0, payload={"info": "greeting"})
trie.add("world", weight=2.0, payload={"info": "planet"})

# Retrieve the ID assigned to 'hello'
print(f"ID assigned to 'hello': {trie.get_info('hello')}")
# Retrieve the ID assigned to 'world'
print(f"ID assigned to 'world': {trie.get_info('world')}")

# Search for words of the trie in a long text
text = "Hello, world! This is a test of the WordTrie class."
print(f"Text: {text}")

# Search for words in the text
print(f"Found instances: {trie.search(text)}")
print(f"Found instances with metadata: {trie.search(text, return_meta=True)}")
print(f"Found words: {trie.search(text, return_type="word")}")
print(f"Found ids: {trie.search(text, return_type="id")}")
print(f"Found payloads: {trie.search(text, return_type="payload")}")

print(f"Searching for a nonexistent word: {trie.search('nonexistent')}")