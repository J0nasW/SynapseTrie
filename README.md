# Synapse Trie

Synapse Trie is a Python package for efficiently storing and searching phrases using a trie data structure, with additional features like weights, text filtering, and more.

## Installation

Install directly using pip:

```bash
pip install git+https://github.com/yourusername/synapse_trie.git
```

## Usage

```python
from synapse_trie import SynapseTrie

trie = SynapseTrie()

trie.add("hello")

print(trie.search("hello")) # True
```