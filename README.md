# Synapse Trie

Synapse Trie is a Python package for efficiently storing and searching phrases using a trie data structure, with additional features like weights, text filtering, and more.

## Installation

Install directly using pip:

```bash
pip install git+https://github.com/J0nasW/SynapseTrie.git
```

## Usage

```python
from SynapseTrie import SynapseTrie

trie = SynapseTrie()

trie.add("hello")

print(trie.search("hello")) # True
```