import json
from collections import defaultdict
from .utilities import filter_string, ensure_valid_key, split_if_string

_RESERVED_KEY = '#'  # Reserved key for node data

class WordTrie:
    def __init__(self, weights=False, word_filter=False, text_filter=False):
        self.root = defaultdict(dict)
        self.weights = weights
        self.word_filter = word_filter
        self.text_filter = text_filter

    def _traverse_and_collect_phrases(self, node, path, phrase_dict, next_id):
        if _RESERVED_KEY in node:
            phrase_info = {'phrase': ' '.join(path)}
            phrase_info['value'] = node[_RESERVED_KEY]['value']
            if self.weights:
                phrase_info['weight'] = node[_RESERVED_KEY].get('weight', None)
            phrase_dict[next_id[0]] = phrase_info
            next_id[0] += 1
        for child in node:
            if child != _RESERVED_KEY:
                self._traverse_and_collect_phrases(node[child], path + [child.lstrip(_RESERVED_KEY)], phrase_dict, next_id)

    def _process_match(self, node, match, values, return_nodes=False):
        if _RESERVED_KEY in node:
            match_data = node[_RESERVED_KEY]
            result = (' '.join(match), match_data['value'])
            if self.weights:
                result += (match_data['weight'],)
            if return_nodes:
                values.append(result)
            else:
                values.append(match_data['value'])

    def add(self, word, value, weight=None):
        if self.weights and weight is None:
            raise ValueError("Weight is required when weights are enabled.")
        if self.word_filter:
            word = filter_string(word)
        node = self.root
        for char in split_if_string(word):
            node = node.setdefault(ensure_valid_key(char), {})
        node_data = {'value': value, 'weight': weight} if self.weights else {'value': value}
        node[_RESERVED_KEY] = node_data

    def remove_by_string(self, phrase):
        def _remove(node, word, index=0):
            word = split_if_string(word)
            for char in word:
                if char not in node:
                    raise ValueError(f"Word '{word}' not found in trie.")
                node = node[char]
            if _RESERVED_KEY not in node:
                raise ValueError(f"Word '{word}' not found in trie.")
            del node[_RESERVED_KEY]
            return node
        try:
            phrase = filter_string(phrase)
            _remove(self.root, phrase)
        except ValueError as e:
            print(e)

    def search(self, text, return_nodes=False, return_meta=False):
        if self.text_filter:
            text = filter_string(text) if isinstance(text, str) else [filter_string(item) for item in text]
        node, match, values, found_words = self.root, [], [], []
        for word in map(ensure_valid_key, split_if_string(text)) if isinstance(text, str) else text:
            if word not in node:
                self._process_match(node, match, values, return_nodes)
                if match and _RESERVED_KEY in node:
                    found_word = ' '.join(match)
                    found_words.append(found_word)
                node = self.root
                match = []
            else:
                node = node[word]
                match.append(word)
        if match and _RESERVED_KEY in node:
            found_word = ' '.join(match)
            found_words.append(found_word)
        self._process_match(node, match, values, return_nodes)
        return values if not return_meta else (values, {'match_length': len(values), 'match_ratio': len(found_words) / len(split_if_string(text)) if text else 0})

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.root, f, indent=2)

    def from_json(self, filename):
        with open(filename) as f:
            self.root = json.load(f)

    def visualize(self, node=None, indent="", last=True):
        if node is None:
            node = self.root
        children = list(node.keys())
        for i, child in enumerate(children):
            if child == _RESERVED_KEY:
                print(f"{indent}└── [END: {node[child]['value']}]")
            else:
                prefix = "└── " if last else "├── "
                print(f"{indent}{prefix}{child}")
                next_indent = "    " if last else "│   "
                self.visualize(node[child], indent + next_indent, i == len(children) - 1)