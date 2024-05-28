import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    yaml._warnings_enabled["YAMLLoadWarning"] = False
except (KeyError, AttributeError, TypeError) as e:
    pass

import json
from tqdm import tqdm
from scipy.sparse import lil_matrix
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
        """
        Helper function to recursively traverse the trie and collect phrases.
        Enhanced to optionally collect weights if enabled.
        """
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
        """Process a match in the trie."""
        if _RESERVED_KEY in node:
            match_data = node[_RESERVED_KEY]
            result = (' '.join(match), match_data['value'])
            if self.weights:
                result += (match_data['weight'],)
            if return_nodes:
                values.append(result)
            else:
                values.append(match_data['value'])
                
    def _get_match_weight(self, node, match):
        """Retrieves the weight associated with a match (if weights are enabled)."""
        return node.get(_RESERVED_KEY, {}).get('weight') if _RESERVED_KEY in node else None
    
    def _get_match_id(self, node, match):
        """Retrieves the ID associated with a match."""
        return node.get(_RESERVED_KEY, {}).get('id') if _RESERVED_KEY in node else None

    # =====================================
    # Adding Words/Phrases Methods
    # =====================================
    
    def add(self, word, value, weight=None):
        """Add a word or phrase to the trie."""
        if self.weights and weight is None:
            raise ValueError("Weight is required when weights are enabled.")
        if self.word_filter:
            word = filter_string(word)
        node = self.root
        for char in split_if_string(word):
            node = node.setdefault(ensure_valid_key(char), {})
        node_data = {'value': value, 'weight': weight} if self.weights else {'value': value}
        node[_RESERVED_KEY] = node_data
        
    def add_list(self, words_list, value_list, weight_list=None):
        """Add multiple words or phrases to the trie."""
        if self.weights and (weight_list is None or len(words_list) != len(weight_list)):
            raise ValueError("Weight list is required and must match the length of words_list when weights are enabled.")
        for i, word in enumerate(words_list):
            self.add(word, value_list[i], weight_list[i] if self.weights else None)
            
    def add_df(self, df, column, value_column=None, weight_column=None):
        """Add words from a pandas DataFrame."""
        if value_column is None:
            value_column = column
        weights = df[weight_column].tolist() if self.weights and weight_column else None
        self.add_bulk(df[column].tolist(), df[value_column].tolist(), weights)
    
    # =====================================
    # Removing Words/Phrases Methods
    # ===================================== 

    def remove_by_string(self, phrase):
        """Remove a phrase from the trie by its string value."""
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
            
    def remove_by_id(self, phrase_id):
        """Remove a phrase from the trie by its value"""
        phrases_with_ids = self.get_phrases_with_ids()
        phrase_info = phrases_with_ids.get(phrase_id)
        if phrase_info:
            self.remove_by_string(phrase_info['phrase'])
        else:
            raise ValueError(f"Phrase with ID {phrase_id} not found in trie.")
        
    def remove_by_value(self, phrase_value):
        """Remove a phrase from the trie by its value"""
        phrases_with_ids = self.get_phrases_with_ids()
        for phrase_id, phrase_info in phrases_with_ids.items():
            if phrase_info['value'] == phrase_value:
                self.remove_by_string(phrase_info['phrase'])
                return
        raise ValueError(f"Phrase with value {phrase_value} not found in trie.")
            
    def remove_bulk(self, items, by_id=False):
        """Remove multiple phrases from the trie by strings or IDs, optimized for large datasets."""
        items_set = set(items)  # Convert list to set for faster lookup

        if by_id:
            phrases_with_ids = self.get_phrases_with_ids()
            for phrase_id in items_set:
                phrase_info = phrases_with_ids.get(phrase_id)
                if phrase_info:
                    self.remove_by_string(phrase_info['phrase'])
        else:
            for phrase in items_set:
                self.remove_by_string(phrase)
                
    # =====================================
    # Phrase ID, Value, Weight Handling
    # =====================================
                
    def get_phrases_with_ids(self):
        """Retrieve all phrases with their corresponding IDs by traversing the trie."""
        phrase_dict = {}
        self._traverse_and_collect_phrases(self.root, [], phrase_dict, [0])
        return dict(sorted(phrase_dict.items()))
    
    def get_phrase_by_id(self, phrase_value):
        """Retrieve a phrase by its value"""
        phrases_with_ids = self.get_phrases_with_ids()
        for phrase_id, phrase_info in phrases_with_ids.items():
            if phrase_info['value'] == phrase_value:
                return phrase_info['phrase']
        return None
    
    def get_weight_by_id(self, phrase_value):
        """Retrieve the weight of a phrase by its ID, applicable when weights are enabled."""
        if not self.weights:
            return None  # Return None if weights are not used in the trie
        
        phrases_with_ids = self.get_phrases_with_ids()
        for phrase_id, phrase_info in phrases_with_ids.items():
            if phrase_info['value'] == phrase_value:
                return phrase_info['weight'] if phrase_info else None
            
    # =====================================
    # Searching Words/Phrases Methods
    # =====================================

    def search(self, text, return_nodes=False, return_meta=False):
        """Search for phrases or words in the trie.

        Args:
            text (str or list): The text or a list of phrases to search within.

        Returns:
            list: A list of matched values, nodes, words, IDs, or weights depending on the arguments.
        """

        if self.text_filter:
            text = self._filter_string(text) if isinstance(text, str) else [self._filter_string(item) for item in text]

        node, match, values, found_words = self.root, [], [], []
        for word in map(self._ensure_valid_key, self._split_if_string(text)) if isinstance(text, str) else text:
            if word not in node:  # Word not found
                self._process_match(node, match, values, return_nodes) 
                if match and _RESERVED_KEY in node:
                    found_word = ' '.join(match)
                    found_words.append(found_word)  
                node = self.root  # Reset to root level 
                match = []  # Reset match
            else:
                node = node[word]
                match.append(word)
                
        # Final check after processing the entire text to ensure we capture the last match if any
        if match and _RESERVED_KEY in node: 
            found_word = ' '.join(match)
            found_words.append(found_word)
        self._process_match(node, match, values, return_nodes)
            
        if self.weights:
            weights = [self.get_weight_by_id(value) for value in values]
            
        if return_meta:
            match_length = len(values) # Count of matches
            match_ratio = sum(len(match.split()) for match in found_words) / len(self._split_if_string(text)) if text else 0 # Count of matching words / total words
            if self.weights:
                mean_weight = sum(weights) / match_length
                
        if self.weights:
            if return_meta:
                return [(value, found_word, self.get_weight_by_id(value)) 
            for value, found_word in zip(values, found_words)], {'match_length': match_length, 'match_ratio': match_ratio, 'mean_weight': mean_weight}
            else:
                return [(value, found_word, self.get_weight_by_id(value)) 
            for value, found_word in zip(values, found_words)]
        else:
            if return_meta:
                return [(value, found_word) for value, found_word in zip(values, found_words)], {'match_length': match_length, 'match_ratio': match_ratio}
            else:
                return [(value, found_word) for value, found_word in zip(values, found_words)]
    
    def search_list(self, words_list):
        """Return a list of values that match the words in words_list."""
        results = []
        for text in words_list:
            results.extend(self.search(text))
        return results
    
    def search_df(self, df, column):
        """Return a list of values that match the words in a pandas DataFrame."""
        return self.search_list(df[column].tolist())

    # =====================================
    # Matrix Operations and Semantic Networks for the Trie
    # =====================================
    
    def build_phrase_document_matrix(self, documents):
        """
        Build a document-phrase matrix where each entry (i, j) represents
        the frequency of phrase j in document i, with progress displayed via tqdm.
        """
        # Create a mapping from phrases to integer IDs
        phrase_to_id = {phrase: idx for idx, phrase in enumerate(self.get_feature_names())}

        # Initialize a sparse matrix
        matrix = lil_matrix((len(documents), len(phrase_to_id)), dtype=int)

        # Process each document, with tqdm tracking progress
        for doc_id, doc in tqdm(enumerate(documents), total=len(documents), desc="Building Matrix"):
            # Convert document to a filtered string if necessary
            doc_text = doc if not self.text_filter else self._filter_string(doc)
            for word in self._split_if_string(doc_text):
                # Ensure valid key and check if it exists in the trie
                word = self._ensure_valid_key(word)
                if word in phrase_to_id:
                    matrix[doc_id, phrase_to_id[word]] += 1

        return matrix.tocsr()  # Convert to CSR for efficient arithmetic and matrix vector operations
    
    def get_feature_names(self):
        """Retrieve sorted list of phrases stored in the trie."""
        def collect_phrases(node, prefix=''):
            if _RESERVED_KEY in node:
                yield prefix
            for char, next_node in node.items():
                if char != _RESERVED_KEY:
                    yield from collect_phrases(next_node, prefix + char)
    
    # Counting and length of the trie
    
    def length(self):
        """Return the number of unique phrases in the trie."""
        def count_nodes(node):
            count = 1 if _RESERVED_KEY in node else 0
            for child in node:
                if child != _RESERVED_KEY:
                    count += count_nodes(node[child])
            return count
        return count_nodes(self.root)
        
    # =====================================
    # Loading and Saving the whole TRIE
    # =====================================

    def to_json(self, filename):
        """Save the trie to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.root, f, indent=2)

    def from_json(self, filename):
        """Load the trie from a JSON file."""
        with open(filename) as f:
            self.root = json.load(f)