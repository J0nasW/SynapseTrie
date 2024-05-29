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
    def __init__(self, word_filter=False, text_filter=False):
        self.root = defaultdict(dict)
        self.word_filter = word_filter
        self.text_filter = text_filter
        self.next_id = 0  # Initialize the ID counter
        self.id_to_node_path = {}  # Maps IDs to node paths

    def _traverse_and_collect_phrases(self, node, path, phrase_dict, next_id):
        if _RESERVED_KEY in node:
            phrase_info = {
                'phrase': ' '.join(path),
                'id': node[_RESERVED_KEY]['id'],
                'weight': node[_RESERVED_KEY].get('weight'),
                'payload': node[_RESERVED_KEY].get('payload')
            }
            phrase_dict[next_id[0]] = phrase_info
            next_id[0] += 1
        for child in node:
            if child != _RESERVED_KEY:
                self._traverse_and_collect_phrases(node[child], path + [child], phrase_dict, next_id)

    def _process_match(self, node, match, values, return_nodes=False):
        if _RESERVED_KEY in node:
            match_data = node[_RESERVED_KEY]
            result = {
                'phrase': ' '.join(match),
                'id': match_data['id'],
                'weight': match_data.get('weight'),
                'payload': match_data.get('payload')
            }
            if return_nodes:
                values.append(result)
            else:
                values.append(match_data['id'])
                
    def _get_match_weight(self, node):
        return node.get(_RESERVED_KEY, {}).get('weight', None) if _RESERVED_KEY in node else None
    
    def _get_match_id(self, node):
        return node.get(_RESERVED_KEY, {}).get('id', None) if _RESERVED_KEY in node else None
    
    def _get_match_payload(self, node):
        return node.get(_RESERVED_KEY, {}).get('payload', None) if _RESERVED_KEY in node else None

    # =====================================
    # Adding Words/Phrases Methods
    # =====================================
    
    def add(self, word, weight=None, payload=None):
        # Check if the word is a string or list, and split if necessary
        if word is not None and not isinstance(word, (str, list)):
            raise ValueError("Word must be a string or a list of strings.")
        """Add a word or phrase to the trie."""
        # Check if weight is int or float, otherwise raise an error
        if weight is not None and not isinstance(weight, (int, float)):
            raise ValueError("Weight must be an integer or float.")
        # Check if payload is a valid type (json, dict)
        if payload is not None and not isinstance(payload, (dict, str)):
            raise ValueError("Payload must be a dictionary or a JSON string.")        
        node = self.root
        path = []
        for char in split_if_string(word):
            path.append(char)
            node = node.setdefault(ensure_valid_key(char), {})
        # Check if the word is already added to avoid duplicating IDs
        if _RESERVED_KEY in node:
            raise ValueError(f"Word '{word}' already exists in trie with the ID {node[_RESERVED_KEY]['id']}.")
        # Assign new ID, store weight, and payload in the node
        node_data = {'id': self.next_id, 'phrase': word, 'weight': weight, 'payload': payload}
        node[_RESERVED_KEY] = node_data
        self.id_to_node_path[self.next_id] = path
        self.next_id += 1  # Increment the ID for the next add
        
    def add_bulk(self, words_list, weight_list=None, payload_list=None):
        """Add multiple words or phrases to the trie, each with optional weights and payloads."""
        if (weight_list is isinstance(weight_list, list)) or len(words_list) != len(weight_list):
            if weight_list is isinstance(weight_list, list):
                raise ValueError(f"Weight list must be of instance list, not {type(weight_list)}.")
            if len(words_list) != len(weight_list):
                raise ValueError(f"Weight list must be a list of the same length as the words list (Length of words: {len(words_list)}, Length of weights: {len(weight_list)}).")
        if (payload_list is isinstance(payload_list, list)) or len(words_list) != len(payload_list):
            if payload_list is isinstance(payload_list, list):
                raise ValueError(f"Payload list must be of instance list, not {type(payload_list)}.")
            if len(words_list) != len(payload_list):
                raise ValueError(f"Payload list must be a list of the same length as the words list (Length of words: {len(words_list)}, Length of payloads: {len(payload_list)}).")
        for i, word in enumerate(words_list):
            self.add(word, weight_list[i] if weight_list else None, payload_list[i] if payload_list else None)
    
    # =====================================
    # Removing Words/Phrases Methods
    # ===================================== 

    def remove(self, items):
        """Remove phrases from the trie by a single string, a single ID, or lists of strings or IDs."""
        if isinstance(items, (str, int)):  # Single item (string or ID)
            self._remove_single(items)
        elif isinstance(items, list):  # List of strings or IDs
            for item in items:
                self._remove_single(item)
        else:
            raise ValueError("Unsupported input type. Must be string, int, or list of strings/ints.")

    def _remove_single(self, item):
        """Helper function to remove a single item, which could be a string or an ID."""
        if isinstance(item, int):  # Item is treated as an ID
            if item in self.id_to_node_path:
                path = self.id_to_node_path[item]
                phrase = ''.join(path)  # Convert path list to string if necessary
                self._remove_by_string(phrase)
                del self.id_to_node_path[item]  # Clean up the ID map after successful removal
            else:
                print(f"ID {item} not found in trie.")
        elif isinstance(item, str):  # Item is treated as a phrase
            self._remove_by_string(item)
        else:
            raise ValueError(f"Unsupported item type: {type(item)}. Must be int (ID) or str (phrase).")

    def _remove_by_string(self, phrase):
        """Remove a phrase from the trie by its string value, managing trie path cleanup."""
        try:
            node, parent, char = self.root, None, None
            path = split_if_string(phrase)
            last_char = path[-1]
            for char in path:
                parent, node = node, node.get(char)
                if node is None:
                    raise ValueError(f"Word '{phrase}' not found in trie.")
            if _RESERVED_KEY in node:
                del parent[last_char]  # Remove the node
                # Optionally clean up empty nodes recursively
                self._cleanup_empty_nodes(parent, last_char)
        except ValueError as e:
            print(e)

    def _cleanup_empty_nodes(self, node, char):
        """Recursively clean up empty nodes from the trie after deletion."""
        if node and not any(node.values()):  # Check if the node is empty
            del node[char]
                
    # =====================================
    # Phrase ID, Value, Weight Handling
    # =====================================
                
    def get_info(self, items, info_type='all'):
        """Fetch information from the trie by string, ID, or lists of strings/IDs.
        
        Args:
            items (int, str, list): Single ID, single string, or list of IDs/strings to fetch info for.
            info_type (str): Specifies the type of info to return ('all', 'id', 'word', 'payload').
        
        Returns:
            dict or list: The information requested as a single dictionary or a list of dictionaries.
        """
        def fetch_info(item):
            if isinstance(item, int):  # Assume it's an ID
                node_path = self.id_to_node_path.get(item)
                if not node_path:
                    return None
                node = self.root
                for char in node_path:
                    node = node[char]
                node_data = node.get(_RESERVED_KEY)
                if not node_data:
                    return None
                return format_info(node_data, info_type)
            elif isinstance(item, str):  # Assume it's a word
                node = self.root
                for char in split_if_string(item):
                    if char in node:
                        node = node[char]
                    else:
                        return None
                node_data = node.get(_RESERVED_KEY)
                if not node_data:
                    return None
                return format_info(node_data, info_type)
            else:
                raise ValueError("Items must be int (ID), str (word), or list of such.")

        def format_info(data, type):
            if type == 'all':
                return data
            elif type == 'id':
                return data.get('id')
            elif type == 'word':
                return data.get('phrase')  # Assuming you store the phrase in data
            elif type == 'payload':
                return data.get('payload')
            else:
                raise ValueError("Invalid info type specified.")

        if isinstance(items, list):
            return [fetch_info(item) for item in items if fetch_info(item) is not None]
        else:
            return fetch_info(items)
            
    # =====================================
    # Searching Words/Phrases Methods
    # =====================================

    def search(self, texts, return_type='all', return_meta=False):
        """Search for phrases or words in the trie and return found phrases, with detailed control over output.
        
        Args:
            texts (str or list): The text or a list of phrases to search within.
            return_type (str): Determines the type of information returned ('all', 'word', 'id', 'payload').
            return_meta (bool): If True, returns metadata about the search results separately.

        Returns:
            list: A list of matched phrases from the trie, and optionally a separate metadata dictionary.
        """
        def search_single(text):
            """Search for phrases within a single string and collect metadata."""
            if self.text_filter:
                text = filter_string(text)
            node = self.root
            current_phrase = []
            matches = []
            text_words = split_if_string(text)
            words_matched = 0

            for char in text_words:
                if char in node:
                    node = node[char]
                    current_phrase.append(char)
                else:
                    # Only add to matches if at a valid ending node before reset
                    if _RESERVED_KEY in node:
                        matches.append(node[_RESERVED_KEY])
                        words_matched += 1
                    node = self.root  # Reset to start search for the next possible phrase
                    current_phrase = []

                    # Reset current phrase and try to match current char in new context
                    if char in node:
                        node = node[char]
                        current_phrase = [char]  # Start new phrase with current character

            # Check at the end of the text to capture any ending phrases
            if _RESERVED_KEY in node:
                matches.append(node[_RESERVED_KEY])
                words_matched += 1

            return matches, words_matched
            
            # Final check to capture any phrase ending at the end of the text
            if _RESERVED_KEY in node:
                matches.append(node[_RESERVED_KEY])
                words_matched += 1

            return matches, words_matched

        def format_results(matches, words_matched):
            """Format the output based on the return_type and optionally add metadata."""
            result_list = []
            for match in matches:
                if return_type == 'all':
                    result_list.append(match)
                elif return_type == 'word':
                    result_list.append(match.get('phrase'))
                elif return_type == 'id':
                    result_list.append(match.get('id'))
                elif return_type == 'payload':
                    result_list.append(match.get('payload'))

            if return_meta:
                meta = {
                    'match_length': len(matches),
                    'match_ratio': words_matched / len(split_if_string(matches)) if matches else 0,
                    'mean_weight': sum(item.get('weight', 0) for item in matches) / len(matches) if matches else 0
                }
                return result_list, meta
            return result_list

        if isinstance(texts, str):
            matches, words_matched = search_single(texts)
            all_results, all_metadata = format_results(matches, words_matched) if return_meta else (format_results(matches, words_matched), None)
        elif isinstance(texts, list):
            all_results = []
            all_metadata = {}
            for text in texts:
                matches, words_matched = search_single(text)
                formatted_results, meta = format_results(matches, words_matched) if return_meta else (format_results(matches, words_matched), None)
                all_results.extend(formatted_results)
                if return_meta:
                    all_metadata.update({text: meta})
        else:
            raise ValueError("Input must be a string or a list of strings.")

        return all_results, all_metadata if return_meta else all_results

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
            doc_text = doc if not self.text_filter else filter_string(doc)
            for word in split_if_string(doc_text):
                # Ensure valid key and check if it exists in the trie
                word = ensure_valid_key(word)
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