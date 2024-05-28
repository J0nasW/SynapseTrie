import re
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()

def filter_string(text):
    """Process and clean text for insertion into the trie."""
    text = ' '.join([_lemmatizer.lemmatize(word).lower() for word in text.split()])
    pattern = r"[^a-zA-Z0-9 ]|[\u0400-\u04FF]+|[\u4E00-\u9FFF]+|\d+|-"
    text = re.sub(pattern, " ", text).strip()
    return re.sub(r"\s+", " ", text)

def ensure_valid_key(word, reserved_key='#'):
    """Ensure the trie keys do not start with the reserved character."""
    return f"{reserved_key}{word}" if word.startswith(reserved_key) else word

def split_if_string(words):
    """Split words if it's a string, else treat as list."""
    return words.split() if isinstance(words, str) else words