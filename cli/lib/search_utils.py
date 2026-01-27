import json, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(ROOT_DIR, "data", "movies.json")
stop_words_path = os.path.join(ROOT_DIR, "data", "stopwords.txt")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 3
DEFAULT_CHUNK_LIMIT = 200

def load_movies():
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words():
    with open(stop_words_path, 'r') as f:
        stop_words = f.read()
        stop_words = stop_words.splitlines()
    return stop_words

def single_token(term):
    if isinstance(term, list):
        if len(term) != 1:
            raise ValueError("Term must be a single token")