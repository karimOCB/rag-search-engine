import json, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(ROOT_DIR, "data", "movies.json")

DEFAULT_SEARCH_LIMIT = 5

def load_movies():
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data["movies"]