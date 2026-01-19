import pickle, os
from .keyword_search import tokenize_text
from .search_utils import load_movies, CACHE_DIR

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.idx_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        term = term.lower()
        if term not in self.index:
            return []

        doc_ids = sorted(self.index[term])
        return doc_ids

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie  
            self.__add_document(movie["id"], f'{movie["title"]} {movie["description"]}')          
        

    def save(self):
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)
        
        with open(self.idx_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        if not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"Error: '{self.idx_path}' needed to continue.")

        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(f"Error: '{self.docmap_path}' needed to continue.")

        with open(self.idx_path, "rb") as f:
            self.index = pickle.load(f)
          
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)