import pickle, os, math
from .keyword_search import tokenize_text
from .search_utils import load_movies, CACHE_DIR
from collections import Counter

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.idx_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1            
            
    def get_documents(self, term):
        term = tokenize_text(term)
        if term[0] not in self.index:
            return []

        doc_ids = sorted(self.index[term[0]])
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

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"Error: '{self.idx_path}' needed to continue.")

        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(f"Error: '{self.docmap_path}' needed to continue.")

        if not os.path.exists(self.term_frequencies_path):
            raise FileNotFoundError(f"Error: '{self.term_frequencies_path}' needed to continue.")

        if not os.path.exists(self.doc_lengths_path):
            raise FileNotFoundError(f"Error: '{self.doc_lengths_path}' needed to continue.")

        with open(self.idx_path, "rb") as f:
            self.index = pickle.load(f)
          
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
            
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id, term):
        term = tokenize_text(term)
        self.single_token(term)
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][term[0]]

    def get_bm25_idf(self, term):
        term = tokenize_text(term)
        self.single_token(term)
        movies = load_movies()
        total_docs = len(movies)
        term_in_docs = len(self.get_documents(term))
        BM25_IDF = math.log((total_docs - term_in_docs + 0.5) / (term_in_docs + 0.5) + 1)
        return BM25_IDF

    def single_token(self, term):
        if isinstance(term, list):
            if len(term) > 1:
                raise Exception(f"Given term: {term} is more than one token!")
            if len(term) == 0:
                raise Exception("Given term is empty!")
    
    def get_bm25_tf(self, doc_id, term, k1, b):
        avg_doc_len = self.__get_avg_doc_length()
        len_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_len)
        raw_tf = self.get_tf(doc_id, term)
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * len_norm)
        return bm25_tf

    def __get_avg_doc_length(self):
        if not len(self.doc_lengths):
            return 0.0
        total = sum(self.doc_lengths.values())
        avg_doc_len = total / len(self.doc_lengths)
        return avg_doc_len

