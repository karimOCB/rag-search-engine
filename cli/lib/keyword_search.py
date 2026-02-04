import string, math, pickle, os

from nltk.stem import PorterStemmer
from collections import Counter
from .search_utils import load_movies, load_stop_words, DEFAULT_SEARCH_LIMIT, CACHE_DIR, BM25_K1, BM25_B, single_token, SCORE_PRECISION

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
        with open(self.idx_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        
        for token in set(tokens):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(tokens) 
        
            
    def get_documents(self, term):
        if term not in self.index:
            return []

        doc_ids = sorted(self.index[term])
        return doc_ids
   
    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        single_token(tokens)
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term):
        tokens = tokenize_text(term)
        single_token(tokens)
        total_doc_count = len(self.docmap)
        term_doc_count = len(self.index[tokens[0]])
        term_idf = math.log((total_doc_count + 1) / (term_doc_count + 1))
        return term_idf

    def get_bm25_idf(self, term):
        tokens = tokenize_text(term)        
        single_token(tokens)
        total_docs = len(self.docmap)
        term_in_docs = len(self.index[tokens[0]])
        BM25_IDF = math.log((total_docs - term_in_docs + 0.5) / (term_in_docs + 0.5) + 1)
        return BM25_IDF
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        avg_doc_len = self.__get_avg_doc_length()
        if avg_doc_len > 0:
            len_norm = 1 - b + b * (self.doc_lengths.get(doc_id, 0) / avg_doc_len)
        else:
            len_norm = 1
        raw_tf = self.get_tf(doc_id, term)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * len_norm)

    def get_tf_idf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def __get_avg_doc_length(self):
        if not len(self.doc_lengths) or len(self.doc_lengths) == 0:
            return 0.0
        total = sum(self.doc_lengths.values())
        avg_doc_len = total / len(self.doc_lengths)
        return avg_doc_len

    def bm25(self, doc_id, term):
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_idf * bm25_tf

    def bm25_search(self, query, limit = DEFAULT_SEARCH_LIMIT):
        tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        results = []
        for doc_id, score in sorted_docs:
            doc = self.docmap[doc_id]
            formatted = {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"],
                "score": round(score, SCORE_PRECISION),
            }
            results.append(formatted)
        return results

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query, limit = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen_ids, result = set(), []

    for q_token in query_tokens:
        ids_set = idx.get_documents(q_token)
        for doc_id in ids_set:
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            movie = idx.docmap[doc_id]
            result.append((movie["id"], movie["title"]))
            if len(result) >= limit:
                return result
    return result

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_text(text):
    stemmer = PorterStemmer() 
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = [token for token in tokens if token != ""]
    stop_words = load_stop_words()
    filtered_tokens = [token for token in valid_tokens if token not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def bm25_tf_command(doc_id, term, k1 = BM25_K1, b = BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)   

def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)

def bm25search_command(query, limit = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)