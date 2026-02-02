import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import DEFAULT_ALPHA_HYBRID, DEFAULT_SEARCH_LIMIT, load_movies

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.idx_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_result = self._bm25_search(query, limit * 500)
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)
        return (bm25_result, semantic_result)

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_command(scores):
    if not scores:
        return
    minimum = min(scores)
    maximum = max(scores)
    if minimum == maximum:
        print(f"{[1.0] * len(scores)}")
    else:
        for score in scores:
            normalized_s = (score - minimum) / (maximum - minimum)
            print(f"* {normalized_s:.4f}")

def weighted_search_command(query, alpha = DEFAULT_ALPHA_HYBRID, limit = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    tuple_result = hybrid_search.weighted_search(query, alpha, limit)
    print(f"{tuple_result[0]}, {tuple_result[1]}")