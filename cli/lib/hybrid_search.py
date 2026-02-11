import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import DEFAULT_ALPHA_HYBRID, DEFAULT_SEARCH_LIMIT, RRF_K1, load_movies
from lib.query_enhancement import enhance_query
from lib.rerank import rerank_result

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.idx_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit = DEFAULT_SEARCH_LIMIT):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        bm25_scores = [result["score"] for result in bm25_results]
        semantic_scores = [result["score"] for result in semantic_results]
        bm_norm_scores = normalize_command(bm25_scores)
        sem_norm_scores = normalize_command(semantic_scores)
        ids_to_k_scores = {}
        ids_to_sem_scores = {}
        for i, result in enumerate(bm25_results):
            ids_to_k_scores[result["id"]] = bm_norm_scores[i]
        for i, result in enumerate(semantic_results):
            ids_to_sem_scores[result["id"]] = sem_norm_scores[i]
        all_ids = set(ids_to_k_scores.keys()) | set(ids_to_sem_scores.keys())
        doc_scores = {}
        for id in all_ids:
            doc_scores[id] = {
                "document": self.semantic_search.document_map[id],
                "bm25_score": ids_to_k_scores.get(id, 0.0),
                "semantic_score": ids_to_sem_scores.get(id, 0.0),
                "hybrid_score": hybrid_score(ids_to_k_scores.get(id, 0.0), ids_to_sem_scores.get(id, 0.0), alpha)
            }
        sorted_scores = sorted(doc_scores.items(), key=lambda item: item[1]["hybrid_score"], reverse=True)
        sorted_scores = [score[1] for score in sorted_scores] 
        return sorted_scores

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        ids_to_k_ranks = {}
        ids_to_sem_ranks = {}
        for i, result in enumerate(bm25_results, start=1):
            ids_to_k_ranks[result["id"]] = i
        for i, result in enumerate(semantic_results, start=1):
            ids_to_sem_ranks[result["id"]] = i
        all_ids = set(ids_to_k_ranks.keys()) | set(ids_to_sem_ranks.keys())
        doc_rankings = {}        
        for i, id in enumerate(all_ids):
            doc_rankings[id] = {
                "document": self.semantic_search.document_map[id],
                "bm25_rank": ids_to_k_ranks.get(id),
                "semantic_rank": ids_to_sem_ranks.get(id),
                "rrf_score": 0.0
            }
            if doc_rankings[id]["bm25_rank"] is not None:
                doc_rankings[id]["rrf_score"] += rrf_score(doc_rankings[id]["bm25_rank"], k)
            if doc_rankings[id]["semantic_rank"] is not None:
                doc_rankings[id]["rrf_score"] += rrf_score(doc_rankings[id]["semantic_rank"], k)
        sorted_rankings = sorted(doc_rankings.items(), key=lambda item: item[1]["rrf_score"], reverse=True)
        sorted_rankings = [ranking[1] for ranking in sorted_rankings]
        return sorted_rankings
    

def normalize_command(scores):
    if not scores:
        return
    
    minimum = min(scores)
    maximum = max(scores)

    if minimum == maximum:
        print(f"{[1.0] * len(scores)}")
        return [1.0] * len(scores)
    
    else:
        normalized_scores = []
        for score in scores:
            normalized_s = (score - minimum) / (maximum - minimum)
            normalized_scores.append(round(normalized_s, 4))
        return normalized_scores


def weighted_search_command(query, alpha = DEFAULT_ALPHA_HYBRID, limit = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    scores = hybrid_search.weighted_search(query, alpha, limit)
    return scores[:limit]

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_search_command(query, k = RRF_K1, enhance = None, rerank_method = None, limit = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    
    original_query = query 
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query
    
    new_limit = limit if rerank_method else limit

    results = hybrid_search.rrf_search(query, k, new_limit)

    if rerank_method:
        results = rerank_result(results[:limit], query, rerank_method)
    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "rerank_method": rerank_method,
        "query": query,
        "k": k,
        "results": results,
    }

def rrf_score(rank, k=60):
    return 1 / (k + rank)