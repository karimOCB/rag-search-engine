from .hybrid_search import HybridSearch
from .search_utils import (
    load_golden_data,
    load_movies,
)
from .semantic_search import SemanticSearch


def precision_at_k(retrieved_docs, relevant_docs, k = 5):
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k

def recall_at_k(retrieved_docs, relevant_docs, k = 5):
    if not relevant_docs:
        return 0.0
    relevant_count = 0
    for doc in retrieved_docs[:k]:
        title = doc["document"]["title"]
        if title in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs)

def evaluate_command(limit = 5):
    movies = load_movies()
    golden_data_tests = load_golden_data()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    total_precision = 0
    results_by_query = {}
    for test_case in golden_data_tests:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])
        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_docs = []
        for result in search_results:
            title = result["document"].get("title", "")
            if title:
                retrieved_docs.append(title)
        
        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(search_results, relevant_docs, limit)
        f1 = 2 * (precision * recall) / (precision + recall)

        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs),
        }

        total_precision += precision
        
    return {
        "test_cases_count": len(golden_data_tests),
        "limit": limit,
        "results": results_by_query,
    }
