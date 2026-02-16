import os, json

from dotenv import load_dotenv
from google import genai
from .hybrid_search import HybridSearch
from .search_utils import (
    load_golden_data,
    load_movies,
)
from .semantic_search import SemanticSearch

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


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

def llm_evaluation(query, results):
    formatted_ranking = []
    for i, ranking in enumerate(results):
        formatted_ranking.append(f'{i}. Title: {ranking['document']['title']}  Description": {ranking['document']['description']}')
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

            Query: "{query}"

            Results:
            {chr(10).join(formatted_ranking)}
            Scale:
            - 3: Highly relevant
            - 2: Relevant
            - 1: Marginally relevant
            - 0: Not relevant

            Do NOT give any numbers out than 0, 1, 2, or 3.

            Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

            [2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    scores = json.loads(corrected)

    llm_valuation = []
    for i, result in enumerate(results, start=1):
        llm_valuation.append(f"{i}. {result["document"]["title"]}: {scores[i-1]}/3")
    if len(scores) == len(results):
        return llm_valuation

    raise ValueError(
        f"LLM response parsing error. Expected {len(results)} scores, got {len(scores)}. Response: {scores}"
    )