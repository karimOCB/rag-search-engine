import argparse

from lib.search_utils import load_golden_data, RRF_K1
from lib.hybrid_search import rrf_search_command

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to evaluate (k for precision@k, recall@k)",)

    args = parser.parse_args()
    limit = args.limit

    golden_data = load_golden_data()
    
    results = []
    for i, case in enumerate(golden_data):
        query = case["query"]
        result = rrf_search_command(query, RRF_K1, enhance = None, rerank_method = None, limit = limit)
        relevant_set = set(case["relevant_docs"])
        retrieved_titles = [r["document"]["title"] for r in result["results"]]
        relevant_retrieved = sum(1 for title in retrieved_titles if title in relevant_set)
        precision = relevant_retrieved / len(result["results"]) 
        result["precision"] = precision
        result["relevant_titles"] = relevant_set
        result["retrieved_titles"] = retrieved_titles
        results.append(result)
    
    print(f"k={limit}\n")
    for r in results:
        print(f"- Query: {r['original_query']}\n - Precision@{limit}: {r['precision']:.4f}\n - Retrieved: {(", ").join(r['retrieved_titles'])}\n - Relevant: {(", ").join(r['relevant_titles'])}")
    

if __name__ == "__main__":
    main()