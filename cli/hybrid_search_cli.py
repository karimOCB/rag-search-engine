import argparse
from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command
from lib.search_utils import DEFAULT_ALPHA_HYBRID, DEFAULT_SEARCH_LIMIT, RRF_K1

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores to 0-1")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="Scores to normalize")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Hybrid Search")
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=DEFAULT_ALPHA_HYBRID, help="Constant to control the weighting between scores")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit search")
    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF Search")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument("-k", type=float, nargs='?', default=RRF_K1, help="Constant to control the weighting of higher vs lower ranked results")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit search")
    rrf_search_parser.add_argument("--enhance", type=str, nargs='?', choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, nargs='?', choices=["individual"], help="Limit search")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_command(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            scores = weighted_search_command(args.query, args.alpha, args.limit)
            for i, score in enumerate(scores):
                print(f"\n{i}. {score['document']['title']}\nHybrid Score: {score['hybrid_score']:.4f})\nBM25: {score['bm25_score']:.4f}, Semantic: {score['semantic_score']:.4f}\n{score['document']['description'][:123]}...")
        case "rrf-search":
            result = rrf_search_command(args.query, args.k, args.enhance, args.limit)
            
            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )

            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )


            for i, ranking in enumerate(result["results"]):
                print(f"\n{i}. {ranking['document']['title']}\nRRF Score: {ranking['rrf_score']:.4f}\nBM25 Rank: {ranking['bm25_rank']:.4f}, Semantic Rank: {ranking['semantic_rank']:.4f}\n{ranking['document']['description'][:123]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

    