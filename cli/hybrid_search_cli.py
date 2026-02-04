import argparse
from lib.hybrid_search import normalize_command, weighted_search_command
from lib.search_utils import DEFAULT_ALPHA_HYBRID, DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores to 0-1")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="Scores to normalize")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Hybrid Search")
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=DEFAULT_ALPHA_HYBRID, help="Constant to control the weighting between scores")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_ALPHA_HYBRID, help="Limit search")
    
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()