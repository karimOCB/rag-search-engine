#!/usr/bin/env python3

import argparse
from lib.keyword_search import keyword_search, build_command
from lib.inverted_index import InvertedIndex

def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Building inverted index")

    args = parser.parse_args()

    match args.command:
        case "build":
            print(f"Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print(f"Searching for: {args.query}")
            inv_idx = InvertedIndex()
            result = keyword_search(args.query, inv_idx)
            for i in range(len(result)):
                print(f"{result[i][1]} {result[i][0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()