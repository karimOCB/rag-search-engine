#!/usr/bin/env python3

import argparse
from lib.keyword_search import keyword_search, build_command, idf_command
from lib.inverted_index import InvertedIndex

def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Building inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequencies")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency")
    idf_parser = subparsers.add_parser("idf", help="Calculate IDF")
    idf_parser.add_argument("term", type=str, help="Term to get the IDF")

    args = parser.parse_args()

    match args.command:
        case "build":
            print(f"Building inverted index...")
            inv_idx = InvertedIndex()
            build_command(inv_idx)
            print("Inverted index built successfully.")
        case "search":
            print(f"Searching for: {args.query}")
            inv_idx = InvertedIndex()

            result = keyword_search(args.query, inv_idx)
            for i in range(len(result)):
                print(f"{result[i][1]} {result[i][0]}")
        case "tf":
            inv_idx = InvertedIndex()
            inv_idx.load()
            frequency = inv_idx.get_tf(args.doc_id, args.term)
            print(f"{frequency}")
        case "idf":
            inv_idx = InvertedIndex()
            term_idf = idf_command(inv_idx, args.term)
            print(f"Inverse document frequency of '{args.term}': {term_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()