#!/usr/bin/env python3

import argparse
from lib.keyword_search import keyword_search, build_command, idf_command, bm25_idf_command
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
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID to calculate the TF-IDF")    
    tfidf_parser.add_argument("term", type=str, help="Term to calculate the TF-IDF")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")   

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
        case "tfidf":
            inv_idx = InvertedIndex()
            inv_idx.load()
            idf = idf_command(inv_idx, args.term)
            tf = inv_idx.get_tf(args.doc_id, args.term)
            tf_idf = tf * idf
            print(f"{tf_idf}")
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            inv_idx = InvertedIndex()
            BM25_IDF = bm25_idf_command(inv_idx, args.term)
            print(f"BM25 IDF score of '{args.term}': {BM25_IDF:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()