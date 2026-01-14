#!/usr/bin/env python3

import argparse, json, os



def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = keyword_search(args.query)
            for i in range(len(result)):
                print(f"{i+1}. {result[i]["title"]} {i+1}")
            pass
        case _:
            parser.print_help()


def keyword_search(query):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "..", "data", "movies.json")
    result = []

    with open(json_path, 'r', encoding='utf-8') as f:
        f_movies = json.load(f)
        movies = f_movies["movies"]
    for movie in movies:
        title = movie["title"]
        if query in title:
            result.append(movie)
    
    return result


if __name__ == "__main__":
    main()