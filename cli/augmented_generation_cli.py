import argparse

from lib.augmented_generation import rag_command, summarize_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser("summarize", help="Synthetize multiple search result")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit search")
    
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            titles, generated_answer = rag_command(query)
            print("Search Results:")
            for title in titles:
                print(title)
            print("\nRAG Response:")
            print(generated_answer)
        case "summarize":
            query = args.query
            titles, generated_summary = summarize_command(query)
            print("Search Results:")
            for title in titles: 
                print(title)
            print("\nLLM Summary:")
            print(generated_summary)    
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()