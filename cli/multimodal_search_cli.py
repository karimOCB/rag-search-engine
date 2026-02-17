import argparse

from lib.multimodal_search import verify_image_embedding_command, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Generate and verify image embedding")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Required image path to embed")
    image_search_parser = subparsers.add_parser("image_search", help="Image to search for similar movies")
    image_search_parser.add_argument("image_path", type=str, help="Required image path to search")
    
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding_command(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result["title"]} (similarity: {result["similarity_score"]:.3f}) \n   {result["description"][:200]}...")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
