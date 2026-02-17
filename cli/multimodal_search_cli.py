import argparse

from lib.multimodal_search import verify_image_embedding_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Generate and verify image embedding")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Required image path to embed")
    
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding_command(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
