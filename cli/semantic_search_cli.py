#!/usr/bin/env python3
import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk, semantic_chunk, embed_chunks
from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verify model")
    embed_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    verify_embedding_parser = subparsers.add_parser("verify_embeddings", help="Embed text")
    embedquery_parser = subparsers.add_parser("embedquery", help="Embed text")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")
    search_parser = subparsers.add_parser("search", help="Search similar movies")
    search_parser.add_argument("query", type=str, help="Query to search similar movies.")
    search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Tunable search limit")
    chunk_parser = subparsers.add_parser("chunk", help="Chunk documents")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=DEFAULT_CHUNK_LIMIT, help="Tunable search limit")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=DEFAULT_CHUNK_LIMIT, help="Tunable search overlap")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk documents semantically")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Tunable search limit")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Tunable search overlap")
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Chunk documents semantically")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
                    
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            chunked_embbedings = embed_chunks()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()