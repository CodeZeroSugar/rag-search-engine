#!/usr/bin/env python3

import argparse

from utils.utils import load_movies
from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_query_text,
    embed_text,
    overlap_chunk,
    semantic_chunk,
    verify_embeddings,
    verify_model,
)

import transformers

transformers.logging.set_verbosity_error()


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify model")

    embed_text_parser = subparsers.add_parser("embed_text", help="embed text")
    embed_text_parser.add_argument("text", type=str, help="text to embed")

    subparsers.add_parser("verify_embeddings", help="verify embeddings")

    embedquery_parser = subparsers.add_parser("embedquery", help="embed query")
    embedquery_parser.add_argument("query", type=str, help="query to embed")

    search_parser = subparsers.add_parser("search", help="search for a query")
    search_parser.add_argument("query", type=str, help="query to search for")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="number of results to return"
    )

    chunk_parser = subparsers.add_parser("chunk", help="chunk input text")
    chunk_parser.add_argument("text", type=str, help="text to be chunked")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="amount of text per chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="specify overlap for chunking"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="chunk semantically"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="text to be chunked")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="set max chunk size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="specify overlap for chunking"
    )

    subparsers.add_parser("embed_chunks", help="embed chunks from a provided text")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="semantically search with chunking"
    )
    search_chunked_parser.add_argument("query", type=str, help="query text")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="number of results to return"
    )

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
            model = SemanticSearch()
            movies = load_movies()
            model.load_or_create_embeddings(movies)
            results = model.search(args.query, args.limit)
            i = 1
            for result in results:
                print(
                    f"{i}. {result['title']} (score: {result['score']})\n   {result['description']}"
                )
                i += 1

        case "chunk":
            split_text = args.text.split()
            chunks = overlap_chunk(split_text, args.overlap, args.chunk_size)

            print(f"Chunking {len(args.text)} characters")
            i = 1
            for c in chunks:
                print(f"{i}. {c}")
                i += 1

        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            if len(chunks) > 0:
                print(f"Semantically chunking {len(args.text)} characters")
                i = 1
                for c in chunks:
                    print(f"{i}. {c}")
                    i += 1

        case "embed_chunks":
            movies = load_movies()
            chunker = ChunkedSemanticSearch()
            embeddings = chunker.load_or_create_chunk_embeddings(movies)

            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            movies = load_movies()
            chunker = ChunkedSemanticSearch()
            embeddings = chunker.load_or_create_chunk_embeddings(movies)
            results = chunker.search_chunks(args.query, args.limit)

            i = 1
            for result in results:
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"    {result['document']}...")
                i += 1

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
