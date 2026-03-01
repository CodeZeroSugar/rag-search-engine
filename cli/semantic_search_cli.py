#!/usr/bin/env python3

import argparse

from utils.utils import load_movies
from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
