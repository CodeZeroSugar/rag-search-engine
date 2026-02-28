#!/usr/bin/env python3

import argparse

from indexing import InvertedIndex
from utils import tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="build inverted index")
    args = parser.parse_args()

    indexer = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                indexer.index, indexer.docmap = indexer.load()
            except FileNotFoundError:
                print("Index not found. Please build first.")
                return
            results = []
            query_tokens = tokenize(args.query)
            maxed = False
            for token in query_tokens:
                doc_ids = indexer.get_documents(token)
                for id in doc_ids:
                    results.append(indexer.docmap[id])
                    if len(results) == 5:
                        maxed = True
                        break
                if maxed:
                    break

            for i in range(len(results)):
                print(f"{i + 1}. {results[i]['title']} : {results[i]['id']}")
        case "build":
            indexer.build()
            indexer.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
