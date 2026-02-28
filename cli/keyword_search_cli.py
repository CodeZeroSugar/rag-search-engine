#!/usr/bin/env python3

import argparse
import math

from indexing import InvertedIndex
from utils import tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="build inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get the term frequency for an ID and a term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to check frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency")
    idf_parser.add_argument(
        "term", type=str, help="Term to check inverse document frequency"
    )

    args = parser.parse_args()

    indexer = InvertedIndex()
    try:
        indexer.load()
    except FileNotFoundError:
        print("Index not found. Please build first.")
        return

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
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
        case "tf":
            try:
                freq = indexer.get_tf(args.doc_id, args.term)
            except Exception:
                print("0")
                return
            print(f"Term frequency for '{args.term}': {freq}")
        case "idf":
            idf_term = tokenize(args.term)[0]
            total_docs = len(indexer.docmap)
            total_match = len(indexer.get_documents(idf_term))
            idf = math.log((total_docs + 1) / (total_match + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
