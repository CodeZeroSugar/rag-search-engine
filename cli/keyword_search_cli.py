#!/usr/bin/env python3

import argparse

from indexing import InvertedIndex, bm25_idf_command, BM25_K1, bm25_tf_command, BM25_B
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

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculate combined frequency score"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to check frequency for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    indexer = InvertedIndex()

    match args.command:
        case "search":
            try:
                indexer.load()
            except FileNotFoundError:
                print("Index not found. Please build first.")
                return
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
                indexer.load()
            except FileNotFoundError:
                print("Index not found. Please build first.")
                return
            try:
                freq = indexer.get_tf(args.doc_id, args.term)
            except Exception:
                print("0")
                return
            print(f"Term frequency for '{args.term}': {freq}")
        case "idf":
            try:
                indexer.load()
            except FileNotFoundError:
                print("Index not found. Please build first.")
                return
            idf = indexer.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            try:
                indexer.load()
            except FileNotFoundError:
                print("Index not found. Please build first.")
                return
            tf_score = indexer.get_tf(args.doc_id, args.term)
            idf_score = indexer.get_idf(args.term)
            tfidf_score = tf_score * idf_score
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf_score:.2f}"
            )

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            try:
                indexer.load()
            except FileNotFoundError:
                print("Index not found. Please build first.")
                return
            bm25_results = indexer.bm25_search(args.query, 5)
            i = 1
            for result in bm25_results:
                print(
                    f"{i}. ({result[0]}) {indexer.docmap[result[0]]['title']} - Score: {result[1]:.2f}"
                )
                i += 1

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
