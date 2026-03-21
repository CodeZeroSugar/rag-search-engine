import argparse
from utils.utils import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="normalize scores"
    )

    weighted_search_parser = subparsers.add_parser("weighted-search")
    weighted_search_parser.add_argument("query", type=str, help="query to search for")
    weighted_search_parser.add_argument(
        "--aplha",
        type=float,
        default=0.5,
        help="dynamically control weight between two scores",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="limit the number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.scores)
        case "weighted-search":

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
