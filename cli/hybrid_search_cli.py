import argparse
from utils.utils import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="normalize scores"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.scores)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
