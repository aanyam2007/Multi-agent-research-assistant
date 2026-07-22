"""Build the local RAG index from documents under ./data.

Usage:
    uv run python ingest.py
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

from tools.rag import build_index, DATA_DIR


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    count = build_index()
    if count == 0:
        print(f"\nNo documents indexed. Drop .txt/.md/.pdf files into {DATA_DIR} and re-run.")
    else:
        print(f"\nIndexed {count} chunk(s) from {DATA_DIR}.")


if __name__ == "__main__":
    main()
