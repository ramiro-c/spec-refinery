"""Corpus ingestion into local Chroma.

Reads markdown with front-matter, persists vectors and builds BM25 over the
same Documents. The CLI `python ingest.py` runs the full ingestion.
"""

from __future__ import annotations

import sys

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from config import CHROMA_DIR, TOP_K
from embeddings import get_embeddings
from retriever import COLLECTION_NAME, load_corpus_documents


def ingest_chroma() -> None:
    """Index the corpus into local Chroma."""
    documents = load_corpus_documents()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"[ingest] Chroma: {len(documents)} documents in {CHROMA_DIR}")


def main() -> None:
    """Ingestion CLI: python ingest.py."""
    try:
        ingest_chroma()
    except Exception as error:  # noqa: BLE001 — CLI with a clear message
        print(f"[ingest] ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
