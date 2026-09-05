"""Ingesta del corpus: Chroma local (default) o Pinecone (VECTOR_BACKEND=pinecone).

Lee markdown con front-matter, persiste vectores y arma BM25 sobre los mismos
Document. El CLI `python ingest.py` ejecuta la ingesta completa.
"""

from __future__ import annotations

import sys

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from config import CHROMA_DIR, INDEX_NAME, PINECONE_API_KEY, TOP_K, VECTOR_BACKEND
from embeddings import get_embeddings
from retriever import (
    COLLECTION_NAME,
    load_corpus_documents,
    set_production_retrievers,
)


def ingest_chroma() -> None:
    """Indexa el corpus en Chroma local y registra BM25 para retrieve()."""
    documentos = load_corpus_documents()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=documentos,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    lexical = BM25Retriever.from_documents(documentos, k=TOP_K)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    semantic = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    set_production_retrievers(lexical, semantic)
    print(f"[ingest] Chroma: {len(documentos)} documentos en {CHROMA_DIR}")


def ingest_pinecone() -> None:
    """Rama Pinecone copiada de P4 (solo si VECTOR_BACKEND=pinecone)."""
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY requerida para VECTOR_BACKEND=pinecone")

    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    documentos = load_corpus_documents()
    cliente = Pinecone(api_key=PINECONE_API_KEY)
    vectorstore = PineconeVectorStore(
        index=cliente.Index(INDEX_NAME),
        embedding=get_embeddings(),
        text_key="texto",
        namespace="docs",
    )
    vectorstore.add_documents(documentos)
    lexical = BM25Retriever.from_documents(documentos, k=TOP_K)
    semantic = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    set_production_retrievers(lexical, semantic)
    print(f"[ingest] Pinecone: {len(documentos)} documentos en índice '{INDEX_NAME}'")


def main() -> None:
    """CLI de ingesta: python ingest.py."""
    try:
        if VECTOR_BACKEND == "pinecone":
            ingest_pinecone()
        else:
            ingest_chroma()
    except Exception as error:  # noqa: BLE001 — CLI con mensaje claro
        print(f"[ingest] ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
