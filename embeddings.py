"""Embeddings locales HuggingFace (sin API key).

get_embeddings() devuelve una instancia cacheada de HuggingFaceEmbeddings para
que indexar y consultar usen el mismo objeto. La carga del modelo es lazy.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Devuelve el cliente de embeddings cacheado (misma instancia indexar/consultar)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
