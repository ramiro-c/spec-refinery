"""Hybrid retriever: BM25 + vectors with RRF fusion (pattern P4).

``retrieve()`` accepts lexical/semantic stubs in tests. In production it uses
EnsembleRetriever (c=60, weights 0.5/0.5) built lazily over Chroma + BM25.
``aretrieve()`` is the async path used by graph nodes: retrievers are awaited
natively and the only blocking step (lazy first build) runs in a worker
thread, never bare inside an async function.
"""

from __future__ import annotations

import asyncio

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import CHROMA_DIR, CORPUS_DIR, TOP_K, VECTOR_BACKEND
from schemas import Citation

# RRF fusion (D8): same parameters as pre-entrega-4.
RRF_C = 60
RRF_WEIGHTS = [0.5, 0.5]
COLLECTION_NAME = "spec-refinery"
# Documents are ingested whole (no splitter), and the largest rule in the
# corpus is ~1.1 KB, so citations carry the full rule: the panel exists to be
# read. This is only a safety valve against a pathological document blowing up
# the interrogator prompt — nothing in the corpus comes close to it.
MAX_EXCERPT_CHARS = 2000

# Lazy production ensemble (built on first retrieval, not at import).
_ensemble: EnsembleRetriever | None = None


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Extract simple front-matter YAML (document_id, title) from markdown."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text[end + 5 :]


def load_corpus_documents() -> list[Document]:
    """Load the markdown corpus with document_id and title from front-matter."""
    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(text)
        document_id = meta.get("document_id") or path.name
        documents.append(
            Document(
                page_content=body.strip(),
                metadata={
                    "document_id": document_id,
                    "title": meta.get("title", ""),
                },
            )
        )
    return documents


def build_hybrid(lexical, semantic) -> EnsembleRetriever:
    """Assemble the RRF ensemble between lexical and semantic retrievers."""
    return EnsembleRetriever(
        retrievers=[lexical, semantic],
        weights=list(RRF_WEIGHTS),
        c=RRF_C,
    )


def _build_production_retrievers():
    """Rebuild retrievers from disk (Chroma) and corpus (BM25)."""
    if VECTOR_BACKEND != "chroma":
        raise RuntimeError(
            f"Unsupported VECTOR_BACKEND: {VECTOR_BACKEND} (only chroma is supported)"
        )
    from embeddings import get_embeddings

    documents = load_corpus_documents()
    lexical = BM25Retriever.from_documents(documents, k=TOP_K)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    semantic = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return lexical, semantic


def _production_retriever() -> EnsembleRetriever:
    """Lazy: BM25 + vector ensemble according to the configured backend."""
    global _ensemble
    if _ensemble is None:
        lexical, semantic = _build_production_retrievers()
        _ensemble = build_hybrid(lexical, semantic)
    return _ensemble


def _excerpt(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    """The rule as retrieved, whole. Cut only if it is absurdly long.

    Line breaks are kept: these are full documents, and their paragraphs are
    part of how the rule reads. If a cut is ever needed it lands on a
    boundary, never mid-word.
    """
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    window = clean[:limit]
    # Only a full stop closes a thought. A semicolon does not, so cutting
    # there and presenting it as complete would misread as a broken excerpt.
    sentence_end = window.rfind(". ")
    if sentence_end >= limit // 2:
        return window[: sentence_end + 1]
    word_end = window.rfind(" ")
    trimmed = window[:word_end] if word_end > 0 else window
    return trimmed.rstrip(" ,;:.—-") + "…"


def _as_citation(doc: Document) -> Citation:
    meta = doc.metadata or {}
    return Citation(
        document_id=str(meta.get("document_id") or ""),
        title=str(meta.get("title") or ""),
        excerpt=_excerpt(doc.page_content),
    )


def _docs_to_citations(docs: list[Document], top_k: int) -> list[Citation]:
    """Convert Documents to Citations, deduplicating by document_id."""
    seen: set[str] = set()
    out: list[Citation] = []
    for doc in docs:
        cit = _as_citation(doc)
        if not cit.document_id or cit.document_id in seen:
            continue
        seen.add(cit.document_id)
        out.append(cit)
        if len(out) >= top_k:
            break
    return out


async def _ainvoke_docs(retriever, query: str) -> list[Document]:
    """Await a retriever natively; fall back to a worker thread for sync stubs."""
    ainvoke = getattr(retriever, "ainvoke", None)
    if ainvoke is not None:
        return list(await ainvoke(query))
    return list(await asyncio.to_thread(retriever.invoke, query))


async def aretrieve(
    query: str,
    *,
    lexical=None,
    semantic=None,
    top_k: int = TOP_K,
) -> list[Citation]:
    """Async top-k retrieval. Tests inject stubs; production uses the ensemble."""
    if lexical is not None and semantic is not None:
        # Test stubs: concatenate lexical + semantic and reuse citation dedupe.
        docs = await _ainvoke_docs(lexical, query) + await _ainvoke_docs(semantic, query)
        return _docs_to_citations(docs, top_k)
    # First ensemble build loads Chroma + BM25 (blocking) — run it off-loop.
    retriever = await asyncio.to_thread(_production_retriever)
    docs = await _ainvoke_docs(retriever, query)
    return _docs_to_citations(docs, top_k)


def retrieve(
    query: str,
    *,
    lexical=None,
    semantic=None,
    top_k: int = TOP_K,
) -> list[Citation]:
    """Sync retrieval for non-async callers (evaluate.py, tests with stubs)."""
    if lexical is not None and semantic is not None:
        # Test stubs: concatenate lexical + semantic and reuse citation dedupe.
        docs = list(lexical.invoke(query)) + list(semantic.invoke(query))
        return _docs_to_citations(docs, top_k)
    docs = _production_retriever().invoke(query)
    return _docs_to_citations(docs, top_k)
