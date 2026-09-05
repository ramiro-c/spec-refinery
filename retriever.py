"""Recuperador híbrido: BM25 + vectores con fusión RRF (patrón P4).

retrieve() acepta stubs lexical/semantic en tests. En producción usa
EnsembleRetriever (c=60, pesos 0.5/0.5) y devuelve citas con document_id.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import CHROMA_DIR, CORPUS_DIR, INDEX_NAME, PINECONE_API_KEY, TOP_K, VECTOR_BACKEND
from schemas import Citation

# Fusión RRF (D8): mismos parámetros que pre-entrega-4.
RRF_C = 60
RRF_WEIGHTS = [0.5, 0.5]
COLLECTION_NAME = "spec-refinery"

# Cache de producción (se arma en ingest).
_bm25_retriever: BM25Retriever | None = None
_vector_retriever = None


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Extrae front-matter YAML simple (document_id, title) del markdown."""
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
    """Carga el corpus markdown con document_id y title del front-matter."""
    documentos: list[Document] = []
    for ruta in sorted(CORPUS_DIR.rglob("*.md")):
        texto = ruta.read_text(encoding="utf-8")
        meta, cuerpo = _parse_front_matter(texto)
        document_id = meta.get("document_id") or ruta.name
        documentos.append(
            Document(
                page_content=cuerpo.strip(),
                metadata={
                    "document_id": document_id,
                    "title": meta.get("title", ""),
                },
            )
        )
    return documentos


def build_hybrid(lexical, semantic) -> EnsembleRetriever:
    """Arma el ensemble RRF entre retriever léxico y semántico (producción)."""
    return EnsembleRetriever(
        retrievers=[lexical, semantic],
        weights=list(RRF_WEIGHTS),
        c=RRF_C,
    )


def _as_citation(doc: Document) -> Citation:
    meta = doc.metadata or {}
    return Citation(
        document_id=str(meta.get("document_id") or ""),
        title=str(meta.get("title") or ""),
        excerpt=doc.page_content[:240],
    )


def _docs_to_citations(docs: list[Document], top_k: int) -> list[Citation]:
    """Convierte Documentos a Citation deduplicando por document_id."""
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


def set_production_retrievers(lexical, semantic) -> None:
    """Registra retrievers de producción tras la ingesta (usado por ingest.py)."""
    global _bm25_retriever, _vector_retriever
    _bm25_retriever = lexical
    _vector_retriever = semantic


def build_production_retrievers():
    """Reconstruye retrievers desde disco (Chroma/Pinecone) y corpus (BM25)."""
    from embeddings import get_embeddings

    documentos = load_corpus_documents()
    lexical = BM25Retriever.from_documents(documentos, k=TOP_K)
    if VECTOR_BACKEND == "pinecone":
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY requerida para VECTOR_BACKEND=pinecone")
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore

        cliente = Pinecone(api_key=PINECONE_API_KEY)
        vectorstore = PineconeVectorStore(
            index=cliente.Index(INDEX_NAME),
            embedding=get_embeddings(),
            text_key="texto",
            namespace="docs",
        )
    else:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
    semantic = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return lexical, semantic


def _production_retriever():
    """Lazy: ensemble BM25 + vector según el backend configurado."""
    global _bm25_retriever, _vector_retriever
    if _bm25_retriever is None or _vector_retriever is None:
        lexical, semantic = build_production_retrievers()
        set_production_retrievers(lexical, semantic)
    return build_hybrid(_bm25_retriever, _vector_retriever)


def retrieve(
    query: str,
    *,
    lexical=None,
    semantic=None,
    top_k: int = TOP_K,
) -> list[Citation]:
    """Recupera top-k citas. Tests inyectan stubs; producción usa EnsembleRetriever."""
    if lexical is not None and semantic is not None:
        seen: set[str] = set()
        out: list[Citation] = []
        for block in (lexical.invoke(query), semantic.invoke(query)):
            for doc in block:
                cit = _as_citation(doc)
                if not cit.document_id or cit.document_id in seen:
                    continue
                seen.add(cit.document_id)
                out.append(cit)
                if len(out) >= top_k:
                    return out
        return out

    docs = _production_retriever().invoke(query)
    return _docs_to_citations(docs, top_k)
