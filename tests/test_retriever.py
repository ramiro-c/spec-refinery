import pathlib

from langchain_core.documents import Document
from retriever import MAX_EXCERPT_CHARS, LocalBM25Retriever, _excerpt, retrieve


class _Stub:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query: str):
        return self._docs


def test_retrieve_dedupes_by_document_id_and_keeps_cart_rule():
    cart = Document(
        page_content="El precio se cierra en el carrito. No se saltea.",
        metadata={"document_id": "adr-cart-price.md", "title": "Precio en carrito"},
    )
    other = Document(
        page_content="Flags de Cyber Monday.",
        metadata={"document_id": "adr-cyber-flags.md", "title": "Flags"},
    )
    citations = retrieve(
        "¿Se puede saltear el carrito?",
        lexical=_Stub([cart, other]),
        semantic=_Stub([cart]),
    )
    ids = [c.document_id for c in citations]
    assert "adr-cart-price.md" in ids
    assert ids.count("adr-cart-price.md") == 1


def test_excerpt_keeps_short_content_intact():
    assert _excerpt("El precio se cierra en el carrito.") == (
        "El precio se cierra en el carrito."
    )


def test_excerpt_keeps_a_whole_rule_untouched():
    """The panel exists to read the rule: a rule-sized document is not cut."""
    rule = pathlib.Path("corpus/rules/adr-cart-price.md").read_text(encoding="utf-8")
    assert len(rule) > 900
    assert _excerpt(rule) == rule.strip()


def test_excerpt_preserves_the_paragraphs_of_the_rule():
    text = "Buyer — compra.\n\nCarrito — contiene ítems."
    assert _excerpt(text) == text


# The cuts below only happen at the safety valve, so they pass an explicit
# small limit; real rules are never trimmed.
def test_excerpt_never_cuts_a_word_in_half():
    """The bug: a hard slice produced tails like "alineado c"."""
    text = "palabra " * 200
    out = _excerpt(text, limit=280)
    assert out.endswith("…")
    assert not out.rstrip("…").endswith("palabr")
    assert all(word == "palabra" for word in out.rstrip("…").split())


def test_excerpt_prefers_a_sentence_boundary_over_an_ellipsis():
    """A sentence that closes late in the window wins: it reads complete."""
    first = (
        "El costo y la dirección de entrega se confirman en el shipping-service "
        "en un paso posterior al cierre del carrito, nunca antes de que el "
        "buyer elija un método. "
    )
    assert len(first) > 280 // 2  # otherwise the cut loses too much
    text = first + "relleno " * 100
    out = _excerpt(text, limit=280)
    assert out == first.strip()
    assert "…" not in out


def test_excerpt_ignores_a_sentence_that_ends_too_early():
    """Cutting on it would throw away most of the excerpt, so keep reading."""
    text = "Corta. " + "relleno " * 100
    out = _excerpt(text, limit=280)
    assert out.endswith("…")
    assert len(out) > 200


def test_excerpt_stays_within_the_safety_valve():
    out = _excerpt("dato " * 2000)
    assert len(out) <= MAX_EXCERPT_CHARS + 1  # +1 for the ellipsis


def test_excerpt_does_not_pass_a_semicolon_off_as_a_full_stop():
    """A semicolon does not close the thought: the cut must be marked."""
    text = (
        "Cualquier cambio de flujo o de UI en esas fechas debe pasar por flag, "
        "sin excepciones para el equipo de checkout ni para promociones; "
    ) + "y la continuación sigue acá " * 20
    out = _excerpt(text, limit=280)
    assert not out.endswith(";")
    assert out.endswith("…")


# --- Local BM25 retriever ------


def _bm25_docs():
    """Three docs whose BM25 ordering for the query "carrito" is known.

    ``b`` is the shortest match, so BM25 length normalization ranks it above
    ``a``; ``c`` shares no term with the query and comes last.
    """
    return [
        Document(
            page_content="el carrito cierra el precio",
            metadata={"document_id": "a", "title": "A"},
        ),
        Document(page_content="el carrito", metadata={"document_id": "b", "title": "B"}),
        Document(
            page_content="flag de cyber monday",
            metadata={"document_id": "c", "title": "C"},
        ),
    ]


def test_local_bm25_orders_documents_by_score():
    retriever = LocalBM25Retriever.from_documents(_bm25_docs(), k=3)
    hits = retriever.invoke("carrito")
    assert [hit.metadata["document_id"] for hit in hits] == ["b", "a", "c"]


def test_local_bm25_respects_k_and_preserves_content_and_metadata():
    retriever = LocalBM25Retriever.from_documents(_bm25_docs(), k=2)
    hits = retriever.invoke("carrito")
    assert len(hits) == 2
    # The top hit is the original Document, content and metadata intact.
    assert hits[0].page_content == "el carrito"
    assert hits[0].metadata == {"document_id": "b", "title": "B"}


async def test_local_bm25_supports_async_invoke():
    """BaseRetriever's async path works, as EnsembleRetriever relies on it."""
    retriever = LocalBM25Retriever.from_documents(_bm25_docs(), k=1)
    hits = await retriever.ainvoke("carrito")
    assert len(hits) == 1
    assert hits[0].metadata["document_id"] == "b"


def test_local_bm25_handles_a_query_with_no_matching_terms():
    """A query nothing matches returns k documents instead of raising."""
    retriever = LocalBM25Retriever.from_documents(_bm25_docs(), k=2)
    hits = retriever.invoke("zzz inexistente")
    assert len(hits) == 2
    known = {"a", "b", "c"}
    assert all(hit.metadata["document_id"] in known for hit in hits)

