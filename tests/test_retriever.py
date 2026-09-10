from langchain_core.documents import Document
from retriever import EXCERPT_CHARS, _excerpt, retrieve


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


def test_excerpt_collapses_the_line_breaks_of_a_chunk():
    assert _excerpt("Buyer — compra.\n\nCarrito — contiene ítems.") == (
        "Buyer — compra. Carrito — contiene ítems."
    )


def test_excerpt_never_cuts_a_word_in_half():
    """The bug: a hard slice produced tails like "alineado c"."""
    text = "palabra " * 200
    out = _excerpt(text)
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
    assert len(first) > EXCERPT_CHARS // 2  # otherwise the cut loses too much
    text = first + "relleno " * 100
    out = _excerpt(text)
    assert out == first.strip()
    assert "…" not in out


def test_excerpt_ignores_a_sentence_that_ends_too_early():
    """Cutting on it would throw away most of the excerpt, so keep reading."""
    text = "Corta. " + "relleno " * 100
    out = _excerpt(text)
    assert out.endswith("…")
    assert len(out) > 200


def test_excerpt_stays_within_the_limit():
    out = _excerpt("dato " * 500)
    assert len(out) <= EXCERPT_CHARS + 1  # +1 for the ellipsis


def test_excerpt_does_not_pass_a_semicolon_off_as_a_full_stop():
    """A semicolon does not close the thought: the cut must be marked."""
    text = (
        "Cualquier cambio de flujo o de UI en esas fechas debe pasar por flag, "
        "sin excepciones para el equipo de checkout ni para promociones; "
    ) + "y la continuación sigue acá " * 20
    out = _excerpt(text)
    assert not out.endswith(";")
    assert out.endswith("…")
