from langchain_core.documents import Document
from retriever import retrieve


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
