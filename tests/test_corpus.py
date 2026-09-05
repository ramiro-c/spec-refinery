from pathlib import Path
from catalog import SERVICE_IDS

CORPUS = Path(__file__).resolve().parents[1] / "corpus"

def test_eleven_markdown_docs_exist():
    files = list(CORPUS.rglob("*.md"))
    assert len(files) == 11

def test_cart_price_rule_has_stable_id_and_no_skip():
    text = (CORPUS / "rules" / "adr-cart-price.md").read_text()
    assert "document_id: adr-cart-price.md" in text
    assert "no se saltea" in text.lower() or "No se saltea" in text

def test_catalog_lists_every_service_id():
    text = (CORPUS / "catalog.md").read_text()
    for sid in SERVICE_IDS:
        assert sid in text
