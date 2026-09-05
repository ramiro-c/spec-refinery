from schemas import Citation
from scoring import (
    CYBER_TICKET,
    empty_slots,
    fanout,
    find_vague_hits,
    rank_questions,
    vaguedad_score,
)

def test_cyber_ticket_vague_hits():
    hits = find_vague_hits(CYBER_TICKET)
    assert "más rápido" in hits
    assert "tipo amazon" in hits

def test_cyber_ticket_has_empty_scope_and_metric_slots():
    slots = empty_slots(CYBER_TICKET)
    assert "criterio_medible" in slots
    assert "alcance" in slots

def test_fanout_finds_catalog_ids_only_when_named():
    assert fanout("tocamos cart-service y checkout-api") == [
        "cart-service",
        "checkout-api",
    ]
    assert fanout(CYBER_TICKET) == []

def test_rank_questions_collision_first_then_slot_then_vague():
    citations = [
        Citation(
            document_id="adr-cart-price.md",
            title="El precio se cierra en el carrito",
            excerpt="No se saltea.",
        )
    ]
    qs = rank_questions(CYBER_TICKET, citations)
    assert len(qs) == 3
    assert "adr-cart-price.md" in qs[0]
    assert "saltea" in qs[0].lower() or "carrito" in qs[0].lower()
    assert any("producto" in q.lower() for q in qs)
    assert any("más rápido" in q.lower() or "rapido" in q.lower() for q in qs)

def test_rank_questions_without_citations_has_no_collision():
    qs = rank_questions(CYBER_TICKET, [])
    assert all("adr-cart-price.md" not in q for q in qs)
    assert len(qs) <= 3

def test_vaguedad_score_is_hits_plus_empty_slots():
    text = CYBER_TICKET
    assert vaguedad_score(text) == len(find_vague_hits(text)) + len(empty_slots(text))
