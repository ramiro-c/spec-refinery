import pytest
from pydantic import ValidationError

from schemas import (
    AcceptanceCriterion,
    Citation,
    DocumentAssessment,
    Interrogation,
    SpecDocument,
    SpecStatus,
)


def test_spec_document_roundtrip_empty_boxes():
    spec = SpecDocument(
        request="ticket",
        understanding="",
        clashes=[],
        services=[],
        criteria=[],
        questions=[],
        status=SpecStatus(can_close=False, vagueness=0, reason="inicio"),
    )
    dumped = spec.model_dump()
    assert dumped["request"] == "ticket"
    assert dumped["questions"] == []
    again = SpecDocument.model_validate(dumped)
    assert again.status.can_close is False


def test_spec_document_missing_closed_defaults_to_false():
    """Schema default: a dict without `closed` validates as an open spec.

    This exercises ``model_validate`` on an in-memory dict only; it does NOT
    run the LangGraph checkpoint codec.
    """
    spec = SpecDocument(
        request="ticket",
        status=SpecStatus(can_close=False, vagueness=0, reason="inicio"),
    )
    assert spec.closed is False

    dumped = spec.model_dump()
    del dumped["closed"]
    revived = SpecDocument.model_validate(dumped)
    assert revived.closed is False


def test_spec_document_context_defaults_to_empty():
    """`context` is independent from `clashes` and round-trips its own data."""
    spec = SpecDocument(
        request="ticket",
        clashes=[Citation(document_id="adr-cart-price.md", title="precio")],
        status=SpecStatus(can_close=False, vagueness=0, reason="inicio"),
    )
    assert spec.context == []

    spec.context = [Citation(document_id="adr-stock-reserve.md", title="stock")]
    dumped = spec.model_dump()
    assert dumped["context"][0]["document_id"] == "adr-stock-reserve.md"
    revived = SpecDocument.model_validate(dumped)
    assert revived.context[0].document_id == "adr-stock-reserve.md"
    assert revived.clashes[0].document_id == "adr-cart-price.md"


def test_spec_document_revives_dump_without_context():
    """Backward compatibility: an old dump lacking `context` still revives."""
    spec = SpecDocument(
        request="ticket",
        status=SpecStatus(can_close=False, vagueness=0, reason="inicio"),
    )
    dumped = spec.model_dump()
    del dumped["context"]
    revived = SpecDocument.model_validate(dumped)
    assert revived.context == []


def test_interrogation_classifications_defaults_to_empty():
    """Declarations are optional, and the close flag stays at index 0."""
    verdict = Interrogation(
        can_close=False,
        vagueness=0,
        reason="inicio",
        classifications=[
            DocumentAssessment(document_id="adr-cart-price.md", kind="clash")
        ],
    )
    assert [(c.document_id, c.kind) for c in verdict.classifications] == [
        ("adr-cart-price.md", "clash")
    ]
    assert list(Interrogation.model_fields)[0] == "human_wants_close"
    bare = Interrogation(can_close=False, vagueness=0, reason="x")
    assert bare.classifications == []


def test_document_assessment_rejects_unknown_kind():
    assert DocumentAssessment(document_id="x", kind="context").kind == "context"
    assert DocumentAssessment(document_id="x", kind="clash").kind == "clash"
    with pytest.raises(ValidationError):
        DocumentAssessment(document_id="x", kind="ni-idea")
