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
        pedido="ticket",
        que_entendimos="",
        choques=[],
        servicios=[],
        criterios=[],
        preguntas=[],
        estado=SpecStatus(se_puede_cerrar=False, vaguedad=0, razon="inicio"),
    )
    dumped = spec.model_dump()
    assert dumped["pedido"] == "ticket"
    assert dumped["preguntas"] == []
    again = SpecDocument.model_validate(dumped)
    assert again.estado.se_puede_cerrar is False


def test_spec_document_missing_cerrada_defaults_to_false():
    """Schema default: a dict without `cerrada` validates as an open spec.

    This exercises ``model_validate`` on an in-memory dict only; it does NOT
    run the LangGraph checkpoint codec.
    """
    spec = SpecDocument(
        pedido="ticket",
        estado=SpecStatus(se_puede_cerrar=False, vaguedad=0, razon="inicio"),
    )
    assert spec.cerrada is False

    dumped = spec.model_dump()
    del dumped["cerrada"]
    revived = SpecDocument.model_validate(dumped)
    assert revived.cerrada is False


def test_spec_document_contexto_defaults_to_empty():
    """`contexto` is independent from `choques` and round-trips its own data."""
    spec = SpecDocument(
        pedido="ticket",
        choques=[Citation(document_id="adr-cart-price.md", title="precio")],
        estado=SpecStatus(se_puede_cerrar=False, vaguedad=0, razon="inicio"),
    )
    assert spec.contexto == []

    spec.contexto = [Citation(document_id="adr-stock-reserve.md", title="stock")]
    dumped = spec.model_dump()
    assert dumped["contexto"][0]["document_id"] == "adr-stock-reserve.md"
    revived = SpecDocument.model_validate(dumped)
    assert revived.contexto[0].document_id == "adr-stock-reserve.md"
    assert revived.choques[0].document_id == "adr-cart-price.md"


def test_spec_document_revives_dump_without_contexto():
    """Backward compatibility: an old dump lacking `contexto` still revives."""
    spec = SpecDocument(
        pedido="ticket",
        estado=SpecStatus(se_puede_cerrar=False, vaguedad=0, razon="inicio"),
    )
    dumped = spec.model_dump()
    del dumped["contexto"]
    revived = SpecDocument.model_validate(dumped)
    assert revived.contexto == []


def test_interrogation_clasificaciones_defaults_to_empty():
    """Declarations are optional, and the close flag stays at index 0."""
    verdict = Interrogation(
        se_puede_cerrar=False,
        vaguedad=0,
        razon="inicio",
        clasificaciones=[
            DocumentAssessment(document_id="adr-cart-price.md", tipo="choque")
        ],
    )
    assert [(c.document_id, c.tipo) for c in verdict.clasificaciones] == [
        ("adr-cart-price.md", "choque")
    ]
    assert list(Interrogation.model_fields)[0] == "human_wants_close"
    bare = Interrogation(se_puede_cerrar=False, vaguedad=0, razon="x")
    assert bare.clasificaciones == []


def test_document_assessment_rejects_unknown_tipo():
    assert DocumentAssessment(document_id="x", tipo="contexto").tipo == "contexto"
    assert DocumentAssessment(document_id="x", tipo="choque").tipo == "choque"
    with pytest.raises(ValidationError):
        DocumentAssessment(document_id="x", tipo="ni-idea")
