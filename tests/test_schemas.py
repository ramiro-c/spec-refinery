from schemas import AcceptanceCriterion, Citation, SpecDocument, SpecStatus


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
