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
