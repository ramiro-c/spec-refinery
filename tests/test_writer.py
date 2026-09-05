"""Writer con LLM fake (RunnableLambda), estilo P4."""

from __future__ import annotations

from langchain_core.runnables import Runnable, RunnableLambda

from agents.writer import writer_turn
from schemas import AcceptanceCriterion, SpecDocument, SpecStatus
from scoring import CYBER_TICKET
from state import empty_spec, initial_fields


def _modelo_llm_fake(salida: dict):
    """LLM fake que devuelve SpecDocument vía with_structured_output."""

    class _ChatModelFake(Runnable):
        def invoke(self, mensajes, config=None, **kwargs):
            return salida

        def with_structured_output(self, schema):
            def _structured_invoke(mensajes, config=None, **kwargs):
                return schema.model_validate(salida)

            return RunnableLambda(_structured_invoke)

    return _ChatModelFake()


def test_writer_llm_reescribe_spec_y_respeta_pedido():
    modelo = _modelo_llm_fake(
        {
            "pedido": "no debe quedar",
            "que_entendimos": "Entendimos el cyber banner.",
            "criterios": [
                {"dado": "cyber activo", "cuando": "checkout", "entonces": "precio fijo"},
            ],
            "preguntas": ["ignorada"],
            "estado": {"se_puede_cerrar": False, "vaguedad": 1, "razon": "llm"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET),
        "questions": ["¿medimos?", "¿stock?", "¿envío?"],
    }
    out = writer_turn(state, modelo)

    spec = out["spec"]
    assert spec.pedido == CYBER_TICKET
    assert spec.que_entendimos == "Entendimos el cyber banner."
    assert len(spec.criterios) == 1
    assert spec.preguntas == ["¿medimos?", "¿stock?", "¿envío?"]
    assert out["last_agent"] == "writer"


def test_writer_close_requested_vacia_preguntas():
    modelo = _modelo_llm_fake(
        {
            "pedido": "x",
            "que_entendimos": "Cierre.",
            "criterios": [],
            "preguntas": ["no"],
            "estado": {"se_puede_cerrar": True, "vaguedad": 0, "razon": "llm"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET, close_requested=True),
        "questions": ["¿medimos?"],
        "spec": empty_spec(CYBER_TICKET),
    }
    out = writer_turn(state, modelo)

    assert out["spec"].preguntas == []
