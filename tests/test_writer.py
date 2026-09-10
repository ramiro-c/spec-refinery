"""Writer with a fake LLM (RunnableLambda), P4 style."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from agents.writer import writer_turn
from config import MAX_ROUNDS
from demo import CYBER_TICKET
from schemas import Decision, SpecStatus
from state import empty_spec, initial_fields


def _modelo_llm_fake(salida: dict, *, capturar=None):
    """Fake LLM that returns a SpecDocument via with_structured_output."""

    class _ChatModelFake(Runnable):
        def invoke(self, mensajes, config=None, **kwargs):
            return salida

        def with_structured_output(self, schema):
            def _structured_invoke(mensajes, config=None, **kwargs):
                if capturar is not None:
                    capturar["mensajes"] = mensajes
                return schema.model_validate(salida)

            return RunnableLambda(_structured_invoke)

    return _ChatModelFake()


async def test_writer_llm_reescribe_spec_y_respeta_pedido():
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
    out = await writer_turn(state, modelo)

    spec = out["spec"]
    assert spec.pedido == CYBER_TICKET
    assert spec.que_entendimos == "Entendimos el cyber banner."
    assert len(spec.criterios) == 1
    assert spec.preguntas == ["¿medimos?", "¿stock?", "¿envío?"]
    assert out["last_agent"] == "writer"


async def test_writer_close_requested_vacia_preguntas():
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
    out = await writer_turn(state, modelo)

    assert out["spec"].preguntas == []


async def test_writer_ve_respuesta_pm_y_spec_anterior():
    """The prompt includes the PM follow-up and the previous spec."""
    capturar: dict = {}
    respuesta_pm = "para todo, no medimos"
    modelo = _modelo_llm_fake(
        {
            "pedido": "x",
            "que_entendimos": f"Entendimos: {respuesta_pm}",
            "criterios": [],
            "preguntas": [],
            "estado": {"se_puede_cerrar": False, "vaguedad": 1, "razon": "llm"},
        },
        capturar=capturar,
    )
    prior = empty_spec(CYBER_TICKET)
    prior.que_entendimos = "Versión anterior."
    state = {
        **initial_fields(CYBER_TICKET),
        "messages": [
            HumanMessage(content=CYBER_TICKET),
            HumanMessage(content=respuesta_pm),
        ],
        "spec": prior,
        "questions": ["¿medimos?"],
    }
    out = await writer_turn(state, modelo)

    human_msg = capturar["mensajes"][-1].content
    assert respuesta_pm in human_msg
    assert "Versión anterior." in human_msg
    assert out["spec"].que_entendimos == f"Entendimos: {respuesta_pm}"


async def test_writer_carries_the_interrogator_verdict_not_its_own():
    """The writer never scores: estado belongs to the interrogator."""
    modelo = _modelo_llm_fake(
        {
            "pedido": "x",
            "que_entendimos": "Comprar ahora acorta la pantalla para SKU 1P.",
            "criterios": [],
            "preguntas": [],
            # The drafting LLM guesses an estado; it must be discarded.
            "estado": {"se_puede_cerrar": True, "vaguedad": 0, "razon": "invento"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET),
        "questions": [],
        "assessment": SpecStatus(
            se_puede_cerrar=False, vaguedad=4, razon="falta el alcance"
        ),
    }
    out = await writer_turn(state, modelo)

    assert out["spec"].estado.vaguedad == 4
    assert out["spec"].estado.se_puede_cerrar is False
    assert out["spec"].estado.razon == "falta el alcance"


async def test_writer_falls_back_to_the_last_verdict_on_an_explicit_close():
    """On /close the interrogator does not run, so the last verdict stands."""
    modelo = _modelo_llm_fake(
        {
            "pedido": "x",
            "que_entendimos": "Cierre.",
            "criterios": [],
            "preguntas": [],
            "estado": {"se_puede_cerrar": True, "vaguedad": 0, "razon": "invento"},
        }
    )
    prior = empty_spec(CYBER_TICKET)
    prior.estado = SpecStatus(
        se_puede_cerrar=False, vaguedad=6, razon="quedaban huecos"
    )
    state = {
        **initial_fields(CYBER_TICKET, close_requested=True),
        "spec": prior,
        "assessment": None,
    }
    out = await writer_turn(state, modelo)

    assert out["spec"].preguntas == []
    assert out["spec"].estado.vaguedad == 6
    # The document closes; the status stays honest about what was left open.
    assert "Cerrada a pedido tuyo" in out["spec"].estado.razon
    assert "vaguedad 6" in out["spec"].estado.razon


async def test_writer_drops_services_outside_the_catalog():
    modelo = _modelo_llm_fake(
        {
            "pedido": "x",
            "que_entendimos": "y",
            "servicios": ["cart-service", "servicio-inventado"],
            "criterios": [],
            "preguntas": [],
            "estado": {"se_puede_cerrar": False, "vaguedad": 3, "razon": "llm"},
        }
    )
    state = {**initial_fields(CYBER_TICKET), "questions": []}
    out = await writer_turn(state, modelo)

    assert out["spec"].servicios == ["cart-service"]


async def test_writer_accumulates_decisions_across_rounds():
    """A rule the PM overruled stays settled instead of resurfacing."""
    modelo = _modelo_llm_fake(
        {
            "pedido": "x",
            "que_entendimos": "y",
            "criterios": [],
            "preguntas": [],
            "estado": {"se_puede_cerrar": False, "vaguedad": 3, "razon": "llm"},
        }
    )
    prior = empty_spec(CYBER_TICKET)
    prior.decisiones = [
        Decision(tema="reserva de stock", decision="se reserva al confirmar")
    ]
    state = {
        **initial_fields(CYBER_TICKET),
        "spec": prior,
        "decisiones": [
            Decision(
                tema="adr-cart-price",
                decision="se deprecia para el flujo nuevo",
                impacto="hay que reescribir la ADR",
            )
        ],
    }
    out = await writer_turn(state, modelo)

    temas = [d.tema for d in out["spec"].decisiones]
    assert temas == ["reserva de stock", "adr-cart-price"]


async def test_writer_lets_a_new_decision_supersede_the_same_topic():
    modelo = _modelo_llm_fake(
        {
            "pedido": "x",
            "que_entendimos": "y",
            "criterios": [],
            "preguntas": [],
            "estado": {"se_puede_cerrar": False, "vaguedad": 3, "razon": "llm"},
        }
    )
    prior = empty_spec(CYBER_TICKET)
    prior.decisiones = [Decision(tema="Alcance", decision="todos los SKU")]
    state = {
        **initial_fields(CYBER_TICKET),
        "spec": prior,
        "decisiones": [Decision(tema="alcance", decision="solo SKU 1P de electro")],
    }
    out = await writer_turn(state, modelo)

    assert len(out["spec"].decisiones) == 1
    assert out["spec"].decisiones[0].decision == "solo SKU 1P de electro"


async def test_writer_freezes_the_spec_when_the_rounds_run_out():
    """The round budget is a promise to the PM: it closes on its own."""
    modelo = _modelo_llm_fake(
        {
            "pedido": "x",
            "que_entendimos": "y",
            "criterios": [],
            "preguntas": [],
            "estado": {"se_puede_cerrar": True, "vaguedad": 0, "razon": "invento"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET),
        "questions": ["¿Y el stock?"],
        "round_count": MAX_ROUNDS,
        "assessment": SpecStatus(se_puede_cerrar=False, vaguedad=6, razon="x"),
    }
    out = await writer_turn(state, modelo)

    assert out["spec"].preguntas == []
    assert f"se agotaron las {MAX_ROUNDS} rondas" in out["spec"].estado.razon
