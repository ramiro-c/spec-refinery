"""Interrogator node with a fake LLM: no question catalog, no keyword scoring."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from agents.intake import interrogate
from config import MAX_QUESTIONS_PER_ROUND, MAX_ROUNDS
from demo import CYBER_TICKET
from schemas import Citation, Decision
from state import empty_spec, initial_fields


def _fake_llm(salida: dict, *, capturar: dict | None = None):
    class _ChatModelFake(Runnable):
        def invoke(self, mensajes, config=None, **kwargs):
            return salida

        def with_structured_output(self, schema):
            def _structured(mensajes, config=None, **kwargs):
                if capturar is not None:
                    capturar["mensajes"] = mensajes
                return schema.model_validate(salida)

            return RunnableLambda(_structured)

    return _ChatModelFake()


async def test_interrogator_respects_the_question_budget():
    """The budget is a promise shown to the PM, not a hint to the model."""
    llm = _fake_llm(
        {
            "preguntas": [f"¿Pregunta {i}?" for i in range(1, 8)],
            "se_puede_cerrar": False,
            "vaguedad": 8,
            "razon": "falta casi todo",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert len(out["questions"]) == MAX_QUESTIONS_PER_ROUND
    assert out["questions"][0] == "¿Pregunta 1?"
    assert out["assessment"].vaguedad == 8
    assert out["grilled"] is True
    assert out["round_count"] == 1


async def test_the_final_round_asks_nothing_and_just_rules():
    llm = _fake_llm(
        {
            "preguntas": ["¿Una más?"],
            "se_puede_cerrar": False,
            "vaguedad": 5,
            "razon": "se acabaron las rondas",
        }
    )
    state = {**initial_fields(CYBER_TICKET), "round_count": MAX_ROUNDS - 1}
    out = await interrogate(state, llm)

    assert out["questions"] == []
    assert out["round_count"] == MAX_ROUNDS


async def test_the_round_budget_reaches_the_prompt():
    capturar: dict = {}
    llm = _fake_llm(
        {
            "preguntas": [],
            "se_puede_cerrar": False,
            "vaguedad": 5,
            "razon": "x",
        },
        capturar=capturar,
    )
    await interrogate({**initial_fields(CYBER_TICKET), "round_count": 1}, llm)

    prompt = capturar["mensajes"][-1].content
    assert f"Round 2 of {MAX_ROUNDS}" in prompt
    assert f"up to {MAX_QUESTIONS_PER_ROUND} questions" in prompt


async def test_the_prompt_requires_glossary_first_disambiguation():
    """The interrogator interprets retrieved terms before calling a rule contradicted."""
    capturar: dict = {}
    llm = _fake_llm(
        {
            "preguntas": [],
            "se_puede_cerrar": False,
            "vaguedad": 0,
            "razon": "x",
        },
        capturar=capturar,
    )
    await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    prompt = capturar["mensajes"][0].content
    assert "interpret the request's terms" in prompt
    assert "no real conflict once interpreted per the glossary" in prompt
    assert "state the benign reading" in prompt
    assert "the clash is REAL" in prompt


async def test_the_prompt_preserves_the_three_resolutions_and_document_naming():
    """The three PM resolutions, document naming and decision recording survive."""
    capturar: dict = {}
    llm = _fake_llm(
        {
            "preguntas": [],
            "se_puede_cerrar": False,
            "vaguedad": 0,
            "razon": "x",
        },
        capturar=capturar,
    )
    await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    prompt = capturar["mensajes"][0].content
    assert "the PM adapts the request to the rule" in prompt
    assert "the PM takes a scoped exception" in prompt
    assert "name the document" in prompt
    assert "decisiones" in prompt


async def test_interrogator_questions_land_in_the_transcript():
    """The next round reads what it already asked from the conversation."""
    llm = _fake_llm(
        {
            "preguntas": ["¿Qué productos entran?"],
            "se_puede_cerrar": False,
            "vaguedad": 5,
            "razon": "falta alcance",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    posted = out["messages"][0]
    assert posted.name == "intake"
    assert "¿Qué productos entran?" in posted.content


async def test_interrogator_sees_the_rules_and_the_whole_conversation():
    capturar: dict = {}
    llm = _fake_llm(
        {
            "preguntas": [],
            "se_puede_cerrar": True,
            "vaguedad": 0,
            "razon": "cerrada",
        },
        capturar=capturar,
    )
    state = {
        **initial_fields(CYBER_TICKET),
        "citations": [
            Citation(
                document_id="adr-cart-price.md",
                title="El precio se cierra en el carrito",
                excerpt="No se saltea.",
            )
        ],
        "messages": [
            HumanMessage(content=CYBER_TICKET),
            AIMessage(content="¿Qué productos entran?", name="intake"),
            HumanMessage(content="solo SKU 1P"),
            AIMessage(content="Spec updated", name="writer"),
        ],
    }
    await interrogate(state, llm)

    prompt = capturar["mensajes"][-1].content
    assert "adr-cart-price.md" in prompt
    assert "PM: solo SKU 1P" in prompt
    assert "Refinery: ¿Qué productos entran?" in prompt
    # Writer bookkeeping is graph noise, not dialogue.
    assert "Spec updated" not in prompt


async def test_interrogator_can_close_when_the_pm_asks():
    llm = _fake_llm(
        {
            "preguntas": [],
            "se_puede_cerrar": False,
            "vaguedad": 4,
            "razon": "el PM pidió cerrar igual",
            "human_wants_close": True,
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert out["close_requested"] is True


async def test_interrogator_does_not_touch_close_when_the_pm_did_not_ask():
    llm = _fake_llm(
        {
            "preguntas": ["¿Alcance?"],
            "se_puede_cerrar": False,
            "vaguedad": 4,
            "razon": "sigue abierto",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert "close_requested" not in out


async def test_interrogator_records_a_rule_the_pm_overruled():
    """A retrieved rule is evidence, not law: the PM can decide to change it."""
    llm = _fake_llm(
        {
            "human_wants_close": False,
            "decisiones": [
                {
                    "tema": "adr-cart-price",
                    "decision": "se deprecia para el flujo de Cyber Monday",
                    "impacto": "hay que reescribir la ADR",
                }
            ],
            "preguntas": ["¿Y las promos, dónde se aplican?"],
            "se_puede_cerrar": False,
            "vaguedad": 3,
            "razon": "queda el tema promos",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert [d.tema for d in out["decisiones"]] == ["adr-cart-price"]


async def test_interrogator_is_told_what_is_already_settled():
    """Settled topics reach the prompt so they are not re-litigated."""
    capturar: dict = {}
    llm = _fake_llm(
        {
            "human_wants_close": False,
            "decisiones": [],
            "preguntas": [],
            "se_puede_cerrar": True,
            "vaguedad": 0,
            "razon": "listo",
        },
        capturar=capturar,
    )
    spec = empty_spec(CYBER_TICKET)
    spec.decisiones = [
        Decision(tema="adr-cart-price", decision="deprecada para el flujo nuevo")
    ]
    await interrogate({**initial_fields(CYBER_TICKET), "spec": spec}, llm)

    prompt = capturar["mensajes"][-1].content
    assert "Already settled" in prompt
    assert "adr-cart-price: deprecada para el flujo nuevo" in prompt


async def test_close_is_the_humans_call_even_on_an_unfinished_spec():
    """The interrogator may judge the spec unready and still must let it close."""
    llm = _fake_llm(
        {
            "human_wants_close": True,
            "decisiones": [],
            "preguntas": ["¿Y el stock?"],
            "se_puede_cerrar": False,
            "vaguedad": 7,
            "razon": "quedan huecos, pero el PM pidió cerrar",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert out["close_requested"] is True
    assert out["assessment"].se_puede_cerrar is False


def test_close_flag_is_answered_before_any_quality_judgement():
    """Field order matters: the fact must not be contaminated by the opinion."""
    from schemas import Interrogation

    assert list(Interrogation.model_fields)[0] == "human_wants_close"


async def test_the_prompt_requires_one_declaration_per_retrieved_document():
    """The prompt demands a choque/contexto type for every retrieved document."""
    capturar: dict = {}
    llm = _fake_llm(
        {
            "preguntas": [],
            "se_puede_cerrar": False,
            "vaguedad": 0,
            "razon": "x",
        },
        capturar=capturar,
    )
    await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    prompt = capturar["mensajes"][0].content
    assert "Declare a type for EVERY retrieved document" in prompt
    assert "clasificaciones" in prompt
    assert "never invent one" in prompt


async def test_interrogator_emits_the_per_document_classification():
    """A declaration per retrieved document travels out in the node update."""
    llm = _fake_llm(
        {
            "preguntas": [],
            "se_puede_cerrar": False,
            "vaguedad": 3,
            "razon": "x",
            "clasificaciones": [
                {"document_id": "adr-cart-price.md", "tipo": "choque"},
                {"document_id": "spec-cyber-banner.md", "tipo": "contexto"},
            ],
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert [(c.document_id, c.tipo) for c in out["clasificaciones"]] == [
        ("adr-cart-price.md", "choque"),
        ("spec-cyber-banner.md", "contexto"),
    ]


async def test_interrogator_defaults_clasificaciones_to_empty():
    """A model that classifies nothing must not fabricate declarations."""
    llm = _fake_llm(
        {
            "preguntas": ["¿Alcance?"],
            "se_puede_cerrar": False,
            "vaguedad": 2,
            "razon": "x",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert out["clasificaciones"] == []
