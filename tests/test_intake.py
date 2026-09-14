"""Interrogator node with a fake LLM: no question catalog, no keyword scoring."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from agents.intake import interrogate
from config import MAX_QUESTIONS_PER_ROUND, MAX_ROUNDS
from demo import CYBER_TICKET
from schemas import Citation, Decision
from state import empty_spec, initial_fields


def _fake_llm(output: dict, *, captured: dict | None = None):
    class _ChatModelFake(Runnable):
        def invoke(self, messages, config=None, **kwargs):
            return output

        def with_structured_output(self, schema):
            def _structured(messages, config=None, **kwargs):
                if captured is not None:
                    captured["messages"] = messages
                return schema.model_validate(output)

            return RunnableLambda(_structured)

    return _ChatModelFake()


async def test_interrogator_respects_the_question_budget():
    """The budget is a promise shown to the PM, not a hint to the model."""
    llm = _fake_llm(
        {
            "questions": [f"¿Pregunta {i}?" for i in range(1, 8)],
            "can_close": False,
            "vagueness": 8,
            "reason": "falta casi todo",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert len(out["questions"]) == MAX_QUESTIONS_PER_ROUND
    assert out["questions"][0] == "¿Pregunta 1?"
    assert out["assessment"].vagueness == 8
    assert out["grilled"] is True
    assert out["round_count"] == 1


async def test_the_final_round_asks_nothing_and_just_rules():
    llm = _fake_llm(
        {
            "questions": ["¿Una más?"],
            "can_close": False,
            "vagueness": 5,
            "reason": "se acabaron las rondas",
        }
    )
    state = {**initial_fields(CYBER_TICKET), "round_count": MAX_ROUNDS - 1}
    out = await interrogate(state, llm)

    assert out["questions"] == []
    assert out["round_count"] == MAX_ROUNDS


async def test_the_round_budget_reaches_the_prompt():
    captured: dict = {}
    llm = _fake_llm(
        {
            "questions": [],
            "can_close": False,
            "vagueness": 5,
            "reason": "x",
        },
        captured=captured,
    )
    await interrogate({**initial_fields(CYBER_TICKET), "round_count": 1}, llm)

    prompt = captured["messages"][-1].content
    assert f"Round 2 of {MAX_ROUNDS}" in prompt
    assert f"up to {MAX_QUESTIONS_PER_ROUND} questions" in prompt


async def test_the_prompt_requires_glossary_first_disambiguation():
    """The interrogator interprets retrieved terms before calling a rule contradicted."""
    captured: dict = {}
    llm = _fake_llm(
        {
            "questions": [],
            "can_close": False,
            "vagueness": 0,
            "reason": "x",
        },
        captured=captured,
    )
    await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    prompt = captured["messages"][0].content
    assert "interpret the request's terms" in prompt
    assert "no real conflict once interpreted per the glossary" in prompt
    assert "state the benign reading" in prompt
    assert "the clash is REAL" in prompt


async def test_the_prompt_preserves_the_three_resolutions_and_document_naming():
    """The three PM resolutions, document naming and decision recording survive."""
    captured: dict = {}
    llm = _fake_llm(
        {
            "questions": [],
            "can_close": False,
            "vagueness": 0,
            "reason": "x",
        },
        captured=captured,
    )
    await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    prompt = captured["messages"][0].content
    assert "the PM adapts the request to the rule" in prompt
    assert "the PM takes a scoped exception" in prompt
    assert "name the document" in prompt
    assert "decisions" in prompt


async def test_interrogator_questions_land_in_the_transcript():
    """The next round reads what it already asked from the conversation."""
    llm = _fake_llm(
        {
            "questions": ["¿Qué productos entran?"],
            "can_close": False,
            "vagueness": 5,
            "reason": "falta alcance",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    posted = out["messages"][0]
    assert posted.name == "intake"
    assert "¿Qué productos entran?" in posted.content


async def test_interrogator_sees_the_rules_and_the_whole_conversation():
    captured: dict = {}
    llm = _fake_llm(
        {
            "questions": [],
            "can_close": True,
            "vagueness": 0,
            "reason": "cerrada",
        },
        captured=captured,
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

    prompt = captured["messages"][-1].content
    assert "adr-cart-price.md" in prompt
    assert "PM: solo SKU 1P" in prompt
    assert "Refinery: ¿Qué productos entran?" in prompt
    # Writer bookkeeping is graph noise, not dialogue.
    assert "Spec updated" not in prompt


async def test_interrogator_can_close_when_the_pm_asks():
    llm = _fake_llm(
        {
            "questions": [],
            "can_close": False,
            "vagueness": 4,
            "reason": "el PM pidió cerrar igual",
            "human_wants_close": True,
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert out["close_requested"] is True


async def test_interrogator_does_not_touch_close_when_the_pm_did_not_ask():
    llm = _fake_llm(
        {
            "questions": ["¿Alcance?"],
            "can_close": False,
            "vagueness": 4,
            "reason": "sigue abierto",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert "close_requested" not in out


async def test_interrogator_records_a_rule_the_pm_overruled():
    """A retrieved rule is evidence, not law: the PM can decide to change it."""
    llm = _fake_llm(
        {
            "human_wants_close": False,
            "decisions": [
                {
                    "topic": "adr-cart-price",
                    "decision": "se deprecia para el flujo de Cyber Monday",
                    "impact": "hay que reescribir la ADR",
                }
            ],
            "questions": ["¿Y las promos, dónde se aplican?"],
            "can_close": False,
            "vagueness": 3,
            "reason": "queda el tema promos",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert [d.topic for d in out["decisions"]] == ["adr-cart-price"]


async def test_interrogator_is_told_what_is_already_settled():
    """Settled topics reach the prompt so they are not re-litigated."""
    captured: dict = {}
    llm = _fake_llm(
        {
            "human_wants_close": False,
            "decisions": [],
            "questions": [],
            "can_close": True,
            "vagueness": 0,
            "reason": "listo",
        },
        captured=captured,
    )
    spec = empty_spec(CYBER_TICKET)
    spec.decisions = [
        Decision(topic="adr-cart-price", decision="deprecada para el flujo nuevo")
    ]
    await interrogate({**initial_fields(CYBER_TICKET), "spec": spec}, llm)

    prompt = captured["messages"][-1].content
    assert "Already settled" in prompt
    assert "adr-cart-price: deprecada para el flujo nuevo" in prompt


async def test_close_is_the_humans_call_even_on_an_unfinished_spec():
    """The interrogator may judge the spec unready and still must let it close."""
    llm = _fake_llm(
        {
            "human_wants_close": True,
            "decisions": [],
            "questions": ["¿Y el stock?"],
            "can_close": False,
            "vagueness": 7,
            "reason": "quedan huecos, pero el PM pidió cerrar",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert out["close_requested"] is True
    assert out["assessment"].can_close is False


def test_close_flag_is_answered_before_any_quality_judgement():
    """Field order matters: the fact must not be contaminated by the opinion."""
    from schemas import Interrogation

    assert list(Interrogation.model_fields)[0] == "human_wants_close"


async def test_the_prompt_requires_one_declaration_per_retrieved_document():
    """The prompt demands a clash/context type for every retrieved document."""
    captured: dict = {}
    llm = _fake_llm(
        {
            "questions": [],
            "can_close": False,
            "vagueness": 0,
            "reason": "x",
        },
        captured=captured,
    )
    await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    prompt = captured["messages"][0].content
    assert "Declare a type for EVERY retrieved document" in prompt
    assert "classifications" in prompt
    assert "never invent one" in prompt


async def test_interrogator_emits_the_per_document_classification():
    """A declaration per retrieved document travels out in the node update."""
    llm = _fake_llm(
        {
            "questions": [],
            "can_close": False,
            "vagueness": 3,
            "reason": "x",
            "classifications": [
                {"document_id": "adr-cart-price.md", "kind": "clash"},
                {"document_id": "spec-cyber-banner.md", "kind": "context"},
            ],
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert [(c.document_id, c.kind) for c in out["classifications"]] == [
        ("adr-cart-price.md", "clash"),
        ("spec-cyber-banner.md", "context"),
    ]


async def test_interrogator_defaults_classifications_to_empty():
    """A model that classifies nothing must not fabricate declarations."""
    llm = _fake_llm(
        {
            "questions": ["¿Alcance?"],
            "can_close": False,
            "vagueness": 2,
            "reason": "x",
        }
    )
    out = await interrogate({**initial_fields(CYBER_TICKET)}, llm)

    assert out["classifications"] == []
