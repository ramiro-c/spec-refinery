"""Writer with a fake LLM (RunnableLambda), P4 style."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from agents.writer import writer_turn
from config import MAX_ROUNDS
from demo import CYBER_TICKET
from schemas import Citation, Decision, DocumentAssessment, SpecStatus
from state import empty_spec, initial_fields


def _fake_llm_model(output: dict, *, captured=None):
    """Fake LLM that returns a SpecDocument via with_structured_output."""

    class _ChatModelFake(Runnable):
        def invoke(self, messages, config=None, **kwargs):
            return output

        def with_structured_output(self, schema):
            def _structured_invoke(messages, config=None, **kwargs):
                if captured is not None:
                    captured["messages"] = messages
                return schema.model_validate(output)

            return RunnableLambda(_structured_invoke)

    return _ChatModelFake()


def _citation(document_id: str) -> Citation:
    return Citation(document_id=document_id, title=document_id, excerpt="regla")


def _draft() -> dict:
    return {
        "request": "x",
        "understanding": "y",
        "criteria": [],
        "questions": [],
        "status": {"can_close": False, "vagueness": 3, "reason": "llm"},
    }


async def test_writer_llm_rewrites_spec_and_respects_ticket():
    model = _fake_llm_model(
        {
            "request": "no debe quedar",
            "understanding": "Entendimos el cyber banner.",
            "criteria": [
                {"given": "cyber activo", "when": "checkout", "then": "precio fijo"},
            ],
            "questions": ["ignorada"],
            "status": {"can_close": False, "vagueness": 1, "reason": "llm"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET),
        "questions": ["¿medimos?", "¿stock?", "¿envío?"],
    }
    out = await writer_turn(state, model)

    spec = out["spec"]
    assert spec.request == CYBER_TICKET
    assert spec.understanding == "Entendimos el cyber banner."
    assert len(spec.criteria) == 1
    assert spec.questions == ["¿medimos?", "¿stock?", "¿envío?"]
    assert out["last_agent"] == "writer"


async def test_writer_close_requested_clears_questions():
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "Cierre.",
            "criteria": [],
            "questions": ["no"],
            "status": {"can_close": True, "vagueness": 0, "reason": "llm"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET, close_requested=True),
        "questions": ["¿medimos?"],
        "spec": empty_spec(CYBER_TICKET),
    }
    out = await writer_turn(state, model)

    assert out["spec"].questions == []


async def test_writer_sees_pm_reply_and_previous_spec():
    """The prompt includes the PM follow-up and the previous spec."""
    captured: dict = {}
    pm_reply = "para todo, no medimos"
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": f"Entendimos: {pm_reply}",
            "criteria": [],
            "questions": [],
            "status": {"can_close": False, "vagueness": 1, "reason": "llm"},
        },
        captured=captured,
    )
    prior = empty_spec(CYBER_TICKET)
    prior.understanding = "Versión anterior."
    state = {
        **initial_fields(CYBER_TICKET),
        "messages": [
            HumanMessage(content=CYBER_TICKET),
            HumanMessage(content=pm_reply),
        ],
        "spec": prior,
        "questions": ["¿medimos?"],
    }
    out = await writer_turn(state, model)

    human_msg = captured["messages"][-1].content
    assert pm_reply in human_msg
    assert "Versión anterior." in human_msg
    assert out["spec"].understanding == f"Entendimos: {pm_reply}"


async def test_writer_carries_the_interrogator_verdict_not_its_own():
    """The writer never scores: status belongs to the interrogator."""
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "Comprar ahora acorta la pantalla para SKU 1P.",
            "criteria": [],
            "questions": [],
            # The drafting LLM guesses a status; it must be discarded.
            "status": {"can_close": True, "vagueness": 0, "reason": "invento"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET),
        "questions": [],
        "assessment": SpecStatus(
            can_close=False, vagueness=4, reason="falta el alcance"
        ),
    }
    out = await writer_turn(state, model)

    assert out["spec"].status.vagueness == 4
    assert out["spec"].status.can_close is False
    assert out["spec"].status.reason == "falta el alcance"


async def test_writer_falls_back_to_the_last_verdict_on_an_explicit_close():
    """On /close the interrogator does not run, so the last verdict stands."""
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "Cierre.",
            "criteria": [],
            "questions": [],
            "status": {"can_close": True, "vagueness": 0, "reason": "invento"},
        }
    )
    prior = empty_spec(CYBER_TICKET)
    prior.status = SpecStatus(
        can_close=False, vagueness=6, reason="quedaban huecos"
    )
    state = {
        **initial_fields(CYBER_TICKET, close_requested=True),
        "spec": prior,
        "assessment": None,
    }
    out = await writer_turn(state, model)

    assert out["spec"].questions == []
    assert out["spec"].status.vagueness == 6
    # The document closes; the status stays honest about what was left open.
    assert "Cerrada a pedido tuyo" in out["spec"].status.reason
    assert "vaguedad 6" in out["spec"].status.reason


async def test_writer_drops_services_outside_the_catalog():
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "y",
            "services": ["cart-service", "servicio-inventado"],
            "criteria": [],
            "questions": [],
            "status": {"can_close": False, "vagueness": 3, "reason": "llm"},
        }
    )
    state = {**initial_fields(CYBER_TICKET), "questions": []}
    out = await writer_turn(state, model)

    assert out["spec"].services == ["cart-service"]


async def test_writer_accumulates_decisions_across_rounds():
    """A rule the PM overruled stays settled instead of resurfacing."""
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "y",
            "criteria": [],
            "questions": [],
            "status": {"can_close": False, "vagueness": 3, "reason": "llm"},
        }
    )
    prior = empty_spec(CYBER_TICKET)
    prior.decisions = [
        Decision(topic="reserva de stock", decision="se reserva al confirmar")
    ]
    state = {
        **initial_fields(CYBER_TICKET),
        "spec": prior,
        "decisions": [
            Decision(
                topic="adr-cart-price",
                decision="se deprecia para el flujo nuevo",
                impact="hay que reescribir la ADR",
            )
        ],
    }
    out = await writer_turn(state, model)

    topics = [d.topic for d in out["spec"].decisions]
    assert topics == ["reserva de stock", "adr-cart-price"]


async def test_writer_lets_a_new_decision_supersede_the_same_topic():
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "y",
            "criteria": [],
            "questions": [],
            "status": {"can_close": False, "vagueness": 3, "reason": "llm"},
        }
    )
    prior = empty_spec(CYBER_TICKET)
    prior.decisions = [Decision(topic="Alcance", decision="todos los SKU")]
    state = {
        **initial_fields(CYBER_TICKET),
        "spec": prior,
        "decisions": [Decision(topic="alcance", decision="solo SKU 1P de electro")],
    }
    out = await writer_turn(state, model)

    assert len(out["spec"].decisions) == 1
    assert out["spec"].decisions[0].decision == "solo SKU 1P de electro"


async def test_writer_flags_a_closed_spec():
    """A close request marks the document so the UI can hide the close controls."""
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "Cierre.",
            "criteria": [],
            "questions": ["ignorada"],
            "status": {"can_close": True, "vagueness": 0, "reason": "llm"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET, close_requested=True),
        "spec": empty_spec(CYBER_TICKET),
    }
    out = await writer_turn(state, model)

    assert out["spec"].closed is True


async def test_writer_does_not_flag_an_open_spec():
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "Sigue abierta.",
            "criteria": [],
            "questions": ["¿medimos?"],
            "status": {"can_close": False, "vagueness": 3, "reason": "llm"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET),
        "questions": ["¿medimos?"],
        "round_count": MAX_ROUNDS - 1,
    }
    out = await writer_turn(state, model)

    assert out["spec"].closed is False


async def test_writer_flags_a_spec_closed_by_round_budget():
    """Out of rounds means frozen, even without an explicit close."""
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "Se agotaron las rondas.",
            "criteria": [],
            "questions": [],
            "status": {"can_close": False, "vagueness": 4, "reason": "llm"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET),
        "questions": ["¿Y el stock?"],
        "round_count": MAX_ROUNDS,
    }
    out = await writer_turn(state, model)

    assert out["spec"].closed is True


async def test_writer_freezes_the_spec_when_the_rounds_run_out():
    """The round budget is a promise to the PM: it closes on its own."""
    model = _fake_llm_model(
        {
            "request": "x",
            "understanding": "y",
            "criteria": [],
            "questions": [],
            "status": {"can_close": True, "vagueness": 0, "reason": "invento"},
        }
    )
    state = {
        **initial_fields(CYBER_TICKET),
        "questions": ["¿Y el stock?"],
        "round_count": MAX_ROUNDS,
        "assessment": SpecStatus(can_close=False, vagueness=6, reason="x"),
    }
    out = await writer_turn(state, model)

    assert out["spec"].questions == []
    assert f"se agotaron las {MAX_ROUNDS} rondas" in out["spec"].status.reason


async def test_writer_keeps_only_declared_clashes():
    """A document typed `context` never lands in `clashes`."""
    model = _fake_llm_model(_draft())
    state = {
        **initial_fields(CYBER_TICKET),
        "citations": [_citation("adr-cart-price.md"), _citation("adr-stock-reserve.md")],
        "classifications": [
            DocumentAssessment(document_id="adr-cart-price.md", kind="clash"),
            DocumentAssessment(document_id="adr-stock-reserve.md", kind="context"),
        ],
    }
    out = await writer_turn(state, model)

    spec = out["spec"]
    assert [c.document_id for c in spec.clashes] == ["adr-cart-price.md"]
    assert [c.document_id for c in spec.context] == [
        "adr-cart-price.md",
        "adr-stock-reserve.md",
    ]


async def test_writer_accumulates_real_clashes_and_dedups():
    """Earlier clashes survive; a redeclared document is stored once."""
    model = _fake_llm_model(_draft())
    prior = empty_spec(CYBER_TICKET)
    prior.clashes = [_citation("adr-cart-price.md"), _citation("adr-shipping-step.md")]
    prior.context = [_citation("adr-cart-price.md"), _citation("adr-shipping-step.md")]
    state = {
        **initial_fields(CYBER_TICKET),
        "spec": prior,
        "citations": [_citation("adr-cart-price.md"), _citation("adr-stock-reserve.md")],
        "classifications": [
            DocumentAssessment(document_id="adr-cart-price.md", kind="clash"),
            DocumentAssessment(document_id="adr-stock-reserve.md", kind="clash"),
        ],
    }
    out = await writer_turn(state, model)

    ids = [c.document_id for c in out["spec"].clashes]
    assert ids == [
        "adr-cart-price.md",
        "adr-shipping-step.md",
        "adr-stock-reserve.md",
    ]
    assert len(ids) == len(set(ids))


async def test_writer_drops_declarations_outside_citations():
    """A hallucinated document_id can never fabricate a clash."""
    model = _fake_llm_model(_draft())
    state = {
        **initial_fields(CYBER_TICKET),
        "citations": [_citation("adr-cart-price.md")],
        "classifications": [
            DocumentAssessment(document_id="adr-cart-price.md", kind="context"),
            DocumentAssessment(document_id="inventada.md", kind="clash"),
        ],
    }
    out = await writer_turn(state, model)

    assert out["spec"].clashes == []
    assert [c.document_id for c in out["spec"].context] == ["adr-cart-price.md"]


async def test_writer_undeclared_citation_lands_in_context():
    """An undeclared citation is context, never a false clash (D1)."""
    model = _fake_llm_model(_draft())
    state = {
        **initial_fields(CYBER_TICKET),
        "citations": [_citation("adr-cart-price.md"), _citation("adr-stock-reserve.md")],
        "classifications": [
            DocumentAssessment(document_id="adr-cart-price.md", kind="clash"),
        ],
    }
    out = await writer_turn(state, model)

    assert [c.document_id for c in out["spec"].clashes] == ["adr-cart-price.md"]
    assert [c.document_id for c in out["spec"].context] == [
        "adr-cart-price.md",
        "adr-stock-reserve.md",
    ]


async def test_writer_carries_context_forward_when_nothing_retrieved():
    """A close turn retrieves nothing: prior context and clashes survive."""
    model = _fake_llm_model(_draft())
    prior = empty_spec(CYBER_TICKET)
    prior.clashes = [_citation("adr-cart-price.md")]
    prior.context = [_citation("adr-stock-reserve.md")]
    state = {
        **initial_fields(CYBER_TICKET, close_requested=True),
        "spec": prior,
        "citations": [],
        "classifications": [],
    }
    out = await writer_turn(state, model)

    assert [c.document_id for c in out["spec"].clashes] == ["adr-cart-price.md"]
    assert [c.document_id for c in out["spec"].context] == [
        "adr-stock-reserve.md"
    ]
