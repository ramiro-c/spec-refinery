"""Supervisor: deterministic policy router.

The route is a pure function of the state; the supervisor never calls a model.
"""

from __future__ import annotations

import inspect

import pytest
from langchain_core.messages import AIMessage

import agents.supervisor as supervisor
from agents.supervisor import apply_rubric, make_supervisor_node, supervisor_turn
from config import MAX_ROUNDS, MAX_STEPS
from schemas import Citation
from state import initial_fields

TICKET = "checkout más rápido tipo Amazon"


def _state(**overrides) -> dict:
    return {**initial_fields(TICKET), **overrides}


def _citation() -> Citation:
    return Citation(
        document_id="adr-cart-price.md",
        title="El precio se cierra en el carrito",
        excerpt="No se saltea.",
    )


# Representative states: the same inputs the rubric used to route on.
_ROUTING_CASES = [
    ({}, "retriever"),
    ({"citations": [_citation()]}, "intake"),
    ({"citations": [_citation()], "grilled": True}, "FINISH"),
    (
        {"citations": [_citation()], "grilled": True, "close_requested": True},
        "FINISH",
    ),
    ({"citations": [_citation()], "grilled": True, "last_error": "boom"}, "FINISH"),
    ({"citations": [_citation()], "grilled": True, "step_count": MAX_STEPS}, "FINISH"),
    ({"citations": [_citation()], "grilled": True, "round_count": MAX_ROUNDS}, "FINISH"),
]

# (state, substring that must appear in the deterministic rationale)
_RATIONALE_CASES = [
    ({}, "citations"),
    ({"citations": [_citation()]}, "grilled"),
    ({"citations": [_citation()], "grilled": True}, "writing"),
    (
        {"citations": [_citation()], "grilled": True, "close_requested": True},
        "close",
    ),
    ({"citations": [_citation()], "grilled": True, "last_error": "boom"}, "boom"),
    ({"citations": [_citation()], "grilled": True, "step_count": MAX_STEPS}, "cap"),
    ({"citations": [_citation()], "grilled": True, "round_count": MAX_ROUNDS}, "rounds"),
]


# --- apply_rubric: the routing policy itself ---


def test_rubric_blocks_finish_without_citations():
    assert apply_rubric(
        citations_empty=True,
        grilled=False,
        step_count=1,
        close_requested=False,
        last_error="",
    ) == "retriever"


def test_rubric_allows_finish_on_close_without_citations():
    assert apply_rubric(
        citations_empty=True,
        grilled=False,
        step_count=1,
        close_requested=True,
        last_error="",
    ) == "FINISH"


def test_rubric_grills_before_finishing():
    assert apply_rubric(
        citations_empty=False,
        grilled=False,
        step_count=2,
        close_requested=False,
        last_error="",
    ) == "intake"


def test_rubric_caps_steps():
    assert apply_rubric(
        citations_empty=False,
        grilled=True,
        step_count=8,
        close_requested=False,
        last_error="",
    ) == "FINISH"


def test_rubric_finishes_on_last_error():
    assert apply_rubric(
        citations_empty=False,
        grilled=False,
        step_count=1,
        close_requested=False,
        last_error="boom",
    ) == "FINISH"


def test_rubric_never_regrills_the_pm_in_the_same_turn():
    """Re-running the interrogator on an unchanged transcript loops the graph."""
    assert apply_rubric(
        citations_empty=False,
        grilled=True,
        step_count=3,
        close_requested=False,
        last_error="",
    ) == "FINISH"


def test_rubric_never_reruns_retriever_once_citations_exist():
    assert apply_rubric(
        citations_empty=False,
        grilled=True,
        step_count=3,
        close_requested=False,
        last_error="",
    ) == "FINISH"


def test_rubric_rounds_exhausted_finishes_even_without_citations():
    assert apply_rubric(
        citations_empty=True,
        grilled=False,
        step_count=1,
        close_requested=False,
        last_error="",
        rounds_exhausted=True,
    ) == "FINISH"


# --- supervisor_turn: deterministic contract ---


@pytest.mark.parametrize(("overrides", "expected"), _ROUTING_CASES)
async def test_supervisor_routes_the_same_states_as_the_rubric(overrides, expected):
    out = await supervisor_turn(_state(**overrides))

    assert out["next_agent"] == expected


@pytest.mark.parametrize(("overrides", "needle"), _RATIONALE_CASES)
async def test_supervisor_rationale_names_the_reason(overrides, needle):
    out = await supervisor_turn(_state(**overrides))
    rationale = out["messages"][0].content

    assert rationale
    assert needle in rationale


async def test_supervisor_update_shape_is_unchanged():
    out = await supervisor_turn(_state())

    assert set(out) == {"messages", "next_agent", "step_count", "last_agent"}
    message = out["messages"][0]
    assert isinstance(message, AIMessage)
    assert message.name == "supervisor"
    assert out["last_agent"] == "supervisor"
    assert out["step_count"] == 1
    assert out["next_agent"] == "retriever"


async def test_supervisor_counts_the_step_and_last_agent():
    out = await supervisor_turn(_state(step_count=2))

    assert out["step_count"] == 3
    assert out["last_agent"] == "supervisor"


# --- no model call ---


def test_supervisor_turn_takes_no_llm_parameter():
    assert "llm" not in inspect.signature(supervisor_turn).parameters
    assert list(inspect.signature(make_supervisor_node).parameters) == []


def test_supervisor_module_keeps_no_model_machinery():
    for name in (
        "BaseChatModel",
        "SystemMessage",
        "HumanMessage",
        "SupervisorDecision",
        "SUPERVISOR_PROMPT",
        "_snapshot",
        "_user_query",
    ):
        assert not hasattr(supervisor, name), f"{name} should be gone"


async def test_supervisor_turn_never_calls_a_model(monkeypatch):
    from langchain_core.language_models import BaseChatModel

    def _explode(*args, **kwargs):
        raise AssertionError("the supervisor must not call a model")

    monkeypatch.setattr(BaseChatModel, "ainvoke", _explode)
    monkeypatch.setattr(BaseChatModel, "with_structured_output", _explode)

    out = await supervisor_turn(_state())

    assert out["next_agent"] == "retriever"


async def test_make_supervisor_node_wraps_the_deterministic_turn():
    node = make_supervisor_node()

    out = await node(_state())

    assert out["last_agent"] == "supervisor"
