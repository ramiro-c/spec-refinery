"""Supervisor: deterministic policy router. It neither searches nor writes;
it only picks the next node from the state.

The route comes from ``apply_rubric``, a pure function over the state: a
``last_error`` or the step cap closes the turn, no citations means no FINISH,
and the interrogator grills the PM before the writer drafts. No model is called.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from config import MAX_ROUNDS, MAX_STEPS
from state import NextAgent, RefineryState


def apply_rubric(
    *,
    citations_empty: bool,
    grilled: bool,
    step_count: int,
    close_requested: bool,
    last_error: str = "",
    rounds_exhausted: bool = False,
) -> NextAgent:
    """Hard rules over the state. The graph never sees an illegal next_agent.

    Each node runs at most once per turn, so once the retriever and the
    interrogator have produced output the turn always ends at the writer.
    """
    if last_error.strip():
        return "FINISH"
    if step_count >= MAX_STEPS:
        return "FINISH"
    if close_requested or rounds_exhausted:
        return "FINISH"
    if citations_empty:
        return "retriever"
    if not grilled:
        return "intake"
    # Retriever and interrogator already ran this turn. Re-running them
    # re-searches the same query and re-grills the PM on the same transcript,
    # which is how the graph used to loop intake until it ran out of questions.
    return "FINISH"


def _rationale(
    *,
    next_agent: NextAgent,
    last_error: str,
    step_count: int,
    rounds_exhausted: bool,
    close_requested: bool,
) -> str:
    """One sentence naming why the router picked ``next_agent``.

    The FINISH reasons follow ``apply_rubric``'s branch order, so the sentence
    names the first hard rule that fired, or the natural end of the turn.
    """
    if next_agent == "retriever":
        return "No citations for this turn yet: retrieving the company rules first."
    if next_agent == "intake":
        return "The rules are in but the PM was not grilled yet: asking the next questions."
    if last_error:
        return f"There is a last_error: I do not retry the same node. {last_error}"
    if step_count >= MAX_STEPS:
        return f"Hit the {MAX_STEPS}-step cap: closing to avoid looping."
    if rounds_exhausted:
        return f"The thread burned its {MAX_ROUNDS} rounds: freezing the spec."
    if close_requested:
        return "The PM asked to close: writing the final spec."
    return "The rules are in and the PM was grilled this turn: writing the spec."


async def supervisor_turn(state: RefineryState) -> dict:
    step_count = int(state.get("step_count") or 0) + 1
    last_error = (state.get("last_error") or "").strip()
    citations = state.get("citations") or []
    grilled = bool(state.get("grilled"))
    close_requested = bool(state.get("close_requested"))
    rounds_exhausted = int(state.get("round_count") or 0) >= MAX_ROUNDS

    next_agent = apply_rubric(
        citations_empty=not citations,
        grilled=grilled,
        step_count=step_count,
        close_requested=close_requested,
        last_error=last_error,
        rounds_exhausted=rounds_exhausted,
    )
    rationale = _rationale(
        next_agent=next_agent,
        last_error=last_error,
        step_count=step_count,
        rounds_exhausted=rounds_exhausted,
        close_requested=close_requested,
    )

    return {
        "messages": [AIMessage(content=rationale, name="supervisor")],
        "next_agent": next_agent,
        "step_count": step_count,
        "last_agent": "supervisor",
    }


def make_supervisor_node():
    async def supervisor_node(state: RefineryState) -> dict:
        return await supervisor_turn(state)

    return supervisor_node
