"""Supervisor: router + rubric. It neither searches nor writes; it only picks
the next node.

The LLM decision is a ``Literal`` (node names). ``apply_rubric`` corrects it if
it violates the hard rules: no citations means no FINISH, and ``MAX_STEPS``
cuts the loop.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.text import message_text
from config import MAX_ROUNDS, MAX_STEPS
from state import NextAgent, RefineryState

SUPERVISOR_PROMPT = """You are the supervisor of a spec refinery.
Do not search the corpus nor draft the spec. Pick the next agent.

Rubric (in order, the first match wins):
1. If the human asked to close, or the thread burned its round budget -> FINISH.
2. If citations is empty -> retriever.
3. If the interrogator has not grilled the PM this turn -> intake.
4. Otherwise -> FINISH; each node runs at most once per turn, so never send
   the turn back to retriever or intake.
5. Never pick an agent that does not exist. Options: retriever, intake, FINISH.

Answer only with next_agent and a short rationale (one sentence).
"""


class SupervisorDecision(BaseModel):
    next_agent: NextAgent = Field(
        description="Next node: retriever, intake or FINISH."
    )
    rationale: str = Field(description="Why that choice, one sentence.")


def apply_rubric(
    *,
    citations_empty: bool,
    grilled: bool,
    step_count: int,
    proposed: NextAgent,
    close_requested: bool,
    last_error: str = "",
    rounds_exhausted: bool = False,
) -> NextAgent:
    """Hard rules on top of the LLM. The graph never sees an illegal next_agent.

    ``proposed`` is the LLM's pick. It is honoured only while it does not
    break a hard rule; each node runs at most once per turn, so once the
    retriever and the interrogator have produced output the turn always ends
    at the writer. The divergence is reported in the supervisor's rationale.
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


def _user_query(messages: list) -> str:
    """Last human question: the most recent query, not graph noise."""
    last = ""
    for message in messages or []:
        if getattr(message, "type", None) == "human":
            last = message_text(message)
    return last


def _snapshot(state: RefineryState) -> str:
    citations = state.get("citations") or []
    questions = state.get("questions") or []
    citations_text = (
        "(empty)"
        if not citations
        else "\n".join(f"- {c.document_id}: {c.title or c.excerpt[:80]}" for c in citations)
    )
    questions_text = (
        "(empty)" if not questions else "\n".join(f"- {q}" for q in questions)
    )
    grilled = bool(state.get("grilled"))
    last_agent = state.get("last_agent") or "(nobody)"
    step_count = int(state.get("step_count") or 0)
    last_error = (state.get("last_error") or "").strip() or "(none)"
    close_requested = bool(state.get("close_requested"))
    ticket = state.get("ticket") or "(no ticket)"
    return (
        f"Ticket: {ticket}\n"
        f"Query: {_user_query(state.get('messages') or [])}\n"
        f"last_agent: {last_agent}\n"
        f"step_count: {step_count}/{MAX_STEPS}\n"
        f"last_error: {last_error}\n"
        f"close_requested: {close_requested}\n"
        f"interrogator already ran this turn: {grilled}\n\n"
        f"citations:\n{citations_text}\n\n"
        f"questions:\n{questions_text}"
    )


async def supervisor_turn(state: RefineryState, llm: BaseChatModel) -> dict:
    step_count = int(state.get("step_count") or 0) + 1
    last_error = (state.get("last_error") or "").strip()
    citations = state.get("citations") or []
    grilled = bool(state.get("grilled"))
    close_requested = bool(state.get("close_requested"))
    rounds_exhausted = int(state.get("round_count") or 0) >= MAX_ROUNDS

    if last_error:
        rationale = f"There is a last_error: I do not retry the same node. {last_error}"
        next_agent: NextAgent = "FINISH"
    elif step_count >= MAX_STEPS:
        rationale = f"Hit the {MAX_STEPS}-step cap: closing to avoid looping."
        next_agent = "FINISH"
    elif rounds_exhausted:
        rationale = f"The thread burned its {MAX_ROUNDS} rounds: freezing the spec."
        next_agent = "FINISH"
    else:
        decision = await llm.with_structured_output(SupervisorDecision).ainvoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=_snapshot({**state, "step_count": step_count})),
            ]
        )
        next_agent = apply_rubric(
            citations_empty=not citations,
            grilled=grilled,
            step_count=step_count,
            proposed=decision.next_agent,
            close_requested=close_requested,
            last_error=last_error,
            rounds_exhausted=rounds_exhausted,
        )
        rationale = decision.rationale
        if next_agent != decision.next_agent:
            rationale = f"{rationale} [rubric: {decision.next_agent} -> {next_agent}]"

    return {
        "messages": [AIMessage(content=rationale, name="supervisor")],
        "next_agent": next_agent,
        "step_count": step_count,
        "last_agent": "supervisor",
    }


def make_supervisor_node(llm: BaseChatModel):
    async def supervisor_node(state: RefineryState) -> dict:
        return await supervisor_turn(state, llm)

    return supervisor_node
