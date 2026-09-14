"""Writer node: rewrites the SpecDocument at the end of every turn."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.text import transcript_lines
from catalog import SERVICE_IDS
from config import MAX_ROUNDS
from schemas import Citation, Decision, SpecDocument, SpecStatus
from state import RefineryState, empty_spec

WRITER_PROMPT = """You are the spec writer of the refinery.
Rewrite understanding, criteria and services from the ticket, the
conversation, the previous spec and the citations. Do not invent rules that
are not in the citations, and pick services only from the catalog you are
given. Reflect the decisions the PM already took, including rules they chose
to change. Request, clashes, decisions, questions and status are fixed by the
system.
"""

_UNSCORED = SpecStatus(
    can_close=False,
    vagueness=10,
    reason="Todavía no evalué el pedido.",
)


def _merge_decisions(state: RefineryState) -> list[Decision]:
    """Decisions accumulate over the thread; the last word on a topic wins."""
    prior = state.get("spec")
    merged: list[Decision] = list(prior.decisions) if prior is not None else []
    for fresh in state.get("decisions") or []:
        key = fresh.topic.strip().lower()
        for index, existing in enumerate(merged):
            if existing.topic.strip().lower() == key:
                merged[index] = fresh
                break
        else:
            merged.append(fresh)
    return merged


def _merge_clashes(prior: list[Citation], fresh: list[Citation]) -> list[Citation]:
    """Real clashes accumulate over the thread; a redeclared id wins in place."""
    merged: list[Citation] = list(prior)
    for citation in fresh:
        for index, existing in enumerate(merged):
            if existing.document_id == citation.document_id:
                merged[index] = citation
                break
        else:
            merged.append(citation)
    return merged


def resolve_clashes_and_context(
    state: RefineryState,
) -> tuple[list[Citation], list[Citation]]:
    """Derive real clashes and retrieved context from declared classifications.

    We iterate THIS turn's citations, not the declarations, so an unknown or
    hallucinated ``document_id`` can never fabricate a clash. A citation with no
    declaration is context, never a silent clash.
    """
    citations = list(state.get("citations") or [])
    declared = {
        assessment.document_id: assessment.kind
        for assessment in (state.get("classifications") or [])
    }
    prior = state.get("spec")
    real_now = [c for c in citations if declared.get(c.document_id) == "clash"]
    clashes = _merge_clashes(list(prior.clashes) if prior is not None else [], real_now)
    if citations:
        context = citations
    else:
        context = list(prior.context) if prior is not None else []
    return clashes, context


def _is_frozen(state: RefineryState) -> bool:
    """Closed on request, or out of rounds."""
    return bool(state.get("close_requested")) or int(
        state.get("round_count") or 0
    ) >= MAX_ROUNDS


def _status(state: RefineryState, open_questions: int) -> SpecStatus:
    """The interrogator owns the verdict; the writer only carries it over.

    On an explicit close the interrogator does not run, so the last verdict
    stored on the thread is the honest one.
    """
    assessment = state.get("assessment")
    if assessment is None:
        prior = state.get("spec")
        if prior is not None and prior.status is not None:
            assessment = prior.status
    status = (assessment or _UNSCORED).model_copy()
    if not _is_frozen(state):
        return status
    # Closing is the human's call, and the round budget is a promise made to
    # them up front. Either way the spec closes; the status stays honest about
    # what was left open instead of refusing.
    if status.can_close:
        status.reason = "Cerrada sin puntos abiertos."
        return status
    # The question count is zero on the last round (it asks nothing), so lead
    # with vagueness there instead of claiming nothing was left open.
    pending = (
        f"{open_questions} pregunta(s) sin responder · vaguedad {status.vagueness}"
        if open_questions
        else f"vaguedad {status.vagueness}"
    )
    if state.get("close_requested"):
        status.reason = f"Cerrada a pedido tuyo, con {pending}."
    else:
        status.reason = (
            f"Cerrada: se agotaron las {MAX_ROUNDS} rondas, con {pending}."
        )
    return status


async def writer_turn(state: RefineryState, llm: BaseChatModel) -> dict:
    ticket = state.get("ticket") or ""
    close_requested = bool(state.get("close_requested"))
    citations = state.get("citations") or []
    questions = state.get("questions") or []
    spec = state.get("spec") or empty_spec(ticket)

    prior = state.get("spec")
    prior_text = prior.model_dump_json(indent=2) if prior is not None else "(none)"
    lines = transcript_lines(state.get("messages") or [])
    conversation = "\n".join(lines) if lines else "(none)"
    draft = await llm.with_structured_output(SpecDocument).ainvoke(
        [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(
                content=(
                    f"Ticket: {ticket}\n"
                    f"Conversation:\n{conversation}\n"
                    f"Previous spec:\n{prior_text}\n"
                    f"Citations: {[c.document_id for c in citations]}\n"
                    f"Decisions already taken: "
                    f"{[f'{d.topic}: {d.decision}' for d in _merge_decisions(state)]}\n"
                    f"Open questions: {questions}\n"
                    f"Service catalog: {', '.join(SERVICE_IDS)}"
                )
            ),
        ]
    )
    spec = draft
    spec.request = ticket
    spec.clashes, spec.context = resolve_clashes_and_context(state)
    spec.questions = [] if _is_frozen(state) else list(questions)
    spec.decisions = _merge_decisions(state)
    spec.services = [s for s in spec.services if s in SERVICE_IDS]
    spec.status = _status(state, len(questions))
    spec.closed = _is_frozen(state)

    return {
        "spec": spec,
        "messages": [AIMessage(content="Spec updated", name="writer")],
        "last_agent": "writer",
    }


def make_writer_node(llm: BaseChatModel):
    async def writer_node(state: RefineryState) -> dict:
        return await writer_turn(state, llm)

    return writer_node
