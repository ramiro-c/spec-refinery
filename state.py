from __future__ import annotations

from typing import Literal

from langgraph.graph import MessagesState

from schemas import Citation, Decision, DocumentAssessment, SpecDocument, SpecStatus

NextAgent = Literal["retriever", "intake", "FINISH"]


class RefineryState(MessagesState):
    next_agent: NextAgent
    citations: list[Citation]
    questions: list[str]
    # Settled this round; the writer folds them into the spec, which is what
    # persists and what the interrogator reads back as "do not re-open".
    decisions: list[Decision]
    # The interrogator's per-document verdict for this turn; the writer
    # intersects it with the citations to derive real clashes.
    classifications: list[DocumentAssessment]
    # The interrogator's verdict for this thread; the writer copies it into the
    # spec instead of recomputing a score of its own.
    assessment: SpecStatus | None
    grilled: bool
    # Rounds of interrogation burned on this thread; persists across turns.
    round_count: int
    spec: SpecDocument | None
    close_requested: bool
    last_agent: str
    step_count: int
    last_error: str
    ticket: str


def empty_spec(ticket: str) -> SpecDocument:
    return SpecDocument(
        request=ticket,
        status=SpecStatus(can_close=False, vagueness=0, reason="inicio"),
    )


def initial_fields(ticket: str, *, close_requested: bool = False) -> dict:
    return {
        "next_agent": "FINISH",
        "citations": [],
        "questions": [],
        "decisions": [],
        "classifications": [],
        "assessment": None,
        "grilled": False,
        "round_count": 0,
        "spec": empty_spec(ticket),
        "close_requested": close_requested,
        "last_agent": "",
        "step_count": 0,
        "last_error": "",
        "ticket": ticket,
    }
