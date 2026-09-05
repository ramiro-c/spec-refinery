from __future__ import annotations

from typing import Literal

from langgraph.graph import MessagesState

from schemas import Citation, SpecDocument, SpecStatus

NextAgent = Literal["retriever", "intake", "FINISH"]


class RefineryState(MessagesState):
    next_agent: NextAgent
    citations: list[Citation]
    questions: list[str]
    spec: SpecDocument | None
    close_requested: bool
    last_agent: str
    step_count: int
    last_error: str
    ticket: str


def empty_spec(ticket: str) -> SpecDocument:
    return SpecDocument(
        pedido=ticket,
        estado=SpecStatus(se_puede_cerrar=False, vaguedad=0, razon="inicio"),
    )


def initial_fields(ticket: str, *, close_requested: bool = False) -> dict:
    return {
        "next_agent": "FINISH",
        "citations": [],
        "questions": [],
        "spec": empty_spec(ticket),
        "close_requested": close_requested,
        "last_agent": "",
        "step_count": 0,
        "last_error": "",
        "ticket": ticket,
    }
