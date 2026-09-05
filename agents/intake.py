"""Nodo intake: ranking de preguntas y fanout de servicios impactados."""

from __future__ import annotations

from scoring import fanout, rank_questions
from state import RefineryState


def _human_texts(messages: list) -> str:
    """Texto acumulado de mensajes humanos para scoring multi-turno."""
    parts: list[str] = []
    for message in messages or []:
        if getattr(message, "type", None) != "human":
            continue
        content = getattr(message, "content", message)
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append(str(content))
    return " ".join(parts)


def intake_turn(state: RefineryState) -> dict:
    ticket = state.get("ticket") or ""
    citations = state.get("citations") or []
    excerpts = " ".join(c.excerpt for c in citations)
    context = ticket + " " + _human_texts(state.get("messages") or [])
    questions = rank_questions(context, citations)
    servicios = fanout(context + excerpts)
    spec = state.get("spec")
    if spec is not None:
        spec.servicios = servicios
    return {"questions": questions, "spec": spec, "last_agent": "intake"}


def make_intake_node():
    def intake_node(state: RefineryState) -> dict:
        return intake_turn(state)

    return intake_node
