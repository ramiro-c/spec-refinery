"""Intake node: question ranking and impacted-services fanout."""

from __future__ import annotations

from agents.text import human_texts
from scoring import fanout, rank_questions
from state import RefineryState


def intake_turn(state: RefineryState) -> dict:
    ticket = state.get("ticket") or ""
    citations = state.get("citations") or []
    excerpts = " ".join(c.excerpt for c in citations)
    context = ticket + " " + " ".join(human_texts(state.get("messages") or []))
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
