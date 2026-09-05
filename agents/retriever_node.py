"""Nodo retriever: busca citas en el corpus para el ticket del turno."""

from __future__ import annotations

from retriever import retrieve
from state import RefineryState


def _last_human_text(messages: list) -> str:
    """Último mensaje humano del turno (consulta o aclaración)."""
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def retriever_turn(state: RefineryState) -> dict:
    ticket = state.get("ticket") or ""
    last_human = _last_human_text(state.get("messages") or [])
    citations = retrieve(ticket + last_human)
    return {"citations": citations, "last_agent": "retriever"}


def make_retriever_node():
    def retriever_node(state: RefineryState) -> dict:
        return retriever_turn(state)

    return retriever_node
