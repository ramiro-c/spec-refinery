"""Retriever node: searches the corpus for citations for the current ticket."""

from __future__ import annotations

from retriever import aretrieve
from state import RefineryState


def _last_human_text(messages: list) -> str:
    """Last human message of the turn (query or clarification)."""
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


async def retriever_turn(state: RefineryState) -> dict:
    ticket = state.get("ticket") or ""
    last_human = _last_human_text(state.get("messages") or [])
    citations = await aretrieve(ticket + last_human)
    return {"citations": citations, "last_agent": "retriever"}


def make_retriever_node():
    async def retriever_node(state: RefineryState) -> dict:
        return await retriever_turn(state)

    return retriever_node
