"""Shared helpers to extract plain text from LangChain messages."""

from __future__ import annotations

from typing import Any


def message_text(message: Any) -> str:
    """Flatten a message content (str or content blocks) into plain text."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def human_texts(messages: list) -> list[str]:
    """All human messages accumulated in the thread."""
    return [
        message_text(message)
        for message in messages or []
        if getattr(message, "type", None) == "human"
    ]


def transcript_lines(messages: list) -> list[str]:
    """The real conversation: what the PM said and what the refinery asked.

    Supervisor rationales and writer bookkeeping are graph noise, not dialogue.
    """
    lines: list[str] = []
    for message in messages or []:
        kind = getattr(message, "type", None)
        text = message_text(message).strip()
        if not text:
            continue
        if kind == "human":
            lines.append(f"PM: {text}")
        elif getattr(message, "name", None) == "intake":
            lines.append(f"Refinery: {text}")
    return lines
