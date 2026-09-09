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
