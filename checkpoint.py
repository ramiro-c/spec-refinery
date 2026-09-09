"""Async SQLite checkpointer on the upstream AsyncSqliteSaver (aiosqlite).

Persistence uses ``langgraph-checkpoint-sqlite``'s ``AsyncSqliteSaver`` over an
``aiosqlite`` connection, matching the async graph runtime (``astream`` /
``ainvoke``). The saver is created lazily: ``aiosqlite.connect()`` returns an
un-awaited ``Connection`` whose background thread starts on first use, so
``create_checkpointer()`` stays a sync factory usable from tests and app
startup. State survives restarts via ``config.CHECKPOINT_PATH``.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import CHECKPOINT_PATH


def create_checkpointer(path: str | Path = ":memory:") -> AsyncSqliteSaver:
    """Create the async SQLite checkpointer handed to ``build_graph``."""
    return AsyncSqliteSaver(aiosqlite.connect(str(path)))


def open_checkpointer() -> AsyncSqliteSaver:
    """Open the persistent checkpointer at ``config.CHECKPOINT_PATH``."""
    return create_checkpointer(CHECKPOINT_PATH)


async def close_checkpointer(saver: AsyncSqliteSaver) -> None:
    """Close the underlying aiosqlite connection (stops its worker thread).

    The worker thread is non-daemon: without an explicit close it keeps the
    process alive after the event loop is gone.
    """
    await saver.conn.close()
