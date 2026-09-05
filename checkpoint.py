"""SqliteSaver local con wrappers async para ``graph.stream`` / ``ainvoke``.

El checkpointer es un ``SqliteSaver`` sync (subclase del upstream) con métodos
``aget_tuple`` / ``aput`` mínimos para que LangGraph no falle en rutas async.
Mismo almacenamiento SQLite vía ``sqlite3``; sin ``AsyncSqliteSaver``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.sqlite import SqliteSaver as _BaseSqliteSaver

from config import CHECKPOINT_PATH


class SqliteSaver(_BaseSqliteSaver):
    """SqliteSaver sync con wrappers async para ``CompiledStateGraph.ainvoke``.

    Hereda del ``SqliteSaver`` upstream (checkpointing SQLite sync). LangGraph
    llama ``aget_tuple`` / ``aput`` / etc. desde ``ainvoke``; el upstream solo
    implementa la API sync. Estos métodos delegan en ``get_tuple`` / ``put`` /
    ``list`` / ``put_writes`` — mismo almacenamiento, sin ``AsyncSqliteSaver``
    ni ``aiosqlite``.

    Obtener instancias vía ``create_checkpointer()`` o ``open_checkpointer()``,
    no con el ``SqliteSaver`` crudo de ``langgraph.checkpoint.sqlite``.
    """

    async def aget_tuple(self, config: RunnableConfig):
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)


def create_checkpointer(path: str | sqlite3.Connection = ":memory:") -> SqliteSaver:
    """Crea el ``SqliteSaver`` local (sync + wrappers async) para ``build_graph``."""
    if isinstance(path, sqlite3.Connection):
        conn = path
    else:
        conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def open_checkpointer() -> SqliteSaver:
    """Abre el checkpointer persistente en ``config.CHECKPOINT_PATH``."""
    return create_checkpointer(str(CHECKPOINT_PATH))
