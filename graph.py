"""Grafo jerárquico: supervisor rutea retriever → intake → writer.

``route_from_supervisor`` devuelve nombres de nodo. ``FINISH`` se mapea a
``writer``, que actualiza la spec y termina en ``END``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.retry import NODE_RETRY, node_error_handler
from config import RECURSION_LIMIT
from state import RefineryState, initial_fields

NodeFn = Callable[[RefineryState], dict]
Route = Literal["retriever", "intake", "writer"]


def route_from_supervisor(state: RefineryState) -> Route:
    """Lee ``next_agent`` y lo traduce a un destino del grafo."""
    nxt = state.get("next_agent")
    if nxt in ("retriever", "intake"):
        return nxt
    return "writer"


def build_graph(
    *,
    supervisor: NodeFn,
    retriever: NodeFn,
    intake: NodeFn,
    writer: NodeFn,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    builder = StateGraph(RefineryState)
    builder.set_node_defaults(
        retry_policy=NODE_RETRY,
        error_handler=node_error_handler,
    )
    builder.add_node("supervisor", supervisor)
    builder.add_node("retriever", retriever)
    builder.add_node("intake", intake)
    builder.add_node("writer", writer, retry_policy=None, error_handler=None)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "retriever": "retriever",
            "intake": "intake",
            "writer": "writer",
        },
    )
    builder.add_edge("retriever", "supervisor")
    builder.add_edge("intake", "supervisor")
    builder.add_edge("writer", END)
    return builder.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str | None = None) -> dict:
    config: dict = {"recursion_limit": RECURSION_LIMIT}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return config


def run_turn(
    graph: CompiledStateGraph,
    ticket: str,
    messages: list,
    *,
    thread_id: str | None = None,
    close_requested: bool = False,
) -> tuple[list[str], dict]:
    """Corre un turno en stream: hops de nodos + estado final."""
    hops: list[str] = []
    final: dict | None = None
    for mode, data in graph.stream(
        {**initial_fields(ticket, close_requested=close_requested), "messages": messages},
        invoke_config(thread_id),
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            for node in data:
                if node.startswith("__"):
                    continue
                hops.append(node)
        else:
            final = data
    if final is None:
        raise RuntimeError("el grafo no emitió estado final")
    return hops, final
