"""Hierarchical graph: supervisor routes retriever -> interrogator -> writer.

``route_from_supervisor`` returns node names. ``FINISH`` maps to ``writer``,
which updates the spec and ends at ``END``. ``run_turn`` streams asynchronously
(``astream``) and preserves the event contract consumed by the API: node hops
plus the final state.
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
    """Reads ``next_agent`` and maps it to a graph destination."""
    nxt = state.get("next_agent")
    if nxt in ("retriever", "intake"):
        return nxt
    return "writer"


def build_graph(
    *,
    supervisor: NodeFn | None = None,
    retriever: NodeFn | None = None,
    intake: NodeFn | None = None,
    writer: NodeFn | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Assemble the graph. Without explicit nodes, uses the LLM factory (supervisor + writer)."""
    if supervisor is None or writer is None or intake is None:
        from agents.intake import make_intake_node
        from agents.supervisor import make_supervisor_node
        from agents.writer import make_writer_node
        from clients.factory import build_role_models

        models = build_role_models()
        supervisor = supervisor or make_supervisor_node(models["supervisor"])
        intake = intake or make_intake_node(models["interrogator"])
        writer = writer or make_writer_node(models["writer"])
    if retriever is None:
        from agents.retriever_node import make_retriever_node

        retriever = make_retriever_node()
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


async def _has_checkpoint(graph: CompiledStateGraph, config: dict) -> bool:
    """True if the thread already has persisted state."""
    checkpointer = graph.checkpointer
    if checkpointer is None:
        return False
    return await checkpointer.aget_tuple(config) is not None


async def run_turn(
    graph: CompiledStateGraph,
    ticket: str,
    messages: list,
    *,
    thread_id: str | None = None,
    close_requested: bool = False,
) -> tuple[list[str], dict]:
    """Run a turn as an async stream: node hops + final state."""
    config = invoke_config(thread_id)
    if await _has_checkpoint(graph, config):
        # Continuation: do not overwrite the checkpointed spec or other fields.
        inputs = {
            "messages": messages,
            "ticket": ticket,
            "citations": [],
            "questions": [],
            "decisiones": [],
            # Per-turn flag; ``assessment`` deliberately survives the turn so an
            # explicit close still reports the last honest verdict.
            "grilled": False,
            "step_count": 0,
            "last_error": "",
        }
        # Closure is sticky: only an explicit close request may set it. On every
        # other turn the key is omitted so the checkpointed value survives;
        # resetting it to False here would reopen a thread closed by an earlier
        # /close. An open thread already persists False, so this is a no-op there.
        if close_requested:
            inputs["close_requested"] = True
    else:
        inputs = {**initial_fields(ticket, close_requested=close_requested), "messages": messages}

    hops: list[str] = []
    final: dict | None = None
    async for mode, data in graph.astream(
        inputs,
        config,
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
        raise RuntimeError("the graph did not emit a final state")
    return hops, final
