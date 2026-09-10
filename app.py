"""FastAPI API: start, continue and close refinement threads.

All endpoints are async; the graph runs on ``astream`` over the async SQLite
checkpointer. The graph (and its checkpointer) is built lazily inside the
request event loop — ``AsyncSqliteSaver`` requires a running loop at
construction — and cached on ``app.state``. The request/response contract is
unchanged.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from checkpoint import close_checkpointer, open_checkpointer
from config import GRAPH_MODE, MAX_QUESTIONS_PER_ROUND, MAX_ROUNDS
from graph import build_graph, invoke_config, run_turn
from schemas import SpecDocument
from tracing import setup_tracing

setup_tracing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The graph itself is built lazily by get_graph (overrideable in tests);
    # shutdown only closes the checkpointer's aiosqlite worker.
    yield
    graph = getattr(app.state, "graph", None)
    checkpointer = getattr(graph, "checkpointer", None) if graph else None
    if checkpointer is not None:
        await close_checkpointer(checkpointer)


app = FastAPI(title="Spec Refinery", lifespan=lifespan)


def _build_default_graph() -> CompiledStateGraph:
    checkpointer = open_checkpointer()
    if GRAPH_MODE == "fake":
        from agents.fakes import fake_intake, fake_retriever, fake_supervisor, fake_writer

        return build_graph(
            supervisor=fake_supervisor,
            retriever=fake_retriever,
            intake=fake_intake,
            writer=fake_writer,
            checkpointer=checkpointer,
        )
    return build_graph(checkpointer=checkpointer)


async def get_graph(request: Request) -> CompiledStateGraph:
    """Compiled graph dependency; overridable in tests."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        request.app.state.graph = _build_default_graph()
    return request.app.state.graph


class StartThreadRequest(BaseModel):
    ticket: str


class MessageRequest(BaseModel):
    content: str


class StartThreadResponse(BaseModel):
    thread_id: str
    spec: SpecDocument
    ronda: int
    max_rondas: int = MAX_ROUNDS
    max_preguntas_por_ronda: int = MAX_QUESTIONS_PER_ROUND


class SpecResponse(BaseModel):
    spec: SpecDocument
    ronda: int
    max_rondas: int = MAX_ROUNDS
    max_preguntas_por_ronda: int = MAX_QUESTIONS_PER_ROUND


def _thread_config(thread_id: str) -> dict:
    return invoke_config(thread_id)


async def _run_turn_or_503(*args, **kwargs) -> tuple[list[str], dict]:
    """Runs a turn; LLM/runtime failures surface as a clean 503, not a raw 500.

    A node failure (e.g. provider rate limit) leaves the raw exception in the
    graph state, which LangGraph then fails to msgpack-serialize — without this
    guard the client gets an opaque ``TypeError`` 500.
    """
    try:
        return await run_turn(*args, **kwargs)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"refinement failed: {type(exc).__name__}: {exc}",
        ) from exc


async def _require_thread(graph: CompiledStateGraph, thread_id: str) -> str:
    """Validates that the thread exists in the checkpointer and returns the ticket."""
    config = _thread_config(thread_id)
    checkpointer = graph.checkpointer
    if checkpointer is None or await checkpointer.aget_tuple(config) is None:
        raise HTTPException(status_code=404, detail="thread no existe")
    state = await graph.aget_state(config)
    ticket = state.values.get("ticket")
    if not ticket:
        raise HTTPException(status_code=404, detail="thread no existe")
    return ticket


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "graph": GRAPH_MODE}


@app.post("/threads", response_model=StartThreadResponse)
async def start_thread(
    body: StartThreadRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> StartThreadResponse:
    """Start: new thread with the initial ticket."""
    thread_id = str(uuid4())
    _, final = await _run_turn_or_503(
        graph,
        body.ticket,
        [HumanMessage(content=body.ticket)],
        thread_id=thread_id,
    )
    return StartThreadResponse(
        thread_id=thread_id,
        spec=final["spec"],
        ronda=int(final.get("round_count") or 0),
    )


@app.post("/threads/{thread_id}/messages", response_model=SpecResponse)
async def continue_thread(
    thread_id: str,
    body: MessageRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> SpecResponse:
    """Continue: human message on an existing thread.

    The human can also close from the chat ("cerrá la spec"): the interrogator
    reads the intent and raises ``close_requested`` mid-turn. The explicit
    /close endpoint stays available for the UI button.
    """
    ticket = await _require_thread(graph, thread_id)
    _, final = await _run_turn_or_503(
        graph,
        ticket,
        [HumanMessage(content=body.content)],
        thread_id=thread_id,
    )
    return SpecResponse(
        spec=final["spec"], ronda=int(final.get("round_count") or 0)
    )


@app.post("/threads/{thread_id}/close", response_model=SpecResponse)
async def close_thread(
    thread_id: str,
    graph: CompiledStateGraph = Depends(get_graph),
) -> SpecResponse:
    """Close: the human closes the thread (no conversation listing)."""
    ticket = await _require_thread(graph, thread_id)
    _, final = await _run_turn_or_503(
        graph,
        ticket,
        [HumanMessage(content="cerrar")],
        thread_id=thread_id,
        close_requested=True,
    )
    return SpecResponse(
        spec=final["spec"], ronda=int(final.get("round_count") or 0)
    )
