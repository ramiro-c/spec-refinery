"""API FastAPI: empezar, seguir y cerrar hilos de refinado."""

from __future__ import annotations

from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from checkpoint import open_checkpointer
from graph import build_graph, invoke_config, run_turn
from schemas import SpecDocument

app = FastAPI()

_graph: CompiledStateGraph | None = None


def _build_default_graph() -> CompiledStateGraph:
    return build_graph(checkpointer=open_checkpointer())


def get_graph() -> CompiledStateGraph:
    """Dependencia del grafo compilado; overrideable en tests."""
    global _graph
    if _graph is None:
        _graph = _build_default_graph()
    return _graph


class StartThreadRequest(BaseModel):
    ticket: str


class MessageRequest(BaseModel):
    content: str


class StartThreadResponse(BaseModel):
    thread_id: str
    spec: SpecDocument


class SpecResponse(BaseModel):
    spec: SpecDocument


def _thread_config(thread_id: str) -> dict:
    return invoke_config(thread_id)


def _require_thread(graph: CompiledStateGraph, thread_id: str) -> str:
    """Valida que el hilo exista en el checkpointer y devuelve el ticket."""
    config = _thread_config(thread_id)
    checkpointer = graph.checkpointer
    if checkpointer is None or checkpointer.get_tuple(config) is None:
        raise HTTPException(status_code=404, detail="thread no existe")
    ticket = graph.get_state(config).values.get("ticket")
    if not ticket:
        raise HTTPException(status_code=404, detail="thread no existe")
    return ticket


@app.post("/threads", response_model=StartThreadResponse)
def start_thread(
    body: StartThreadRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> StartThreadResponse:
    """Empezar: nuevo hilo con el ticket inicial."""
    thread_id = str(uuid4())
    _, final = run_turn(
        graph,
        body.ticket,
        [HumanMessage(content=body.ticket)],
        thread_id=thread_id,
    )
    return StartThreadResponse(thread_id=thread_id, spec=final["spec"])


@app.post("/threads/{thread_id}/messages", response_model=SpecResponse)
def continue_thread(
    thread_id: str,
    body: MessageRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> SpecResponse:
    """Seguir: mensaje humano sobre un hilo existente."""
    ticket = _require_thread(graph, thread_id)
    _, final = run_turn(
        graph,
        ticket,
        [HumanMessage(content=body.content)],
        thread_id=thread_id,
    )
    return SpecResponse(spec=final["spec"])


@app.post("/threads/{thread_id}/close", response_model=SpecResponse)
def close_thread(
    thread_id: str,
    graph: CompiledStateGraph = Depends(get_graph),
) -> SpecResponse:
    """Cerrar: el humano cierra el hilo (sin listar conversaciones)."""
    ticket = _require_thread(graph, thread_id)
    _, final = run_turn(
        graph,
        ticket,
        [HumanMessage(content="cerrar")],
        thread_id=thread_id,
        close_requested=True,
    )
    return SpecResponse(spec=final["spec"])
