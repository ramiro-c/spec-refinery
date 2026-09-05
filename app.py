"""API FastAPI: empezar, seguir y cerrar hilos de refinado."""

from __future__ import annotations

from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agents.intake import make_intake_node
from agents.retriever_node import make_retriever_node
from agents.writer import make_writer_node
from checkpoint import open_checkpointer
from graph import build_graph, invoke_config, run_turn
from schemas import SpecDocument
from state import RefineryState

app = FastAPI()

_graph: CompiledStateGraph | None = None


def _rubric_supervisor(state: RefineryState) -> dict:
    """Supervisor determinístico por rúbrica (provisional hasta factory LLM en task 9)."""
    from agents.supervisor import apply_rubric

    step = int(state.get("step_count") or 0) + 1
    nxt = apply_rubric(
        citations_empty=not state.get("citations"),
        questions_empty=not state.get("questions"),
        step_count=step,
        proposed="FINISH",
        close_requested=bool(state.get("close_requested")),
        last_error=state.get("last_error") or "",
    )
    return {"next_agent": nxt, "step_count": step, "last_agent": "supervisor"}


def _build_default_graph() -> CompiledStateGraph:
    return build_graph(
        supervisor=_rubric_supervisor,
        retriever=make_retriever_node(),
        intake=make_intake_node(),
        writer=make_writer_node(None),
        checkpointer=open_checkpointer(),
    )


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
