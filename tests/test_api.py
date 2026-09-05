"""Tests de la API FastAPI: empezar / seguir / cerrar hilos."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from checkpoint import create_checkpointer
from graph import build_graph
from schemas import Citation
from scoring import CYBER_TICKET


def _supervisor(state):
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


def _retriever(state):
    return {
        "citations": [
            Citation(
                document_id="adr-cart-price.md",
                title="El precio se cierra en el carrito",
                excerpt="No se saltea.",
            )
        ],
        "last_agent": "retriever",
    }


def _intake(state):
    from scoring import rank_questions

    qs = rank_questions(state["ticket"], state["citations"])
    return {"questions": qs, "last_agent": "intake"}


def _writer(state):
    spec = state["spec"]
    spec.preguntas = [] if state.get("close_requested") else list(state.get("questions") or [])
    spec.choques = list(state.get("citations") or [])
    spec.estado.razon = "dummy"
    spec.estado.se_puede_cerrar = bool(state.get("close_requested")) and spec.estado.vaguedad == 0
    return {"spec": spec, "last_agent": "writer"}


@pytest.fixture
def client(tmp_path):
    """App con grafo dummy del task 6 y checkpointer en tmpdir."""
    cp = create_checkpointer(tmp_path / "api-test.sqlite")
    graph = build_graph(
        supervisor=_supervisor,
        retriever=_retriever,
        intake=_intake,
        writer=_writer,
        checkpointer=cp,
    )
    from app import app, get_graph

    app.dependency_overrides[get_graph] = lambda: graph
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_start_continue_close_and_unknown_thread(client: TestClient):
    r = client.post("/threads", json={"ticket": CYBER_TICKET})
    assert r.status_code == 200
    thread_id = r.json()["thread_id"]
    spec = r.json()["spec"]
    assert spec["pedido"] == CYBER_TICKET
    assert len(spec["preguntas"]) == 3

    r2 = client.post(f"/threads/{thread_id}/messages", json={"content": "para todo, no medimos"})
    assert r2.status_code == 200

    r3 = client.post(f"/threads/{thread_id}/close")
    assert r3.status_code == 200
    assert r3.json()["spec"]["preguntas"] == []

    missing = client.post("/threads/does-not-exist/messages", json={"content": "hola"})
    assert missing.status_code == 404
    assert missing.json() == {"detail": "thread no existe"}

    missing_close = client.post("/threads/does-not-exist/close")
    assert missing_close.status_code == 404
    assert missing_close.json() == {"detail": "thread no existe"}
