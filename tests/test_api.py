"""FastAPI tests: start / continue / close threads (dummy graph from agents.fakes)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from agents.fakes import fake_intake, fake_retriever, fake_supervisor, fake_writer
from checkpoint import close_checkpointer, create_checkpointer
from graph import build_graph
from demo import CYBER_TICKET


def _build_test_graph(tmp_path):
    """Build the graph with its checkpointer; AsyncSqliteSaver needs a running loop."""

    async def _build():
        cp = create_checkpointer(tmp_path / "api-test.sqlite")
        return build_graph(
            supervisor=fake_supervisor,
            retriever=fake_retriever,
            intake=fake_intake,
            writer=fake_writer,
            checkpointer=cp,
        )

    return asyncio.run(_build())


@pytest.fixture
def client(tmp_path):
    """App with the dummy graph and checkpointer in tmpdir."""
    graph = _build_test_graph(tmp_path)
    from app import app, get_graph

    app.dependency_overrides[get_graph] = lambda: graph
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    # The aiosqlite worker thread is non-daemon: close it or pytest never exits.
    asyncio.run(close_checkpointer(graph.checkpointer))


def test_start_continue_close_and_unknown_thread(client: TestClient):
    r = client.post("/threads", json={"ticket": CYBER_TICKET})
    assert r.status_code == 200
    thread_id = r.json()["thread_id"]
    spec = r.json()["spec"]
    assert spec["request"] == CYBER_TICKET
    assert len(spec["questions"]) == 3

    r2 = client.post(f"/threads/{thread_id}/messages", json={"content": "para todo, no medimos"})
    assert r2.status_code == 200

    r3 = client.post(f"/threads/{thread_id}/close")
    assert r3.status_code == 200
    assert r3.json()["spec"]["questions"] == []

    missing = client.post("/threads/does-not-exist/messages", json={"content": "hola"})
    assert missing.status_code == 404
    assert missing.json() == {"detail": "thread no existe"}

    missing_close = client.post("/threads/does-not-exist/close")
    assert missing_close.status_code == 404
    assert missing_close.json() == {"detail": "thread no existe"}


def test_a_plain_answer_keeps_the_thread_open(client: TestClient):
    """Only the interrogator (or /close) closes; an answer never does."""
    thread_id = client.post("/threads", json={"ticket": CYBER_TICKET}).json()["thread_id"]

    spec = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "todavía no cierres, falta definir el alcance"},
    ).json()["spec"]

    assert spec["questions"]


def test_close_reports_the_last_verdict(client: TestClient):
    """Closing freezes the document and keeps an honest readiness signal."""
    thread_id = client.post("/threads", json={"ticket": CYBER_TICKET}).json()["thread_id"]

    spec = client.post(f"/threads/{thread_id}/close").json()["spec"]

    assert spec["questions"] == []
    assert "vagueness" in spec["status"]
