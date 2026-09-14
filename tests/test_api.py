"""FastAPI tests: start / continue / close threads (dummy graph from agents.fakes).

The app is driven in-process through ``httpx.ASGITransport`` instead of
``fastapi.testclient``: importing starlette's TestClient module trips an
upstream anyio ``BlockingPortal`` deprecation warning. Entering the router's
lifespan context keeps app startup/shutdown running.
"""

from __future__ import annotations

import httpx
import pytest

from agents.fakes import fake_intake, fake_retriever, fake_supervisor, fake_writer
from checkpoint import close_checkpointer, create_checkpointer
from graph import build_graph
from demo import CYBER_TICKET


async def _build_test_graph(tmp_path):
    """Build the graph with its checkpointer (AsyncSqliteSaver needs a running loop)."""
    cp = create_checkpointer(tmp_path / "api-test.sqlite")
    return build_graph(
        supervisor=fake_supervisor,
        retriever=fake_retriever,
        intake=fake_intake,
        writer=fake_writer,
        checkpointer=cp,
    )


@pytest.fixture
async def client(tmp_path):
    """App with the dummy graph and checkpointer in tmpdir."""
    graph = await _build_test_graph(tmp_path)
    from app import app, get_graph

    app.dependency_overrides[get_graph] = lambda: graph
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as test_client:
            yield test_client
    app.dependency_overrides.clear()
    # The aiosqlite worker thread is non-daemon: close it or pytest never exits.
    await close_checkpointer(graph.checkpointer)


async def test_start_continue_close_and_unknown_thread(client: httpx.AsyncClient):
    r = await client.post("/threads", json={"ticket": CYBER_TICKET})
    assert r.status_code == 200
    thread_id = r.json()["thread_id"]
    spec = r.json()["spec"]
    assert spec["request"] == CYBER_TICKET
    assert len(spec["questions"]) == 3

    r2 = await client.post(f"/threads/{thread_id}/messages", json={"content": "para todo, no medimos"})
    assert r2.status_code == 200

    r3 = await client.post(f"/threads/{thread_id}/close")
    assert r3.status_code == 200
    assert r3.json()["spec"]["questions"] == []

    missing = await client.post("/threads/does-not-exist/messages", json={"content": "hola"})
    assert missing.status_code == 404
    assert missing.json() == {"detail": "thread no existe"}

    missing_close = await client.post("/threads/does-not-exist/close")
    assert missing_close.status_code == 404
    assert missing_close.json() == {"detail": "thread no existe"}


async def test_a_plain_answer_keeps_the_thread_open(client: httpx.AsyncClient):
    """Only the interrogator (or /close) closes; an answer never does."""
    thread_id = (await client.post("/threads", json={"ticket": CYBER_TICKET})).json()["thread_id"]

    spec = (
        await client.post(
            f"/threads/{thread_id}/messages",
            json={"content": "todavía no cierres, falta definir el alcance"},
        )
    ).json()["spec"]

    assert spec["questions"]


async def test_close_reports_the_last_verdict(client: httpx.AsyncClient):
    """Closing freezes the document and keeps an honest readiness signal."""
    thread_id = (await client.post("/threads", json={"ticket": CYBER_TICKET})).json()["thread_id"]

    spec = (await client.post(f"/threads/{thread_id}/close")).json()["spec"]

    assert spec["questions"] == []
    assert "vagueness" in spec["status"]
