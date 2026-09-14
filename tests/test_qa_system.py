"""System-evidence test runner (qa_system.py) against the fake graph.

Drives the app in-process through ``httpx.ASGITransport``. Entering the
router's lifespan context keeps app startup/shutdown running. The checkpointer
lives in a tmp dir and is explicitly closed: the aiosqlite worker thread is
non-daemon, so leaving it open keeps pytest alive.
"""

from __future__ import annotations

import httpx
import pytest

from agents.fakes import fake_intake, fake_retriever, fake_supervisor, fake_writer
from checkpoint import close_checkpointer, create_checkpointer
from graph import build_graph

import qa_system
from qa_system import PASS, TraceReport, compute_exit_code, render_summary, run_scenarios, sc5_malformed_payload


async def _build_test_graph(tmp_path):
    cp = create_checkpointer(tmp_path / "qa-system-test.sqlite")
    return build_graph(
        supervisor=fake_supervisor,
        retriever=fake_retriever,
        intake=fake_intake,
        writer=fake_writer,
        checkpointer=cp,
    )


@pytest.fixture
async def asgi_client(tmp_path):
    """In-process client (AsyncClient over the ASGI transport)."""
    graph = await _build_test_graph(tmp_path)
    from app import app, get_graph

    app.dependency_overrides[get_graph] = lambda: graph
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client
    app.dependency_overrides.clear()
    # Non-daemon aiosqlite worker: close it or pytest never exits.
    await close_checkpointer(graph.checkpointer)


async def test_qa_system_scenarios_all_pass_on_fake_graph(asgi_client: httpx.AsyncClient):
    results, windows = await run_scenarios(asgi_client)
    by_id = {r.id: r for r in results}
    assert set(by_id) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    for sid in ("S1", "S2", "S3", "S4", "S5", "S6"):
        assert by_id[sid].status == PASS, f"{sid} failed: {by_id[sid].detail}"
    # Distinct threads for the two retrieval scenarios.
    assert by_id["S2"].detail != by_id["S3"].detail
    # One recorded time window per scenario (trace-correlation contract).
    assert [sid for sid, _t0, _t1 in windows] == ["S1", "S2", "S3", "S4", "S5", "S6"]

    # The summary/exit-code path reports all mandatory scenarios passing.
    trace = TraceReport(
        qa_system.PASS, "mocked: Phoenix returned spans for every scenario window"
    )
    summary = render_summary(results, trace)
    assert "5/5 scenarios passed" in summary
    assert "[PASS]" in summary
    assert compute_exit_code(results, trace) == 0


async def test_sc5_malformed_payload_unit(asgi_client: httpx.AsyncClient):
    detail = await sc5_malformed_payload(asgi_client, ctx={})
    assert "422" in detail
    assert "ticket" in detail


async def test_sc5_malformed_payload_with_asgi_transport(tmp_path):
    """Same scenario exercised directly through the in-process ASGI transport."""
    graph = await _build_test_graph(tmp_path)
    from app import app, get_graph

    app.dependency_overrides[get_graph] = lambda: graph
    transport = httpx.ASGITransport(app=app)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                detail = await sc5_malformed_payload(client, ctx={})
        assert "422" in detail
    finally:
        app.dependency_overrides.clear()
        await close_checkpointer(graph.checkpointer)


def test_verify_traces_langsmith_active_is_not_applicable(monkeypatch):
    """When LangSmith tracing is on, the Phoenix check is N/A (no network)."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    report = qa_system.verify_phoenix_traces(
        [("S1", qa_system._utcnow(), qa_system._utcnow())],
        phoenix_url="http://localhost:6010",
    )
    assert report.status == qa_system.NA
    assert "LangSmith" in report.detail


def test_verify_traces_unreachable_phoenix_is_skipped(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    report = qa_system.verify_phoenix_traces(
        [("S1", qa_system._utcnow(), qa_system._utcnow())],
        phoenix_url="http://127.0.0.1:1",  # nothing listens here
        wait_seconds=1.0,
    )
    assert report.status == qa_system.SKIP
    assert "not reachable" in report.detail


def test_compute_exit_code_blocks_on_trace_fail():
    from qa_system import ScenarioResult

    results = [ScenarioResult("S1", "x", PASS), ScenarioResult("S2", "y", PASS)]
    assert compute_exit_code(results, TraceReport(qa_system.FAIL, "missing spans")) == 1
    assert compute_exit_code(results, TraceReport(qa_system.SKIP, "unreachable")) == 0
    results.append(ScenarioResult("S1", "x", qa_system.FAIL))
    assert compute_exit_code(results, TraceReport(qa_system.PASS, "ok")) == 1
