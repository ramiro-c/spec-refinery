"""System-evidence tests for Spec Refinery (course final deliverable).

Runs 6 end-to-end scenarios against the live API (started separately via
``run.sh`` or ``docker compose up``) and verifies that every scenario produced
spans in Arize Phoenix within its execution window. This is the scripted,
repeatable evidence for the rubric item "at least 5 system tests with traces
visible in Arize Phoenix or LangSmith".

Usage:
    python qa_system.py                      # against http://localhost:8000
    python qa_system.py --selftest           # in-process (fake graph), no server
    python qa_system.py --base-url http://localhost:8000 --phoenix-url http://localhost:6010
    python qa_system.py --skip-traces        # scenarios only

Scenarios (S1-S5 mandatory, S6 bonus):
    S1  GET /health + POST /threads        -> 200, valid Pydantic response
    S2  Golden question #1 (new thread)    -> 200, spec.choques has citations
    S3  Golden question #2 (new thread)    -> 200, citations, different thread
    S4  Follow-up on the SAME thread       -> 200, checkpointer kept state
    S5  Malformed payload (no "ticket")    -> 422, Pydantic error detail
    S6  Close the thread (bonus)           -> 200, spec reflects closure

Exit code is 0 only if every mandatory scenario passes AND the trace check
passes (trace check SKIP/N/A does not block; it never fabricates a pass).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent
GOLDEN_PATH = REPO_ROOT / "golden_set.json"
PROJECT_NAME = "spec-refinery"

DEFAULT_BASE_URL = os.getenv("QA_BASE_URL", "http://localhost:8000")
DEFAULT_PHOENIX_URL = os.getenv("QA_PHOENIX_URL", "http://localhost:6010")

PASS, FAIL, SKIP, NA = "PASS", "FAIL", "SKIP", "N/A"
TRACE_WINDOW_EPSILON = dt.timedelta(seconds=2)  # clock-slack on window edges

# Follow-up message for the multi-turn scenario. Rich in rubric slots
# (measurable criterion, scope, out-of-scope, dependency) so the real graph
# can plausibly use it to refine the spec.
FOLLOW_UP = (
    "Aclaro: aplica a todos los SKUs 1P, medimos p95 < 2 segundos en comprar "
    "ahora; marketplace queda fuera de alcance; depende del equipo de pagos."
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _qa_ticket() -> str:
    """Cyber Monday ticket, same one used by scoring/demo."""
    try:
        from scoring import CYBER_TICKET

        return CYBER_TICKET
    except Exception:
        return (
            "Para el Cyber Monday queremos un checkout más rápido, tipo Amazon: "
            "que el comprar ahora no pase por el carrito."
        )


# --------------------------------------------------------------------------
# Scenarios. Each takes (client, ctx), raises AssertionError on failure and
# returns a human-readable evidence detail. ctx carries cross-scenario state.
# --------------------------------------------------------------------------


def sc1_health_and_thread_start(client: httpx.Client, ctx: dict) -> str:
    r = client.get("/health")
    _require(r.status_code == 200, f"GET /health returned {r.status_code}: {r.text[:200]}")
    payload = r.json()
    _require(payload.get("status") == "ok", f"/health payload unexpected: {payload}")
    ctx["graph_mode"] = str(payload.get("graph", "unknown"))

    ticket = ctx["ticket"]
    r = client.post("/threads", json={"ticket": ticket})
    _require(r.status_code == 200, f"POST /threads returned {r.status_code}: {r.text[:200]}")
    body = r.json()
    thread_id = body.get("thread_id")
    _require(
        isinstance(thread_id, str) and bool(thread_id),
        f"POST /threads response missing thread_id: {body}",
    )
    spec = body.get("spec")
    _require(isinstance(spec, dict), f"POST /threads response missing spec: {body}")
    _require(spec.get("pedido") == ticket, f"spec.pedido != ticket: {spec.get('pedido')!r}")
    estado = spec.get("estado") or {}
    _require(
        isinstance(estado.get("razon"), str) and bool(estado["razon"]),
        f"spec.estado.razon missing: {estado}",
    )
    ctx["thread_id"] = thread_id
    return f"health ok (graph={ctx['graph_mode']}); thread {thread_id[:8]}…; spec.pedido == ticket"


def _assert_citations(spec: dict, where: str) -> list[str]:
    choques = spec.get("choques")
    _require(isinstance(choques, list), f"{where}: spec.choques missing or not a list: {spec}")
    _require(len(choques) > 0, f"{where}: no citations returned for a corpus-grounded question")
    doc_ids = [str(c.get("document_id", "")) for c in choques]
    _require(
        all(d for d in doc_ids),
        f"{where}: citation without document_id: {choques}",
    )
    return doc_ids


def _rag_golden_question(client: httpx.Client, ctx: dict, index: int) -> str:
    question, expected_doc = ctx["golden"][index]
    r = client.post("/threads", json={"ticket": question})
    _require(r.status_code == 200, f"POST /threads returned {r.status_code}: {r.text[:200]}")
    body = r.json()
    thread_id = body.get("thread_id")
    spec = body.get("spec")
    _require(isinstance(spec, dict), f"POST /threads response missing spec: {body}")
    doc_ids = _assert_citations(spec, f"golden[{index}] {question!r}")
    key = f"golden{index + 1}_thread_id"
    ctx[key] = thread_id
    return (
        f"golden {question!r} → {len(doc_ids)} citation(s) {doc_ids} "
        f"(expected doc {expected_doc}); thread {str(thread_id)[:8]}…"
    )


def sc2_rag_golden_question(client: httpx.Client, ctx: dict) -> str:
    return _rag_golden_question(client, ctx, 0)


def sc3_second_golden_question(client: httpx.Client, ctx: dict) -> str:
    detail = _rag_golden_question(client, ctx, 1)
    prev = ctx.get("golden1_thread_id")
    curr = ctx.get("golden2_thread_id")
    _require(
        prev and curr and prev != curr,
        "S3 did not run on a fresh thread (thread_id repeated)",
    )
    return detail + "; new thread confirmed"


def sc4_multiturn_continuity(client: httpx.Client, ctx: dict) -> str:
    thread_id = ctx.get("thread_id")
    _require(isinstance(thread_id, str) and thread_id, "S4 needs S1's thread_id")
    ticket = ctx["ticket"]
    marker = uuid.uuid4().hex[:8]  # unique marker findable in traces' input values
    r = client.post(f"/threads/{thread_id}/messages", json={"content": f"{FOLLOW_UP} [qa:{marker}]"})
    _require(
        r.status_code == 200,
        f"POST /threads/{{id}}/messages returned {r.status_code}: {r.text[:200]} "
        "(a 404 here would mean the checkpointer lost the thread)",
    )
    spec = r.json().get("spec") or {}
    _require(
        spec.get("pedido") == ticket,
        f"continuity broken: spec.pedido changed to {spec.get('pedido')!r}",
    )
    estado = spec.get("estado") or {}
    _require(
        isinstance(estado.get("razon"), str) and bool(estado["razon"]),
        f"second response has no estado.razon: {estado}",
    )
    return (
        f"same thread {thread_id[:8]}… accepted a follow-up (marker qa:{marker}); "
        "spec.pedido still the original ticket → checkpointer kept state"
    )


def sc5_malformed_payload(client: httpx.Client, ctx: dict) -> str:
    r = client.post("/threads", json={})
    _require(
        r.status_code == 422,
        f"expected 422 for a payload missing 'ticket', got {r.status_code}: {r.text[:200]}",
    )
    detail = r.json().get("detail")
    _require(
        isinstance(detail, list) and bool(detail),
        f"422 response without Pydantic/FastAPI error detail: {r.text[:200]}",
    )
    locs = [str(e.get("loc", "")) for e in detail if isinstance(e, dict)]
    msgs = [str(e.get("msg", "")) for e in detail if isinstance(e, dict)]
    _require(
        any("ticket" in loc for loc in locs),
        f"error detail does not point at the 'ticket' field: locs={locs}, msgs={msgs}",
    )
    return f"422 with Pydantic error detail (locs={locs}, msgs={msgs})"


def sc6_close_thread(client: httpx.Client, ctx: dict) -> str:
    thread_id = ctx.get("thread_id")
    _require(isinstance(thread_id, str) and thread_id, "S6 needs S1's thread_id")
    r = client.post(f"/threads/{thread_id}/close")
    _require(r.status_code == 200, f"POST /threads/{{id}}/close returned {r.status_code}: {r.text[:200]}")
    spec = r.json().get("spec") or {}
    detail = "close accepted (200); final spec returned"
    if ctx.get("graph_mode") == "fake":
        _require(
            spec.get("preguntas") == [],
            f"fake graph should clear preguntas on close, got {spec.get('preguntas')!r}",
        )
        _require(
            (spec.get("estado") or {}).get("se_puede_cerrar") is True,
            "fake graph should mark se_puede_cerrar=True after close",
        )
        detail += "; preguntas cleared and se_puede_cerrar=True"
    return detail


SCENARIOS: tuple[tuple[str, str, object], ...] = (
    ("S1", "health + thread start", sc1_health_and_thread_start),
    ("S2", "RAG golden question #1 → citations", sc2_rag_golden_question),
    ("S3", "RAG golden question #2, new thread", sc3_second_golden_question),
    ("S4", "multi-turn continuity (checkpointer)", sc4_multiturn_continuity),
    ("S5", "Pydantic validation (malformed payload)", sc5_malformed_payload),
    ("S6", "close thread (bonus)", sc6_close_thread),
)
MANDATORY_IDS = frozenset({"S1", "S2", "S3", "S4", "S5"})


@dataclass
class ScenarioResult:
    id: str
    name: str
    status: str
    detail: str = ""

    @property
    def is_mandatory(self) -> bool:
        return self.id in MANDATORY_IDS


@dataclass
class TraceReport:
    status: str  # PASS | FAIL | SKIP | NA
    detail: str
    per_scenario: dict[str, str] = field(default_factory=dict)


def load_golden(path: Path = GOLDEN_PATH, needed: int = 2) -> list[tuple[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))["casos"]
    cases = [(c["pregunta"], c["documento_id_esperado"]) for c in data]
    if len(cases) < needed:
        raise ValueError(f"golden_set.json needs at least {needed} cases, found {len(cases)}")
    return cases


def run_scenarios(
    client: httpx.Client,
    *,
    golden: list[tuple[str, str]] | None = None,
    ticket: str | None = None,
) -> tuple[list[ScenarioResult], list[tuple[str, dt.datetime, dt.datetime]]]:
    """Run all scenarios against ``client`` (real HTTP or ASGI transport).

    Returns the results plus, per scenario, the (start, end) UTC wall-clock
    window in which its API calls happened — used by the trace check.
    """
    ctx: dict = {
        "golden": golden if golden is not None else load_golden(),
        "ticket": ticket or _qa_ticket(),
    }
    results: list[ScenarioResult] = []
    windows: list[tuple[str, dt.datetime, dt.datetime]] = []
    for sid, name, fn in SCENARIOS:
        started = _utcnow()
        try:
            detail = str(fn(client, ctx))
            results.append(ScenarioResult(sid, name, PASS, detail))
        except Exception as exc:  # noqa: BLE001 — any failure is a scenario FAIL
            results.append(ScenarioResult(sid, name, FAIL, str(exc)))
        finally:
            windows.append((sid, started, _utcnow()))
    return results, windows


# --------------------------------------------------------------------------
# Trace verification (Arize Phoenix). Injected in tests via a TraceReport.
# --------------------------------------------------------------------------


def _langsmith_active() -> bool:
    return any(
        os.getenv(name, "").strip().lower() in ("1", "true", "yes")
        for name in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING")
    )


def verify_phoenix_traces(
    windows: list[tuple[str, dt.datetime, dt.datetime]],
    *,
    phoenix_url: str = DEFAULT_PHOENIX_URL,
    project: str = PROJECT_NAME,
    wait_seconds: float = 60.0,
) -> TraceReport:
    """Verify every scenario window has spans in Phoenix.

    Fallbacks (never fabricates a pass):
      - LangSmith tracing active       → NA (Phoenix check not applicable)
      - phoenix package missing        → SKIP
      - Phoenix unreachable            → SKIP (clear message)
      - reachable but a window has no spans after the wait → FAIL
    """
    if _langsmith_active():
        project_ls = (
            os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"
        )
        return TraceReport(
            NA,
            "LangSmith tracing is active (LANGCHAIN_TRACING_V2/LANGSMITH_TRACING): "
            f"traces land in the LangSmith UI under project '{project_ls}'. "
            "Phoenix check not applicable.",
        )
    try:
        from phoenix.client import Client
    except ImportError as exc:
        return TraceReport(
            SKIP,
            f"phoenix package not installed ({exc}); install arize-phoenix or use "
            "docker compose. Trace check skipped — NOT counted as a pass.",
        )

    try:
        httpx.get(phoenix_url.rstrip("/") + "/", timeout=5.0)
    except httpx.HTTPError as exc:
        return TraceReport(
            SKIP,
            f"Phoenix not reachable at {phoenix_url} ({exc.__class__.__name__}). "
            "Start docker compose or a local `phoenix serve`, or pass "
            "--phoenix-url. Trace check skipped — NOT counted as a pass.",
        )

    client = Client(base_url=phoenix_url)
    pending: dict[str, tuple[dt.datetime, dt.datetime]] = {
        sid: (t0 - TRACE_WINDOW_EPSILON, t1) for sid, t0, t1 in windows
    }
    per_scenario: dict[str, str] = {}
    last_error = ""
    deadline = time.monotonic() + wait_seconds
    while True:
        for sid, (t0, t1) in list(pending.items()):
            try:
                df = client.spans.get_spans_dataframe(
                    project_identifier=project,
                    start_time=t0,
                    end_time=t1,
                    timeout=10,
                )
            except ImportError as exc:
                return TraceReport(
                    SKIP, f"pandas (required by the Phoenix client) is missing: {exc}", {}
                )
            except httpx.HTTPError as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                continue
            if df is not None and len(df) > 0:
                names = (
                    df["name"].value_counts().to_dict() if "name" in df.columns else {}
                )
                top = ", ".join(f"{k}×{v}" for k, v in list(names.items())[:4])
                per_scenario[sid] = f"{len(df)} span(s) in window [{top}]"
                del pending[sid]
        if not pending or time.monotonic() > deadline:
            break
        time.sleep(2.0)

    if not pending:
        return TraceReport(
            PASS,
            f"Phoenix project '{project}' returned spans for all {len(per_scenario)} "
            "scenario windows — traces are visible and correlable per scenario.",
            per_scenario,
        )
    missing = ", ".join(sorted(pending))
    detail = (
        f"Phoenix reachable at {phoenix_url} but no spans found for: {missing} "
        f"after {wait_seconds:.0f}s."
    )
    if last_error:
        detail += f" Last query error: {last_error}"
    detail += (
        " Verify the API process has PHOENIX_COLLECTOR_ENDPOINT set and that "
        f"instrumentation is registered (project '{project}')."
    )
    return TraceReport(FAIL, detail, per_scenario)


# --------------------------------------------------------------------------
# Reporting / orchestration
# --------------------------------------------------------------------------


def compute_exit_code(results: list[ScenarioResult], trace_report: TraceReport | None) -> int:
    if any(r.status == FAIL for r in results if r.is_mandatory):
        return 1
    if trace_report is not None and trace_report.status == FAIL:
        return 1
    return 0


def render_table(results: list[ScenarioResult]) -> str:
    lines = [
        f"{'ID':<4} {'MAND':<5} {'RESULT':<6} SCENARIO",
        "-" * 78,
    ]
    for r in results:
        mand = "yes" if r.is_mandatory else "bonus"
        lines.append(f"{r.id:<4} {mand:<5} {r.status:<6} {r.name}")
        for chunk in (r.detail or "").splitlines() or [""]:
            lines.append(f"{'':<17} {chunk}")
    return "\n".join(lines)


def render_summary(results: list[ScenarioResult], trace_report: TraceReport | None) -> str:
    mandatory = [r for r in results if r.is_mandatory]
    passed = sum(1 for r in mandatory if r.status == PASS)
    lines = [f"==> {passed}/{len(mandatory)} scenarios passed"]
    if trace_report is not None:
        lines.append(f"==> traces: [{trace_report.status}] {trace_report.detail}")
        for sid, note in trace_report.per_scenario.items():
            lines.append(f"    {sid}: {note}")
    return "\n".join(lines)


def _selftest() -> tuple[list[ScenarioResult], list[tuple[str, dt.datetime, dt.datetime]], TraceReport]:
    """Run the scenarios in-process against the fake graph (no server, no creds)."""
    import asyncio

    # config.py reads this at import time: make /health report the fake mode
    # (and thus enable the fake-specific assertions in S6).
    os.environ.setdefault("SPEC_REFINERY_GRAPH", "fake")

    from agents.fakes import fake_intake, fake_retriever, fake_supervisor, fake_writer
    from app import app, get_graph
    from checkpoint import close_checkpointer, create_checkpointer
    from graph import build_graph

    async def _build():
        cp = create_checkpointer(
            Path(tempfile.mkdtemp(prefix="qa-system-selftest-")) / "qa.sqlite"
        )
        graph = build_graph(
            supervisor=fake_supervisor,
            retriever=fake_retriever,
            intake=fake_intake,
            writer=fake_writer,
            checkpointer=cp,
        )
        return graph, cp

    graph, checkpointer = asyncio.run(_build())
    app.dependency_overrides[get_graph] = lambda: graph
    try:
        # TestClient = httpx over the ASGI transport (httpx 0.28 dropped sync
        # ASGITransport context-manager support; TestClient handles it).
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            results, windows = run_scenarios(client)
    finally:
        app.dependency_overrides.clear()
        # aiosqlite's worker thread is non-daemon: close it or the process hangs.
        asyncio.run(close_checkpointer(checkpointer))
    trace = TraceReport(
        SKIP,
        "--selftest runs in-process against the fake graph; no Phoenix spans are "
        "emitted here. Run against docker compose (python qa_system.py) for the "
        "real trace verification.",
    )
    return results, windows, trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="System-evidence tests for Spec Refinery.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL (QA_BASE_URL)")
    parser.add_argument(
        "--phoenix-url",
        default=DEFAULT_PHOENIX_URL,
        help="Phoenix UI/API URL for the trace check (QA_PHOENIX_URL)",
    )
    parser.add_argument(
        "--skip-traces", action="store_true", help="run scenarios only, skip the trace check"
    )
    parser.add_argument(
        "--trace-wait",
        type=float,
        default=60.0,
        help="seconds to wait for spans to appear in Phoenix",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run in-process against the fake graph (no server, no Phoenix check)",
    )
    args = parser.parse_args(argv)

    print("Spec Refinery — system evidence tests (qa_system.py)")
    if args.selftest:
        print("mode: in-process selftest (fake graph, no server)\n")
        results, _windows, trace_report = _selftest()
    else:
        print(f"mode: live API at {args.base_url} (server must already be running)\n")
        with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
            results, windows = run_scenarios(client)
        if args.skip_traces:
            trace_report = TraceReport(SKIP, "trace check disabled with --skip-traces.")
        else:
            print("verifying Phoenix traces for each scenario window…")
            trace_report = verify_phoenix_traces(
                windows, phoenix_url=args.phoenix_url, wait_seconds=args.trace_wait
            )

    print(render_table(results))
    print()
    print(render_summary(results, trace_report))
    return compute_exit_code(results, trace_report)


if __name__ == "__main__":
    raise SystemExit(main())
