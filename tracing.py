"""Instrumentación Phoenix si hay collector (no embebe la UI)."""

from __future__ import annotations

import os


def setup_tracing() -> None:
    """Registra OpenTelemetry hacia Phoenix. No-op si falta endpoint o paquete."""
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    if not endpoint:
        return
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        return
    register(project_name="spec-refinery", endpoint=endpoint)
    LangChainInstrumentor().instrument()
