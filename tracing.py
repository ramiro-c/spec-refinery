"""Instrumentación Phoenix si hay collector (no embebe la UI)."""

from __future__ import annotations

import os

# Host-side por defecto: docker-compose publica Phoenix en 6010 -> 6006.
DEFAULT_PHOENIX_ENDPOINT = "http://localhost:6010/v1/traces"


def resolve_endpoint() -> str:
    """Endpoint de Phoenix: valor explícito de env o default host-side."""
    return os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip() or DEFAULT_PHOENIX_ENDPOINT


def setup_tracing() -> None:
    """Registra OpenTelemetry hacia Phoenix. No-op si falta el paquete."""
    endpoint = resolve_endpoint()
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        return
    register(project_name="spec-refinery", endpoint=endpoint, batch=True)
    LangChainInstrumentor().instrument()
