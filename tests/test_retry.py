"""Política de reintentos del grafo.

El grafo no reintenta un nodo por su cuenta: los clientes del modelo ya
reintentan los fallos transitorios del proveedor (``LLM_MAX_RETRIES``), así que
un reintento a nivel de nodo multiplicaría el peor caso de un turno.
"""

from __future__ import annotations

from agents.retry import NODE_RETRY, is_transient_error


class _UnauthorizedError(Exception):
    """Error de cliente: reintentar no lo arregla."""


def test_the_graph_does_not_re_attempt_a_node() -> None:
    assert NODE_RETRY.max_attempts == 1


def test_transient_provider_failures_are_retried_by_the_classifier() -> None:
    assert is_transient_error(RuntimeError("Provider returned error")) is True


def test_client_errors_are_not_retried() -> None:
    assert is_transient_error(_UnauthorizedError("nope")) is False
