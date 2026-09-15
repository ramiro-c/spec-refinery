"""Política de reintentos del grafo.

El grafo hace un único reintento por nodo (dos intentos en total): los clientes
del modelo ya reintentan los fallos transitorios del proveedor
(``LLM_MAX_RETRIES``) y subir ``max_attempts`` alarga el peor caso de un turno.
"""

from __future__ import annotations

from agents.retry import NODE_RETRY, is_transient_error


class _UnauthorizedError(Exception):
    """Error de cliente: reintentar no lo arregla."""


def test_the_graph_re_attempts_a_node_once() -> None:
    assert NODE_RETRY.max_attempts == 2


def test_transient_provider_failures_are_retried_by_the_classifier() -> None:
    assert is_transient_error(RuntimeError("Provider returned error")) is True


def test_client_errors_are_not_retried() -> None:
    assert is_transient_error(_UnauthorizedError("nope")) is False
