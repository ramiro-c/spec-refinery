"""Fault tolerance del grafo: RetryPolicy + error_handler (LangGraph ≥ 1.2).

``Provider returned error`` (OpenRouter) es transitorio, pero a veces llega
como ``RuntimeError``. ``default_retry_on`` no reintenta RuntimeError, así
que el clasificador mira el mensaje / el tipo del SDK.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy, default_retry_on

# Mensaje y nombre de tipo (p. ej. BadGatewayError → "badgateway").
_RETRY_TOKENS = (
    "provider returned error",
    "overloaded",
    "rate-limited",
    "rate limit",
    "timeout",
    "temporar",
    "bad gateway",
    "badgateway",
    "unavailable",
    "toomanyrequests",
    "internalserver",
    "502",
    "503",
    "504",
    "429",
)

_NO_RETRY_TYPE_TOKENS = (
    "forbidden",
    "unauthorized",
    "badrequest",
    "notfound",
    "paymentrequired",
    "unprocessable",
)


def is_transient_error(exc: BaseException) -> bool:
    """True si el sistema debe reintentar solo (red / free tier / 5xx)."""
    name = type(exc).__name__.lower()
    if any(token in name for token in _NO_RETRY_TYPE_TOKENS):
        return False
    blob = f"{name} {exc}".lower()
    if any(token in blob for token in _RETRY_TOKENS):
        return True
    return default_retry_on(exc) if isinstance(exc, Exception) else False


# Los clientes del modelo ya reintentan los errores transitorios del proveedor
# (LLM_MAX_RETRIES); el grafo no vuelve a intentar el nodo por su cuenta: un
# intento por nodo mantiene acotado el peor caso. Subir max_attempts reactiva
# los reintentos del grafo.
NODE_RETRY = RetryPolicy(
    max_attempts=1,
    initial_interval=2.0,
    backoff_factor=2.0,
    retry_on=is_transient_error,
)


def node_error_handler(state: dict, error: NodeError) -> Command:
    """Después de agotar retries: anotá el fallo y mandá a ``writer``."""
    detail = error.error
    msg = (
        f"El nodo `{error.node}` falló después de reintentos "
        f"({type(detail).__name__}: {detail}). Paso al writer."
    )
    return Command(
        update={
            "last_error": msg,
            "messages": [AIMessage(content=msg, name=error.node)],
            "next_agent": "FINISH",
            "last_agent": error.node,
        },
        goto="writer",
    )
