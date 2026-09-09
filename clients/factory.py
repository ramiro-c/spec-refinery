"""Multi-provider chat model factory (Vertex / OpenRouter).

With ``LLM_PROVIDER=openrouter`` each graph role can use its own model via
``SUPERVISOR_MODEL`` / ``WRITER_MODEL`` env vars (supervisor routes, writer
drafts the spec). Gemini shares a single model unless a role env override is
set. Model IDs are never hardcoded in call sites: defaults live here, env
vars (read through ``config``) always win.
"""

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel

from config import GEMINI_API_KEY, LLM_PROVIDER, SUPERVISOR_MODEL, WRITER_MODEL
from schemas import ProviderName, RoleName

GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def _normalize_provider(provider: str) -> ProviderName:
    value = provider.strip().lower()
    if value not in {"gemini", "openrouter"}:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider} use gemini (Vertex) or openrouter"
        )
    return cast(ProviderName, value)


def _model_for(provider: str, role: RoleName | None, model: str | None) -> str:
    """Resolve the model id: explicit arg > role env override > provider default."""
    if model:
        return model
    if role == "supervisor" and SUPERVISOR_MODEL:
        return SUPERVISOR_MODEL
    if role == "writer" and WRITER_MODEL:
        return WRITER_MODEL
    return GEMINI_DEFAULT_MODEL if provider == "gemini" else OPENROUTER_DEFAULT_MODEL


def build_chat_model(
    provider: str | None = None,
    temperature: float | None = None,
    *,
    role: RoleName | None = None,
    model: str | None = None,
) -> BaseChatModel:
    resolved = _normalize_provider(provider or LLM_PROVIDER)
    if temperature is None:
        temperature = 0.0 if role is not None else 0.2

    if resolved == "gemini":
        from google.genai.types import AutomaticFunctionCallingConfig
        from langchain_google_genai import ChatGoogleGenerativeAI

        # create_agent already executes the tools. Gemini's AFC warns (and can
        # clash with ReAct) if Models.generate_content auto-calls functions.
        class _GeminiChat(ChatGoogleGenerativeAI):
            def _prepare_request(self, *args, **kwargs):
                request = super()._prepare_request(*args, **kwargs)
                config = request.get("config")
                if config is not None:
                    config.automatic_function_calling = AutomaticFunctionCallingConfig(
                        disable=True
                    )
                return request

        return _GeminiChat(
            api_key=GEMINI_API_KEY,
            model=_model_for("gemini", role, model),
            temperature=temperature,
        )

    if resolved == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=_model_for("openrouter", role, model),
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider: {resolved}")


def build_role_models(
    provider: str | None = None,
) -> dict[RoleName, BaseChatModel]:
    """Two LLMs ready for the graph. OpenRouter differentiates them; gemini doesn't."""
    resolved = _normalize_provider(provider or LLM_PROVIDER)
    if resolved == "openrouter":
        return {
            "supervisor": build_chat_model(provider="openrouter", role="supervisor"),
            "writer": build_chat_model(provider="openrouter", role="writer"),
        }
    llm = build_chat_model(provider=resolved)
    return {"supervisor": llm, "writer": llm}
