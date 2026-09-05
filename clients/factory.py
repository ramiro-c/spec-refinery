"""Factory multi-proveedor (mismo wiring que pre-entrega-6, recortado).

Con ``LLM_PROVIDER=openrouter`` cada rol del grafo usa un modelo distinto:
supervisor rutea, writer redacta la spec. Gemini comparte un solo modelo
(``DEFAULT_MODELS``).
"""

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel

from config import GEMINI_API_KEY, LLM_PROVIDER
from schemas import ProviderName, RoleName

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
}

# OpenRouter :free — un modelo por rol.
# supervisor: Nemotron 3 Ultra. Ruteo corto; structured output.
# writer: MiniMax M3. Mejor razonamiento/IF para redactar la spec.
OPENROUTER_ROLE_MODELS: dict[RoleName, str] = {
    "supervisor": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "writer": "minimax/minimax-m3:free",
}

ROLE_TEMPERATURE: dict[RoleName, float] = {
    "supervisor": 0.0,
    "writer": 0.0,
}


def _normalize_provider(provider: str) -> ProviderName:
    value = provider.strip().lower()
    if value not in {"gemini", "openrouter"}:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider} use gemini (Vertex) or openrouter"
        )
    return cast(ProviderName, value)


def _openrouter_model(role: RoleName | None, model: str | None) -> str:
    if model:
        return model
    if role:
        return OPENROUTER_ROLE_MODELS[role]
    return DEFAULT_MODELS["openrouter"]


def build_chat_model(
    provider: str | None = None,
    temperature: float | None = None,
    *,
    role: RoleName | None = None,
    model: str | None = None,
) -> BaseChatModel:
    resolved = _normalize_provider(provider or LLM_PROVIDER)
    if temperature is None:
        temperature = ROLE_TEMPERATURE.get(role, 0.2) if role else 0.2

    if resolved == "gemini":
        from google.genai.types import AutomaticFunctionCallingConfig
        from langchain_google_genai import ChatGoogleGenerativeAI

        # create_agent ya ejecuta las tools. El AFC de Gemini avisa (y puede
        # pelearse con el ReAct) si Models.generate_content auto-llama funciones.
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
            model=model or DEFAULT_MODELS["gemini"],
            temperature=temperature,
        )

    if resolved == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=_openrouter_model(role, model),
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider: {resolved}")


def build_role_models(
    provider: str | None = None,
) -> dict[RoleName, BaseChatModel]:
    """Dos LLMs listos para el grafo. OpenRouter los diferencia; gemini no."""
    resolved = _normalize_provider(provider or LLM_PROVIDER)
    if resolved == "openrouter":
        return {
            "supervisor": build_chat_model(provider="openrouter", role="supervisor"),
            "writer": build_chat_model(provider="openrouter", role="writer"),
        }
    llm = build_chat_model(provider=resolved)
    return {"supervisor": llm, "writer": llm}
