"""Factory multi-proveedor: solo gemini (Vertex) y openrouter."""

from __future__ import annotations

import sys
import types

import pytest

from clients.factory import (
    DEFAULT_MODELS,
    OPENROUTER_ROLE_MODELS,
    ROLE_TEMPERATURE,
    _normalize_provider,
    build_chat_model,
    build_role_models,
)


def _inyectar_openrouter(monkeypatch, fake):
    modulo = types.ModuleType("langchain_openrouter")
    modulo.ChatOpenRouter = fake
    monkeypatch.setitem(sys.modules, "langchain_openrouter", modulo)


def _fake_chat():
    llamadas: list[dict] = []

    class _ChatFake:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            llamadas.append(kwargs)

    return _ChatFake, llamadas


def test_openai_provider_lanza_value_error():
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER: openai use gemini \\(Vertex\\) or openrouter"):
        build_chat_model(provider="openai")


def test_normalize_provider_solo_gemini_u_openrouter():
    assert _normalize_provider("gemini") == "gemini"
    assert _normalize_provider("  openrouter ") == "openrouter"


def test_provider_invalido_lanza_value_error():
    with pytest.raises(ValueError, match="use gemini \\(Vertex\\) or openrouter"):
        _normalize_provider("groq")


def test_openrouter_asigna_modelo_por_rol(monkeypatch):
    fake, llamadas = _fake_chat()
    _inyectar_openrouter(monkeypatch, fake)

    build_chat_model(provider="openrouter", role="supervisor")
    build_chat_model(provider="openrouter", role="writer")

    assert llamadas[0]["model"] == OPENROUTER_ROLE_MODELS["supervisor"]
    assert llamadas[1]["model"] == OPENROUTER_ROLE_MODELS["writer"]
    assert llamadas[0]["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert llamadas[1]["model"] == "minimax/minimax-m3:free"


def test_openrouter_temperatura_por_rol(monkeypatch):
    fake, llamadas = _fake_chat()
    _inyectar_openrouter(monkeypatch, fake)

    build_chat_model(provider="openrouter", role="supervisor")
    build_chat_model(provider="openrouter", role="writer")

    assert llamadas[0]["temperature"] == ROLE_TEMPERATURE["supervisor"] == 0.0
    assert llamadas[1]["temperature"] == ROLE_TEMPERATURE["writer"] == 0.0


def test_build_role_models_openrouter_dos_instancias(monkeypatch):
    fake, llamadas = _fake_chat()
    _inyectar_openrouter(monkeypatch, fake)

    models = build_role_models(provider="openrouter")

    assert set(models) == {"supervisor", "writer"}
    assert [c["model"] for c in llamadas] == [
        OPENROUTER_ROLE_MODELS["supervisor"],
        OPENROUTER_ROLE_MODELS["writer"],
    ]


def test_gemini_envuelve_chat_para_apagar_afc(monkeypatch):
    fake, llamadas = _fake_chat()
    modulo = types.ModuleType("langchain_google_genai")
    modulo.ChatGoogleGenerativeAI = fake
    monkeypatch.setitem(sys.modules, "langchain_google_genai", modulo)
    genai_types = types.ModuleType("google.genai.types")
    genai_types.AutomaticFunctionCallingConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    if "google.genai" not in sys.modules:
        genai_mod = types.ModuleType("google.genai")
        genai_mod.types = genai_types
        monkeypatch.setitem(sys.modules, "google.genai", genai_mod)

    modelo = build_chat_model(provider="gemini")

    assert type(modelo).__name__ == "_GeminiChat"
    assert isinstance(modelo, fake)
    assert llamadas[0]["model"] == DEFAULT_MODELS["gemini"]
