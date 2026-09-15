"""Multi-provider factory: gemini (Vertex) and openrouter only."""

from __future__ import annotations

import sys
import types

import pytest

import clients.factory as factory
from clients.factory import (
    GEMINI_DEFAULT_MODEL,
    OPENROUTER_DEFAULT_MODEL,
    _normalize_provider,
    build_chat_model,
    build_role_models,
)
from config import LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS


def _inject_openrouter(monkeypatch, fake):
    module = types.ModuleType("langchain_openrouter")
    module.ChatOpenRouter = fake
    monkeypatch.setitem(sys.modules, "langchain_openrouter", module)


def _inject_gemini(monkeypatch, fake):
    module = types.ModuleType("langchain_google_genai")
    module.ChatGoogleGenerativeAI = fake
    monkeypatch.setitem(sys.modules, "langchain_google_genai", module)
    genai_types = types.ModuleType("google.genai.types")
    genai_types.AutomaticFunctionCallingConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    if "google.genai" not in sys.modules:
        genai_module = types.ModuleType("google.genai")
        genai_module.types = genai_types
        monkeypatch.setitem(sys.modules, "google.genai", genai_module)


def _fake_chat():
    calls: list[dict] = []

    class _ChatFake:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

    return _ChatFake, calls


def test_openai_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER: openai use gemini \\(Vertex\\) or openrouter"):
        build_chat_model(provider="openai")


def test_normalize_provider_allows_only_gemini_or_openrouter():
    assert _normalize_provider("gemini") == "gemini"
    assert _normalize_provider("  openrouter ") == "openrouter"


def test_invalid_provider_raises_value_error():
    with pytest.raises(ValueError, match="use gemini \\(Vertex\\) or openrouter"):
        _normalize_provider("groq")


def test_openrouter_uses_per_role_defaults(monkeypatch):
    fake, calls = _fake_chat()
    _inject_openrouter(monkeypatch, fake)

    build_chat_model(provider="openrouter", role="interrogator")
    build_chat_model(provider="openrouter", role="writer")

    assert calls[0]["model"] == OPENROUTER_DEFAULT_MODEL
    assert calls[1]["model"] == OPENROUTER_DEFAULT_MODEL
    assert calls[0]["temperature"] == 0.0
    assert calls[1]["temperature"] == 0.0


def test_openrouter_models_from_env(monkeypatch):
    fake, calls = _fake_chat()
    _inject_openrouter(monkeypatch, fake)
    monkeypatch.setattr(factory, "INTERROGATOR_MODEL", "test/interrogator:free")
    monkeypatch.setattr(factory, "WRITER_MODEL", "test/writer:free")

    build_chat_model(provider="openrouter", role="interrogator")
    build_chat_model(provider="openrouter", role="writer")

    assert calls[0]["model"] == "test/interrogator:free"
    assert calls[1]["model"] == "test/writer:free"


def test_gemini_shares_one_model(monkeypatch):
    fake, calls = _fake_chat()
    _inject_gemini(monkeypatch, fake)

    build_chat_model(provider="gemini", role="interrogator")
    build_chat_model(provider="gemini", role="writer")

    assert calls[0]["model"] == GEMINI_DEFAULT_MODEL
    assert calls[1]["model"] == GEMINI_DEFAULT_MODEL


def test_gemini_wraps_chat_to_disable_afc(monkeypatch):
    fake, calls = _fake_chat()
    _inject_gemini(monkeypatch, fake)

    model = build_chat_model(provider="gemini")

    assert type(model).__name__ == "_GeminiChat"
    assert isinstance(model, fake)
    assert calls[0]["model"] == GEMINI_DEFAULT_MODEL


def test_build_role_models_openrouter_one_instance_per_role(monkeypatch):
    fake, calls = _fake_chat()
    _inject_openrouter(monkeypatch, fake)

    models = build_role_models(provider="openrouter")

    assert set(models) == {"interrogator", "writer"}
    assert [c["model"] for c in calls] == [OPENROUTER_DEFAULT_MODEL] * 2


def test_gemini_threads_timeout_and_retries(monkeypatch):
    fake, calls = _fake_chat()
    _inject_gemini(monkeypatch, fake)

    build_chat_model(provider="gemini")

    assert calls[0]["timeout"] == LLM_TIMEOUT_SECONDS
    assert calls[0]["max_retries"] == LLM_MAX_RETRIES


def test_openrouter_threads_timeout_ms_and_retries(monkeypatch):
    fake, calls = _fake_chat()
    _inject_openrouter(monkeypatch, fake)

    build_chat_model(provider="openrouter", role="writer")

    # langchain-openrouter expects request_timeout in milliseconds.
    assert calls[0]["request_timeout"] == int(LLM_TIMEOUT_SECONDS * 1000)
    assert calls[0]["max_retries"] == LLM_MAX_RETRIES


def test_build_role_models_threads_timeout_and_retries(monkeypatch):
    fake, calls = _fake_chat()
    _inject_openrouter(monkeypatch, fake)

    build_role_models(provider="openrouter")

    assert all(
        c["request_timeout"] == int(LLM_TIMEOUT_SECONDS * 1000) for c in calls
    )
    assert all(c["max_retries"] == LLM_MAX_RETRIES for c in calls)
