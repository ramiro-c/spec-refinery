"""LLM request timeout / bounded retries exposed through config."""

from __future__ import annotations

import importlib

import config


def test_llm_timeout_and_retries_defaults():
    """Documented defaults: 90 s per request, 2 bounded SDK retries."""
    assert config.LLM_TIMEOUT_SECONDS == 90.0
    assert config.LLM_MAX_RETRIES == 2


def test_llm_timeout_and_retries_env_override(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "17.5")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")

    reloaded = importlib.reload(config)
    try:
        assert reloaded.LLM_TIMEOUT_SECONDS == 17.5
        assert reloaded.LLM_MAX_RETRIES == 1
    finally:
        importlib.reload(config)
