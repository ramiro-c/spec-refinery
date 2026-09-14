import pytest

from tracing import DEFAULT_PHOENIX_ENDPOINT, resolve_endpoint, setup_tracing


def test_resolver_uses_explicit_env_value(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
    assert resolve_endpoint() == "http://phoenix:6006/v1/traces"


def test_resolver_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    assert resolve_endpoint() == "http://localhost:6010/v1/traces"
    assert resolve_endpoint() == DEFAULT_PHOENIX_ENDPOINT


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_resolver_falls_back_when_blank(monkeypatch, blank):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", blank)
    assert resolve_endpoint() == DEFAULT_PHOENIX_ENDPOINT


def test_setup_tracing_does_not_raise_when_unset(monkeypatch):
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    setup_tracing()
