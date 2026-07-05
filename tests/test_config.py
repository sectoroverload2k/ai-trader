import pytest

from ai_trader.config import LIVE_ENDPOINT, PAPER_ENDPOINT, load_settings


def test_reads_api_secret_name(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setenv("ALPACA_API_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    s = load_settings()
    assert s.is_configured
    assert s.secret_key == "s"
    assert s.is_paper is True
    s.require_paper()  # should not raise


def test_falls_back_to_secret_key_name(monkeypatch):
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "legacy")
    monkeypatch.delenv("ALPACA_API_ENDPOINT", raising=False)
    s = load_settings()
    assert s.secret_key == "legacy"


def test_live_endpoint_blocks_orders(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setenv("ALPACA_API_ENDPOINT", LIVE_ENDPOINT)
    s = load_settings()
    assert s.is_paper is False
    with pytest.raises(RuntimeError):
        s.require_paper()


def test_missing_endpoint_defaults_to_paper(monkeypatch):
    monkeypatch.delenv("ALPACA_API_ENDPOINT", raising=False)
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    s = load_settings()
    assert s.endpoint == PAPER_ENDPOINT
    assert s.is_paper is True


def test_not_configured_when_keys_missing(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert load_settings().is_configured is False
