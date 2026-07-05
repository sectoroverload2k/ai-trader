"""Configuration loaded from environment variables (.env).

Secrets are read from the environment so they never get hard-coded or committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads .env from the project root if present
except ImportError:  # python-dotenv is optional at runtime
    pass


@dataclass(frozen=True)
class Settings:
    api_key: str
    secret_key: str
    paper: bool

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.secret_key)


def load_settings() -> Settings:
    """Read Alpaca credentials + mode from the environment.

    Never raises on missing keys — callers check `settings.is_configured` so the
    rest of the code (e.g. backtests) can run without live credentials.
    """
    return Settings(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper=os.getenv("ALPACA_PAPER", "true").lower() != "false",
    )
