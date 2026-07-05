"""Configuration loaded from environment variables (.env or GitHub Actions secrets).

Secrets are read from the environment so they never get hard-coded or committed.

Accepts either naming convention:
  - ALPACA_API_KEY        (key id)
  - ALPACA_API_SECRET  or ALPACA_SECRET_KEY   (secret)
  - ALPACA_API_ENDPOINT   (base URL; determines paper vs live) or ALPACA_PAPER=true/false
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads .env from the project root if present
except ImportError:  # python-dotenv is optional at runtime
    pass

PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
LIVE_ENDPOINT = "https://api.alpaca.markets"


@dataclass(frozen=True)
class Settings:
    api_key: str
    secret_key: str
    endpoint: str

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.secret_key)

    @property
    def is_paper(self) -> bool:
        """True unless the endpoint is explicitly the live one.

        Fail-safe: anything that isn't clearly the live host counts as paper, so a
        blank/misconfigured endpoint never silently trades real money.
        """
        return "paper" in self.endpoint or self.endpoint != LIVE_ENDPOINT

    def require_paper(self) -> None:
        """Guard for any code path that could place orders."""
        if not self.is_paper:
            raise RuntimeError(
                "Refusing to run: ALPACA_API_ENDPOINT points at the LIVE endpoint "
                f"({self.endpoint}). This project is paper-only. Set it to "
                f"{PAPER_ENDPOINT}."
            )


def load_settings() -> Settings:
    """Read Alpaca credentials + endpoint from the environment.

    Never raises on missing keys — callers check `settings.is_configured` so the
    rest of the code (e.g. backtests) can run without live credentials.
    """
    secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY", "")

    endpoint = os.getenv("ALPACA_API_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        # Fall back to the ALPACA_PAPER flag if no explicit endpoint was given.
        paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        endpoint = PAPER_ENDPOINT if paper else LIVE_ENDPOINT

    return Settings(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=secret,
        endpoint=endpoint,
    )
