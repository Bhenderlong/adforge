"""
Publishing adapter contract.

Every adapter goes through the platform's official API. That is a deliberate
design constraint, not an accident of convenience:

  - There is nothing to detect. Automated posting through a documented API is
    the sanctioned path; posting through a scripted browser session is what
    account bans and rate-limit escalation exist to catch.
  - Failures are legible. An API returns a documented error you can act on; a
    scraped UI returns a changed selector and silent breakage.
  - On Reddit specifically, evasion risks a sitewide DOMAIN ban, which would
    prevent anyone - not just this tool - from linking inferix.co anywhere on
    the platform. That is unrecoverable and vastly outweighs the upside.

Adapters must honour `dry_run`: build and log the exact request, transmit
nothing. This is the default until the user explicitly enables an account.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

log = logging.getLogger("adforge.publish")


class PublishError(RuntimeError):
    """Adapter could not publish. `retryable` distinguishes transient failures."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class NotConfigured(PublishError):
    def __init__(self, what: str):
        super().__init__(f"not configured: {what}", retryable=False)


@dataclass
class Payload:
    """What gets published, already validated against the platform spec."""

    brand: str
    platform: str
    body: str
    title: str = ""
    link: str = ""
    image: Path | None = None
    video: Path | None = None
    extra_parts: list[str] = field(default_factory=list)
    # Platform-specific: subreddit, discord channel id, youtube privacy, etc.
    options: dict = field(default_factory=dict)


@dataclass
class Result:
    ok: bool
    remote_id: str = ""
    url: str = ""
    detail: str = ""
    dry_run: bool = False


class Adapter(Protocol):
    key: str
    # Human-readable list of credential fields the UI should collect.
    required_credentials: tuple[str, ...]

    def validate(self, creds: dict, options: dict) -> list[str]:
        """Return a list of problems; empty means ready to publish."""

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        ...


class BaseAdapter:
    key = "base"
    required_credentials: tuple[str, ...] = ()
    # Conservative floor between two posts to the same account, in seconds.
    # Well inside every platform's published limit; the point is to avoid
    # bursts, not to approach the ceiling.
    min_interval_seconds = 60

    _last_post: dict[str, float] = {}

    def validate(self, creds: dict, options: dict) -> list[str]:
        return [f"missing credential: {k}" for k in self.required_credentials
                if not str(creds.get(k, "")).strip()]

    def throttle(self, account_key: str) -> None:
        last = self._last_post.get(account_key, 0.0)
        wait = self.min_interval_seconds - (time.time() - last)
        if wait > 0:
            log.info("%s: throttling %.0fs", self.key, wait)
            time.sleep(wait)
        self._last_post[account_key] = time.time()

    def dry(self, payload: Payload, note: str = "") -> Result:
        log.info(
            "[DRY RUN] %s/%s would publish:\n  title: %s\n  body: %s\n"
            "  link: %s\n  image: %s\n  video: %s\n  options: %s\n  %s",
            payload.brand, self.key, payload.title or "-",
            payload.body[:400].replace("\n", " | "),
            payload.link or "-", payload.image or "-", payload.video or "-",
            payload.options or "-", note,
        )
        return Result(ok=True, dry_run=True, detail="dry run - nothing transmitted")
