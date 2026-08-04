"""
Conversation radar: find threads where the product genuinely answers a question.

Two stages, cheap before expensive:

  1. Keyword prefilter over recent posts in the configured targets. No model
     involved - this is what keeps the scan affordable at a 30-minute cadence.
  2. LLM relevance scoring on whatever survives, which decides whether the
     product is actually a useful answer here or would just be an ad.

Only threads that clear the target's `min_relevance` get a drafted reply, and
every draft requires human approval before it is sent.
"""

from __future__ import annotations

import datetime as dt
import logging

from ..config import settings
from ..db import Account, RadarTarget, RadarThread, session_scope, utcnow
from ..gpu import TEXT, gpu
from ..llm.client import LLMError, chat_json

log = logging.getLogger("adforge.radar")

# How far back a scan looks. Replying to a week-old thread is low value and
# reads as trawling.
MAX_AGE_HOURS = 72
PREFILTER_LIMIT = 60


SCORE_SYSTEM = """\
You judge whether a product is a genuinely useful answer to a forum thread.

You are protecting the brand from looking like a spammer, so be harsh. Score
0.0-1.0 where:
  0.9-1.0  the author is explicitly asking for exactly what this product does
  0.7-0.8  the author has a problem this product directly solves
  0.4-0.6  adjacent topic; a reply could be useful but the product is a stretch
  0.0-0.3  off-topic, or mentioning any product would be unwelcome

Score LOW when: the thread is a support question about a competitor's tool, a
job post, a rant, a beginner question answered better without any product, or
already resolved.

Also decide whether a human expert could write a genuinely useful reply here
even if they never named the product. If not, the score must be below 0.4.

Reply with ONLY JSON:
{"relevance":0.0-1.0,"reason":"one sentence","useful_without_product":true|false,
 "what_they_need":"one sentence"}
"""


def _score_thread(brand_key: str, brand_about: str, title: str, body: str) -> dict:
    user = f"""PRODUCT: {brand_about.strip()[:900]}

THREAD TITLE: {title}
THREAD BODY: {body[:1500]}

Score it."""
    try:
        with gpu(TEXT):
            return chat_json(
                [{"role": "system", "content": SCORE_SYSTEM},
                 {"role": "user", "content": user}],
                models=settings.fast_chain,
                temperature=0.2,
                max_tokens=350,
            )
    except LLMError as e:
        log.warning("scoring failed: %s", e)
        return {"relevance": 0.0, "reason": f"scoring failed: {e}"}


def _matches(text: str, keywords: list[str]) -> list[str]:
    low = text.lower()
    return [k for k in keywords if k in low]


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------


def _reddit_client(creds: dict):
    import praw

    return praw.Reddit(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        username=creds.get("username"),
        password=creds.get("password"),
        user_agent=creds.get("user_agent") or "adforge-radar",
    )


def scan_reddit(session, target: RadarTarget, creds: dict) -> int:
    """Scan one subreddit. Returns the number of new threads recorded."""
    from ..brands import get_brand

    brand = get_brand(target.brand)
    keywords = target.keyword_list()
    if not keywords:
        log.info("radar target r/%s has no keywords, skipping", target.target)
        return 0

    reddit = _reddit_client(creds)
    cutoff = utcnow() - dt.timedelta(hours=MAX_AGE_HOURS)
    found = 0

    for submission in reddit.subreddit(target.target).new(limit=PREFILTER_LIMIT):
        created = dt.datetime.fromtimestamp(
            submission.created_utc, tz=dt.timezone.utc
        )
        if created < cutoff:
            break  # .new() is ordered, so everything after this is older

        blob = f"{submission.title}\n{submission.selftext or ''}"
        hits = _matches(blob, keywords)
        if not hits:
            continue

        ext = f"t3_{submission.id}"
        if session.query(RadarThread.id).filter_by(
            source="reddit", external_id=ext
        ).first():
            continue

        verdict = _score_thread(
            target.brand, brand.about, submission.title, submission.selftext or ""
        )
        relevance = float(verdict.get("relevance") or 0.0)
        if not verdict.get("useful_without_product", True):
            # If a human expert could not add value without pitching, a reply
            # would be an advertisement regardless of how it is phrased.
            relevance = min(relevance, 0.35)

        session.add(
            RadarThread(
                brand=target.brand,
                source="reddit",
                external_id=ext,
                where=f"r/{target.target}",
                author=str(submission.author) if submission.author else "[deleted]",
                title=submission.title[:590],
                excerpt=(submission.selftext or "")[:2000],
                url=f"https://reddit.com{submission.permalink}",
                posted_at=created,
                relevance=relevance,
                score_reason=str(verdict.get("reason", ""))[:900],
                matched_keywords=",".join(hits)[:390],
                promo_allowed=bool(target.promo_allowed),
                rules_note=target.rules_note or "",
            )
        )
        found += 1

    return found


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------


def scan_discord(session, target: RadarTarget, creds: dict) -> int:
    """Scan one Discord channel via the REST API.

    Uses plain REST rather than a gateway connection: reading recent messages
    in a channel the bot is already in needs no privileged intent and no
    persistent socket. `target` is "guild_id/channel_id".
    """
    import httpx

    from ..brands import get_brand

    brand = get_brand(target.brand)
    keywords = target.keyword_list()
    if not keywords:
        return 0

    token = creds.get("bot_token")
    if not token:
        log.warning("discord radar needs a bot_token")
        return 0

    try:
        _, channel_id = target.target.split("/", 1)
    except ValueError:
        channel_id = target.target

    cutoff = utcnow() - dt.timedelta(hours=MAX_AGE_HOURS)
    found = 0
    with httpx.Client(timeout=60) as client:
        r = client.get(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers={"Authorization": f"Bot {token}"},
            params={"limit": min(100, PREFILTER_LIMIT)},
        )
        if r.status_code >= 400:
            log.warning("discord scan of %s failed: %s", target.target, r.text[:200])
            return 0
        messages = r.json()

    for msg in messages:
        content = msg.get("content") or ""
        if len(content) < 40:
            continue  # one-liners are not threads worth answering
        ts = msg.get("timestamp", "")
        try:
            created = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if created < cutoff:
            continue

        hits = _matches(content, keywords)
        if not hits:
            continue

        ext = f"dm_{msg['id']}"
        if session.query(RadarThread.id).filter_by(
            source="discord", external_id=ext
        ).first():
            continue

        verdict = _score_thread(target.brand, brand.about, content[:200], content)
        relevance = float(verdict.get("relevance") or 0.0)
        if not verdict.get("useful_without_product", True):
            relevance = min(relevance, 0.35)

        author = (msg.get("author") or {}).get("username", "?")
        session.add(
            RadarThread(
                brand=target.brand,
                source="discord",
                external_id=ext,
                where=target.target,
                author=author,
                title=content[:180],
                excerpt=content[:2000],
                url=f"https://discord.com/channels/{target.target}/{msg['id']}",
                posted_at=created,
                relevance=relevance,
                score_reason=str(verdict.get("reason", ""))[:900],
                matched_keywords=",".join(hits)[:390],
                promo_allowed=bool(target.promo_allowed),
                rules_note=target.rules_note or "",
            )
        )
        found += 1

    return found


def scan_all() -> int:
    """Scan every enabled target. Returns total new threads found."""
    total = 0
    with session_scope() as session:
        targets = session.query(RadarTarget).filter(RadarTarget.enabled.is_(True)).all()
        for target in targets:
            account = (
                session.query(Account)
                .filter(Account.brand == target.brand,
                        Account.platform == target.source)
                .first()
            )
            if account is None:
                log.info("no %s account for %s, skipping radar target %s",
                         target.source, target.brand, target.target)
                continue
            try:
                if target.source == "reddit":
                    total += scan_reddit(session, target, account.creds())
                elif target.source == "discord":
                    total += scan_discord(session, target, account.creds())
            except Exception as e:  # noqa: BLE001 - one bad target must not
                # abort the whole scan
                log.exception("radar scan of %s failed: %s", target.target, e)
    if total:
        log.info("radar found %d new threads", total)
    return total
