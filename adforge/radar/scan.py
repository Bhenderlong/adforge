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
# Sitewide search casts a wider net: it is already keyword-filtered server-side
# and most results are still scored away.
SEARCH_LIMIT = 120

# Target values meaning "everything I can reach" rather than one place.
SITEWIDE = {"all", "*", "sitewide"}


def is_sitewide(target: str) -> bool:
    return target.strip().lower().removeprefix("r/") in SITEWIDE


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


def _search_query(keywords: list[str]) -> str:
    """Reddit search syntax: quoted phrases OR'd together.

    Quoting matters - an unquoted multi-word keyword is parsed as separate
    terms and returns their union, which for "gpu cloud" is most of the site.
    """
    return " OR ".join(f'"{k}"' if " " in k else k for k in keywords)


def _submissions(reddit, target: RadarTarget, keywords: list[str]):
    """Candidate submissions, sitewide or from one subreddit.

    Sitewide uses Reddit's SEARCH endpoint, not the new-posts firehose.
    r/all/new is thousands of posts an hour across every subreddit, so a local
    keyword filter over the newest 60 finds essentially nothing. Search matches
    server-side across the whole site, which is what scanning all of Reddit
    actually requires.
    """
    if is_sitewide(target.target):
        return reddit.subreddit("all").search(
            _search_query(keywords), sort="new", time_filter="week",
            limit=SEARCH_LIMIT,
        )
    return reddit.subreddit(target.target).new(limit=PREFILTER_LIMIT)


def scan_reddit(session, target: RadarTarget, creds: dict) -> int:
    """Scan one subreddit, or the whole site. Returns new threads recorded."""
    from ..brands import get_brand

    brand = get_brand(target.brand)
    keywords = target.keyword_list()
    if not keywords:
        log.info("radar target r/%s has no keywords, skipping", target.target)
        return 0

    sitewide = is_sitewide(target.target)
    reddit = _reddit_client(creds)
    cutoff = utcnow() - dt.timedelta(hours=MAX_AGE_HOURS)
    found = 0

    for submission in _submissions(reddit, target, keywords):
        created = dt.datetime.fromtimestamp(
            submission.created_utc, tz=dt.timezone.utc
        )
        if created < cutoff:
            if sitewide:
                continue  # search results are not strictly time-ordered
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

        # Release the transaction before scoring. The dedupe query above
        # autoflushes, so with a pending add the session holds a SQLite
        # RESERVED lock - and _score_thread then blocks on the GPU lock, which
        # a concurrent Wan render can hold for minutes. The publish tick would
        # fail with "database is locked" for that whole time.
        session.commit()

        verdict = _score_thread(
            target.brand, brand.about, submission.title, submission.selftext or ""
        )
        relevance = float(verdict.get("relevance") or 0.0)
        if not verdict.get("useful_without_product", True):
            # If a human expert could not add value without pitching, a reply
            # would be an advertisement regardless of how it is phrased.
            relevance = min(relevance, 0.35)

        # The subreddit the thread actually lives in, not the target - for a
        # sitewide scan those differ, and "r/all" is not somewhere you can post.
        actual_sub = str(submission.subreddit.display_name)

        # A sitewide hit is in a community whose self-promotion rules nobody has
        # read. promo_allowed is snapshotted per thread and ANDed with the
        # target's flag at reply time, so forcing it false here means such a
        # reply must stand on its own merits with no link until the user
        # explicitly allowlists that subreddit.
        promo = bool(target.promo_allowed) and not sitewide
        note = target.rules_note or ""
        if sitewide:
            note = (f"sitewide hit; r/{actual_sub} rules not reviewed - "
                    "links suppressed")

        session.add(
            RadarThread(
                brand=target.brand,
                source="reddit",
                external_id=ext,
                where=f"r/{actual_sub}",
                author=str(submission.author) if submission.author else "[deleted]",
                title=submission.title[:590],
                excerpt=(submission.selftext or "")[:2000],
                url=f"https://reddit.com{submission.permalink}",
                posted_at=created,
                relevance=relevance,
                score_reason=str(verdict.get("reason", ""))[:900],
                matched_keywords=",".join(hits)[:390],
                promo_allowed=promo,
                rules_note=note[:390],
            )
        )
        session.commit()
        found += 1

    return found


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------


def discord_channels(bot_token: str) -> list[dict]:
    """Every readable text channel, across every server this bot is in.

    There is no way to search Discord globally. A bot sees only servers it has
    been invited to; there is no public directory and no cross-server search
    endpoint. The only thing that could scan "all of Discord" is a self-bot
    driving a user account, which Discord's ToS prohibits outright and which
    gets the account terminated. So a `*` target expands to this instead: what
    the bot can legitimately reach.
    """
    import httpx

    out: list[dict] = []
    headers = {"Authorization": f"Bot {bot_token}"}
    with httpx.Client(timeout=60) as client:
        r = client.get("https://discord.com/api/v10/users/@me/guilds",
                       headers=headers)
        if r.status_code >= 400:
            log.warning("discord guild list failed [%s]: %s",
                        r.status_code, r.text[:200])
            return out
        for guild in r.json():
            gid, gname = guild["id"], guild.get("name", guild["id"])
            cr = client.get(
                f"https://discord.com/api/v10/guilds/{gid}/channels",
                headers=headers,
            )
            if cr.status_code >= 400:
                log.info("cannot list channels in %s: %s", gname, cr.status_code)
                continue
            for ch in cr.json():
                # 0 text, 5 announcement, 15 forum. Voice and categories have
                # nothing to read.
                if ch.get("type") in (0, 5, 15):
                    out.append({
                        "guild_id": gid, "guild_name": gname,
                        "channel_id": ch["id"],
                        "channel_name": ch.get("name", ""),
                        "target": f"{gid}/{ch['id']}",
                    })
    return out


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

        session.commit()  # same reason as the reddit path above
        verdict = _score_thread(target.brand, brand.about, content[:200], content)
        relevance = float(verdict.get("relevance") or 0.0)
        if not verdict.get("useful_without_product", True):
            relevance = min(relevance, 0.35)

        author = (msg.get("author") or {}).get("username", "?")
        # promo/note were read below but never assigned in this function: a
        # NameError on the first matching message, swallowed by scan_all's
        # except clause. The Discord radar therefore reported a clean scan of
        # zero threads, forever, and had never stored one.
        #
        # Same rule as Reddit: a hit from a wildcard target is in a channel
        # whose rules nobody has read, so it carries no link.
        wildcard = is_sitewide(target.target)
        promo = bool(target.promo_allowed) and not wildcard
        note = target.rules_note or ""
        if wildcard:
            note = ("wildcard hit; this server's rules were not reviewed - "
                    "links suppressed")

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
                promo_allowed=promo,
                rules_note=note[:390],
            )
        )
        session.commit()
        found += 1

    return found


# Brand/source pairs already reported as unconfigured, so the warning is
# logged once rather than on every 30-minute pass.
_WARNED_MISSING: set = set()


def scan_all() -> int:
    """Scan every enabled target. Returns total new threads found."""
    total = 0
    _WARNED_MISSING.clear()
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

            # Check the credentials exist BEFORE scanning. Without this the
            # scan raised KeyError('client_id') or "Illegal header value
            # b'Bot '" on every pass - once per target, every 30 minutes -
            # which says nothing about the actual problem and buries real
            # errors in noise. An unconfigured target is a normal state while
            # you are still setting up, not an error.
            creds = account.creds()
            need = {"reddit": ("client_id", "client_secret", "username", "password"),
                    "discord": ("bot_token",)}.get(target.source, ())
            missing = [k for k in need if not str(creds.get(k, "")).strip()]
            if missing:
                key = (target.brand, target.source)
                if key not in _WARNED_MISSING:
                    _WARNED_MISSING.add(key)
                    log.warning(
                        "radar: %s/%s has no credentials yet (%s) - skipping its "
                        "targets. Add them on the Accounts page; this is logged "
                        "once per run.",
                        target.brand, target.source, ", ".join(missing),
                    )
                continue
            try:
                if target.source == "reddit":
                    total += scan_reddit(session, target, account.creds())
                elif target.source == "discord":
                    if is_sitewide(target.target):
                        # Fan out over every channel the bot can actually read.
                        chans = discord_channels(account.creds().get("bot_token", ""))
                        log.info("discord '*' expands to %d channel(s)", len(chans))
                        for ch in chans:
                            sub = RadarTarget(
                                brand=target.brand, source="discord",
                                target=ch["target"], keywords=target.keywords,
                                enabled=True, promo_allowed=False,
                                rules_note=f"{ch['guild_name']} #{ch['channel_name']}",
                                min_relevance=target.min_relevance,
                            )
                            total += scan_discord(session, sub, account.creds())
                    else:
                        total += scan_discord(session, target, account.creds())
            except Exception as e:  # noqa: BLE001 - one bad target must not
                # abort the whole scan
                log.exception("radar scan of %s failed: %s", target.target, e)
    if total:
        log.info("radar found %d new threads", total)
    return total
