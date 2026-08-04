"""
Turning a Schedule into concrete post times, and generating what fills them.

Two separate concerns, deliberately:

  plan_slots()  - when should posts go out (pure, testable, no side effects)
  fill_slot()   - generate the copy and media for one slot (slow, GPU-bound)

Keeping them apart means the UI can show tomorrow's plan instantly without
waiting on a 70B model.
"""

from __future__ import annotations

import datetime as dt
import logging
import random
import uuid
import zoneinfo

from ..brands import get_brand
from ..config import settings
from ..db import Post, PostMode, PostStatus, Schedule, utcnow
from ..llm.copywriter import media_prompt, write_post
from ..platforms.spec import VIDEO_FIRST, spec

log = logging.getLogger("adforge.plan")


def tz() -> zoneinfo.ZoneInfo:
    try:
        return zoneinfo.ZoneInfo(settings.timezone)
    except Exception:  # noqa: BLE001
        log.warning("unknown timezone %r, using UTC", settings.timezone)
        return zoneinfo.ZoneInfo("UTC")


def _parse_hhmm(s: str) -> tuple[int, int] | None:
    try:
        h, m = s.strip().split(":")
        h, m = int(h), int(m)
    except (ValueError, AttributeError):
        return None
    return (h, m) if 0 <= h < 24 and 0 <= m < 60 else None


def plan_slots_detailed(
    sched: Schedule, day: dt.date, rng: random.Random | None = None
) -> list[tuple[dt.datetime, dt.datetime]]:
    """(nominal, jittered) instants for one schedule on one day.

    Returns timezone-aware datetimes in UTC, ready to store.
    """
    rng = rng or random
    if not sched.enabled or day.weekday() not in sched.day_list():
        return []

    n = max(0, min(int(sched.posts_per_day or 0), settings.daily_post_cap))
    if n == 0:
        return []

    pinned = [t for t in (_parse_hhmm(x) for x in sched.time_list()) if t]
    times: list[tuple[int, int]] = pinned[:n]

    # Spread any remaining posts across the window rather than stacking them on
    # the last pinned time.
    if len(times) < n:
        start = _parse_hhmm(sched.window_start) or (9, 0)
        end = _parse_hhmm(sched.window_end) or (17, 0)
        s_min, e_min = start[0] * 60 + start[1], end[0] * 60 + end[1]
        if e_min <= s_min:
            e_min = s_min + 60
        remaining = n - len(times)
        span = e_min - s_min
        for i in range(remaining):
            # Midpoint of each equal sub-interval, so posts do not cluster at
            # the window edges.
            offset = s_min + int(span * (i + 0.5) / remaining)
            times.append((offset // 60, offset % 60))

    zone = tz()
    out = []
    for h, m in sorted(set(times))[:n]:
        nominal = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=zone)
        actual = nominal
        if sched.jitter_minutes:
            actual += dt.timedelta(
                minutes=rng.randint(-sched.jitter_minutes, sched.jitter_minutes)
            )
        out.append((nominal.astimezone(dt.timezone.utc),
                    actual.astimezone(dt.timezone.utc)))
    return sorted(out)


def slot_key(sched: Schedule, nominal: dt.datetime) -> str:
    """Stable identity for one scheduled slot, independent of jitter."""
    return (f"{sched.brand}/{sched.platform}/"
            f"{nominal.astimezone(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M')}")


def plan_slots(
    sched: Schedule, day: dt.date, rng: random.Random | None = None
) -> list[dt.datetime]:
    """Just the times posts go out, for previews and tests."""
    return [actual for _, actual in plan_slots_detailed(sched, day, rng)]


def recent_bodies(session, brand: str, platform: str, limit: int = 12) -> list[str]:
    """Recent copy for this account, so the writer does not repeat itself."""
    rows = (
        session.query(Post.body)
        .filter(Post.brand == brand, Post.platform == platform)
        .filter(Post.status.in_([PostStatus.PUBLISHED, PostStatus.APPROVED,
                                PostStatus.REVIEW]))
        .order_by(Post.id.desc())
        .limit(limit)
        .all()
    )
    return [r[0] for r in rows if r[0]]


def fill_slot(
    session,
    sched: Schedule,
    when: dt.datetime,
    account=None,
    rng: random.Random | None = None,
    slot_key: str = "",
) -> Post:
    """Generate one post for a planned slot. Slow: runs the writer and media."""
    rng = rng or random
    brand = get_brand(sched.brand)
    ps = spec(sched.platform)

    keys = sched.pillar_list()
    pillar = (
        brand.pillar(rng.choice(keys)) if keys else brand.pick_pillar(rng)
    )

    # Read what we need, then CLOSE the transaction before generating.
    #
    # This used to run the writer and the whole media pipeline inside the
    # caller's open session. The first flush starts a SQLite write
    # transaction, so a planning run with video held the database locked for
    # the length of the render - up to comfy_timeout, 30 minutes - and every
    # other writer (approving a post, saving an account, the radar scan)
    # blocked for busy_timeout and then returned a 500. WAL also could not
    # checkpoint for the duration.
    recent = recent_bodies(session, sched.brand, sched.platform)
    sched_brand, sched_platform = sched.brand, sched.platform
    attach_media = bool(sched.attach_media)
    account_id = account.id if account is not None else None

    # COMMIT, not rollback. This used to be `session.rollback()` with the
    # comment "nothing is pending yet" - true on the first slot only. The
    # caller loops over slots on one session, so from the second slot onward
    # the session holds the previous slot's flushed INSERT, and rolling back
    # discarded it. A planning run reported creating five posts and persisted
    # exactly one: the last. Near-term slots were the ones lost, and a run
    # spanning two brands wiped the first brand's posts too.
    #
    # Committing releases the read transaction just as well, and has the
    # better property that each post becomes durable as soon as it is made
    # rather than riding on the whole run finishing.
    session.commit()

    draft = write_post(brand, ps, pillar, recent=recent, rng=rng)

    mode = account.mode if account is not None else PostMode.AUTO
    review_minutes = settings.review_window_minutes

    post = Post(
        brand=sched.brand,
        platform=sched.platform,
        account_id=account.id if account is not None else None,
        pillar=pillar.key,
        status=PostStatus.REVIEW,
        mode=mode,
        body=draft.text,
        hashtags=",".join(draft.hashtags),
        link=brand.url,
        scheduled_for=when,
        quality_score=draft.score,
        critic_notes="; ".join(draft.problems[:6]),
        generation_meta=f'{{"angle": {draft.angle[:300]!r}, "attempts": {draft.attempts}}}',
        slot_key=slot_key,
    )

    # A post that never cleared the quality bar always waits for a human,
    # whatever the account's mode is.
    if not draft.passed:
        post.mode = PostMode.MANUAL
        post.critic_notes = (
            f"below quality bar (score {draft.score:.1f}); "
            + post.critic_notes
        )

    # The review window runs from generation, not from the scheduled time, so
    # a post generated well in advance is reviewable for its whole lead time
    # rather than only in the hour before it goes out. Publication still waits
    # for `scheduled_for`; review_until only controls the REVIEW -> APPROVED
    # promotion.
    if post.mode == PostMode.AUTO:
        post.review_until = utcnow() + dt.timedelta(minutes=max(0, review_minutes))

    if ps.key in ("reddit", "youtube"):
        post.title = _headline(brand, draft.text, ps)

    # Media is generated BEFORE the insert, for the same reason: rendering a
    # video with the row already flushed would hold the write lock for the
    # whole render. The asset slug therefore cannot use post.id, which does not
    # exist yet - a random suffix serves the same purpose.
    if attach_media:
        slug = f"{sched_brand}_{sched_platform}_{uuid.uuid4().hex[:10]}"
        try:
            _attach_media(brand, post, ps, draft, rng, slug)
        except Exception as e:  # noqa: BLE001 - a missing image must not lose the copy
            log.warning("media generation failed for %s: %s", slug, e)
            post.critic_notes = f"media failed: {str(e)[:200]}; {post.critic_notes}"

    session.add(post)
    # Commit immediately so this post survives a later failure in the run, and
    # so the write lock is held only for the insert rather than until the
    # caller finishes every remaining slot.
    session.commit()
    return post


def _headline(brand, body: str, ps) -> str:
    """A title for platforms that require one, derived from the body."""
    first = body.strip().split("\n")[0]
    first = first.rstrip(".!").strip()
    limit = 290 if ps.key == "reddit" else 95
    return first[:limit]


def _attach_media(brand, post: Post, ps, draft, rng, slug: str) -> None:
    from ..media.images import generate_image

    if ps.key in VIDEO_FIRST and settings.enable_video and ps.video_size:
        from ..media.script import caption_for, write_script
        from ..media.video import render_script

        script = write_script(brand, ps, brand.pillar(post.pillar), draft.angle)
        video = render_script(brand, script, slug, ps.video_size)
        post.video_path = str(video)
        post.title = script.title[:95]
        post.body = caption_for(brand, ps, script)
        post.hashtags = ",".join(
            __import__("re").findall(r"(?<!\w)#(\w+)", post.body)
        )

        # Surface a degraded render. The Ken Burns fallback looks fine in
        # isolation, so a broken Wan config would quietly turn every short into
        # a slideshow. Force a human look rather than shipping that unnoticed.
        stills = [s for s in script.scenes if not s.animated]
        if stills:
            reason = next((s.fallback_reason for s in stills if s.fallback_reason), "")
            post.mode = PostMode.MANUAL
            post.critic_notes = (
                f"NO REAL MOTION in {len(stills)}/{len(script.scenes)} scenes - "
                f"rendered as stills with a slow zoom ({reason}); "
                + (post.critic_notes or "")
            )[:2000]
        return

    if ps.supports_image:
        prompt = media_prompt(brand, draft.text, draft.angle)
        post.media_prompt = prompt
        post.image_path = str(generate_image(brand, ps, prompt, slug))
