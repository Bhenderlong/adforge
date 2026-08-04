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


def plan_slots(
    sched: Schedule, day: dt.date, rng: random.Random | None = None
) -> list[dt.datetime]:
    """Local-time posting instants for one schedule on one day.

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
        local = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=zone)
        if sched.jitter_minutes:
            local += dt.timedelta(
                minutes=rng.randint(-sched.jitter_minutes, sched.jitter_minutes)
            )
        out.append(local.astimezone(dt.timezone.utc))
    return sorted(out)


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
) -> Post:
    """Generate one post for a planned slot. Slow: runs the writer and media."""
    rng = rng or random
    brand = get_brand(sched.brand)
    ps = spec(sched.platform)

    keys = sched.pillar_list()
    pillar = (
        brand.pillar(rng.choice(keys)) if keys else brand.pick_pillar(rng)
    )

    recent = recent_bodies(session, sched.brand, sched.platform)
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

    session.add(post)
    session.flush()

    if sched.attach_media:
        try:
            _attach_media(brand, post, ps, draft, rng)
        except Exception as e:  # noqa: BLE001 - a missing image must not lose the copy
            log.warning("media generation failed for post %s: %s", post.id, e)
            post.critic_notes = f"media failed: {str(e)[:200]}; {post.critic_notes}"

    return post


def _headline(brand, body: str, ps) -> str:
    """A title for platforms that require one, derived from the body."""
    first = body.strip().split("\n")[0]
    first = first.rstrip(".!").strip()
    limit = 290 if ps.key == "reddit" else 95
    return first[:limit]


def _attach_media(brand, post: Post, ps, draft, rng) -> None:
    from ..media.images import generate_image

    slug = f"{post.brand}_{post.platform}_{post.id}"

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
        return

    if ps.supports_image:
        prompt = media_prompt(brand, draft.text, draft.angle)
        post.media_prompt = prompt
        post.image_path = str(generate_image(brand, ps, prompt, slug))
