"""
Turning a Schedule into concrete post times, and generating what fills them.

Two separate concerns, deliberately:

  plan_slots()  - when should posts go out (pure, testable, no side effects)
  fill_slot()   - generate the copy and media for one slot (slow, GPU-bound)

Keeping them apart means the UI can show tomorrow's plan instantly without
waiting on a 70B model.
"""

from __future__ import annotations

import pathlib

import datetime as dt
import json
import logging
import random
import uuid
import zoneinfo

from ..brands import get_brand
from ..config import settings
from ..db import Post, PostMode, PostStatus, Schedule, utcnow
from ..llm.copywriter import (extract_hashtags, media_prompt, write_post,
                              write_thread)
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
    #
    # Video is the exception, and it is an explicit one. The critic scores the
    # written copy while the script model is writing on-screen captions, so
    # video lands a point or two low for a reason that has nothing to do with
    # the finished ad. Admitting it between the video floor and the real bar
    # is recorded in the notes rather than left silent - a bar that was
    # lowered invisibly is worse than one that was never there.
    video = ps.key in VIDEO_FIRST and bool(post.video_path)
    admitted = (video and not draft.passed
                and draft.score >= settings.video_quality_floor)

    if admitted:
        post.critic_notes = (
            f"admitted on the video floor (score {draft.score:.1f}, floor "
            f"{settings.video_quality_floor:.1f}, text bar 7.0); "
            + post.critic_notes
        )
    elif not draft.passed:
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

    # X threads. The adapter has always chained extra_parts into a real thread,
    # but nothing ever produced any, so a mechanism that needs 600 characters
    # got 280 and was cut off at the interesting part.
    if ps.key == "x" and draft.passed:
        try:
            head, parts = write_thread(brand, ps, pillar, draft.text, draft.angle)
            if parts:
                # The body becomes the thread's OWN first tweet. Keeping the
                # original opener would leave the continuations following a
                # tweet nobody sees.
                post.body = head
                post.hashtags = ",".join(extract_hashtags(head))
                post.extra_parts = json.dumps(parts)
        except Exception as e:  # noqa: BLE001 - a thread is a bonus, not the post
            log.warning("thread generation failed for %s: %s", sched_brand, e)

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


def _short_data(brand, script) -> dict:
    """Map a generated Script onto the statement template's fields.

    Derived rather than asked for separately: the copy model already produced
    a title and per-scene captions, and a second LLM call to reshape them is
    another thing that can fail between a good script and a rendered video.
    The claims come from the brand's proof points, so nothing on screen is
    invented for the sake of the layout.
    """
    caps = [s.caption.strip() for s in script.scenes if s.caption.strip()]

    def split3(text: str) -> list[str]:
        """Break one line into three short ones on word boundaries."""
        words, lines, cur = text.split(), [], ""
        target = max(1, len(text) // 3)
        for w in words:
            if cur and len(cur) + 1 + len(w) > target and len(lines) < 2:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        while len(lines) < 3:
            lines.append("")
        return lines[:3]

    hook = split3(caps[0] if caps else script.title)
    kick = caps[-1] if len(caps) > 1 else script.title
    kwords = kick.split()
    mid = max(1, len(kwords) // 2)
    return {
        "eyebrow": brand.name,
        "hook_a": hook[0], "hook_b": hook[1], "hook_dim": hook[2],
        "points": caps[1:4] if len(caps) > 2 else caps[:3],
        "kicker_a": " ".join(kwords[:mid]), "kicker_b": " ".join(kwords[mid:]),
        # short_claims, not truncated proof points: chopping one at 34
        # characters produced "Per-second billing with no", which inverts it.
        "claims": (brand.short_claims or brand.proof_points)[:3],
        "tagline": brand.tagline,
    }


def _render_cinematic_ad(brand, script, slug: str, ps):
    """Four graded AI shots with the typography composited over them.

    The script model already writes a `visual` per scene - those ARE shot
    descriptions, so no second generation step is needed to get them.
    """
    import json

    from ..config import ASSETS
    from ..media import cinematic, motion
    from ..media.shorts import brand_theme

    w, h = ps.video_size
    fps = 24
    scenes = [s for s in script.scenes if (s.visual or "").strip()][:4]
    if len(scenes) < 2:
        raise cinematic.CinematicError(
            f"only {len(scenes)} scene(s) carry a visual prompt; the cinematic "
            f"format needs at least two shots")

    ad = cinematic.Ad(slug=slug, width=w, height=h, fps=fps)
    ad.shots = [cinematic.Shot(s.visual.strip()) for s in scenes]
    # Footage is generated near the resolution Wan was trained at and upscaled
    # on assembly; rendering straight to 1080x1920 is slower and no sharper.
    for i, shot in enumerate(ad.shots):
        cinematic.render_shot(shot, i, slug, 720, 1280, fps)

    total = sum(s.seconds for s in ad.shots)
    # One statement per shot, changing on the cut. Text that outlives its shot
    # is what makes a cut sequence read as a slideshow.
    cards, at = [], 0.3
    for i, sc in enumerate(scenes):
        cap = (sc.caption or "").strip()
        words = cap.split()
        mid = max(1, len(words) // 2)
        cards.append([at, at + ad.shots[i].seconds - 0.3,
                      brand.name if i == 0 else "",
                      " ".join(words[:mid]), " ".join(words[mid:])])
        at += ad.shots[i].seconds
    data = {"brand": brand_theme(brand), "seconds": total, "cards": cards,
            "end_at": max(0.0, total - 2.0), "tagline": brand.tagline}

    html = (pathlib.Path(__file__).parent.parent / "media" / "shorts"
            / "overlay.html").read_text().replace(
        "<!--DATA-->", f"<script>window.__DATA = {json.dumps(data)};</script>")
    odir = ASSETS / f".{slug}_overlay"
    motion.render_frames(html, odir, seconds=total, fps=fps,
                         width=w, height=h, transparent=True)

    out = ASSETS / f"{slug}.mp4"
    try:
        cinematic.assemble(ad, odir, out)
        motion._add_audio(out, total)
    finally:
        for f in odir.glob("*.png"):
            f.unlink()
        odir.rmdir()
        # The plates and untrimmed clips are regenerable and large; one ad
        # leaves about 1.5GB of them behind, which filled the disk once.
        for s in ad.shots:
            for leftover in (s.plate, s.clip):
                if leftover:
                    leftover.unlink(missing_ok=True)
    return out


def _render_motion_short(brand, script, slug: str, ps):
    from ..media.shorts import render_short
    from ..config import ASSETS

    if settings.short_template == "cinematic":
        return _render_cinematic_ad(brand, script, slug, ps)
    return render_short(brand, settings.short_template,
                        _short_data(brand, script),
                        ASSETS / f"{slug}.mp4", size=ps.video_size)


def _attach_media(brand, post: Post, ps, draft, rng, slug: str) -> None:
    from ..media.images import generate_image

    if ps.key in VIDEO_FIRST and settings.enable_video and ps.video_size:
        from ..media.script import caption_for, write_script

        script = write_script(brand, ps, brand.pillar(post.pillar), draft.angle)
        if settings.video_engine == "motion":
            video = _render_motion_short(brand, script, slug, ps)
        else:
            from ..media.video import render_script

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
        #
        # Only meaningful for the Wan path. Scene.animated is never set on the
        # motion path - there are no Wan scenes to animate - so this check
        # reported "NO REAL MOTION in 8/8 scenes" on a correct motion-graphics
        # render and pinned it to MANUAL. A warning that fires on healthy
        # output is worse than none: it trains you to ignore the real one.
        stills = ([s for s in script.scenes if not s.animated]
                  if settings.video_engine != "motion" else [])
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
