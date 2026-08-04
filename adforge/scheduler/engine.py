"""
The scheduler loop.

Three jobs on independent intervals:

  plan     - hourly, looks ahead `lookahead_hours` and creates any missing
             posts for the slots in that window
  promote  - every minute, moves REVIEW -> APPROVED once the review window has
             expired (AUTO accounts only), then publishes anything APPROVED
             whose scheduled time has arrived
  radar    - the Reddit/Discord scan, on its own interval

Generation is deliberately decoupled from publication: a 70B writer plus a Wan
render can take minutes, and doing that work inside the publish tick would make
posts fire late. Content is prepared ahead of its slot and simply sent when the
time comes.
"""

from __future__ import annotations

import datetime as dt
import logging
import random

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import and_

from ..config import settings
from ..db import (
    Account,
    Post,
    PostMode,
    PostStatus,
    Schedule,
    session_scope,
    utcnow,
)
from ..platforms.base import PublishError
from ..platforms.registry import publish_post
from .planner import fill_slot, plan_slots

log = logging.getLogger("adforge.sched")

LOOKAHEAD_HOURS = 26  # a bit over a day, so tomorrow's early slots exist tonight
MAX_PUBLISH_ATTEMPTS = 3


def _account_for(session, brand: str, platform: str) -> Account | None:
    return (
        session.query(Account)
        .filter(Account.brand == brand, Account.platform == platform)
        .filter(Account.enabled.is_(True))
        .first()
    )


def plan_ahead(rng: random.Random | None = None) -> int:
    """Create posts for any unfilled slot inside the lookahead window."""
    rng = rng or random
    created = 0
    now = utcnow()
    horizon = now + dt.timedelta(hours=LOOKAHEAD_HOURS)

    with session_scope() as session:
        for sched in session.query(Schedule).filter(Schedule.enabled.is_(True)).all():
            account = _account_for(session, sched.brand, sched.platform)
            if account is None:
                log.debug("%s/%s has no enabled account, skipping",
                          sched.brand, sched.platform)
                continue

            for day in (now.date(), now.date() + dt.timedelta(days=1)):
                for when in plan_slots(sched, day, rng):
                    if not (now < when <= horizon):
                        continue
                    # One post per slot. A 30-minute window absorbs jitter
                    # differences between planning runs without duplicating.
                    exists = (
                        session.query(Post.id)
                        .filter(
                            Post.brand == sched.brand,
                            Post.platform == sched.platform,
                            and_(
                                Post.scheduled_for >= when - dt.timedelta(minutes=30),
                                Post.scheduled_for <= when + dt.timedelta(minutes=30),
                            ),
                        )
                        .first()
                    )
                    if exists:
                        continue

                    if _published_today(session, sched) >= settings.daily_post_cap:
                        log.info("%s/%s hit the daily cap of %d",
                                 sched.brand, sched.platform, settings.daily_post_cap)
                        break

                    try:
                        post = fill_slot(session, sched, when, account, rng)
                        created += 1
                        log.info(
                            "planned %s/%s for %s (score %.1f, mode %s)",
                            post.brand, post.platform,
                            when.isoformat(timespec="minutes"),
                            post.quality_score, post.mode.value,
                        )
                    except Exception as e:  # noqa: BLE001 - one bad slot must
                        # not stop the whole planning run
                        log.exception("failed to fill %s/%s slot at %s: %s",
                                      sched.brand, sched.platform, when, e)
    return created


def _published_today(session, sched: Schedule) -> int:
    since = utcnow() - dt.timedelta(hours=24)
    return (
        session.query(Post.id)
        .filter(
            Post.brand == sched.brand,
            Post.platform == sched.platform,
            Post.status == PostStatus.PUBLISHED,
            Post.published_at >= since,
        )
        .count()
    )


def promote_and_publish() -> tuple[int, int]:
    """Expire review windows, then publish anything due."""
    promoted = published = 0
    now = utcnow()

    with session_scope() as session:
        # REVIEW -> APPROVED, auto accounts whose window has run out.
        for post in (
            session.query(Post)
            .filter(Post.status == PostStatus.REVIEW, Post.mode == PostMode.AUTO)
            .filter(Post.review_until.isnot(None), Post.review_until <= now)
            .all()
        ):
            post.status = PostStatus.APPROVED
            promoted += 1
            log.info("review window expired, approved post %s", post.id)

        # APPROVED and due -> publish.
        due = (
            session.query(Post)
            .filter(Post.status.in_([PostStatus.APPROVED, PostStatus.FAILED]))
            .filter(Post.scheduled_for.isnot(None), Post.scheduled_for <= now)
            .filter(Post.attempts < MAX_PUBLISH_ATTEMPTS)
            .order_by(Post.scheduled_for)
            .all()
        )
        for post in due:
            account = session.get(Account, post.account_id) if post.account_id else None
            if account is None or not account.enabled:
                post.error = "no enabled account for this destination"
                post.status = PostStatus.REJECTED
                continue
            try:
                publish_post(post, account)
                published += 1
            except PublishError as e:
                # publish_post already set status/error. A retryable failure
                # stays FAILED and is picked up again next tick until the
                # attempt cap.
                log.warning("post %s failed: %s", post.id, e)
            except Exception as e:  # noqa: BLE001
                post.status = PostStatus.FAILED
                post.error = str(e)[:900]
                log.exception("unexpected error publishing post %s", post.id)

    return promoted, published


def run_radar() -> int:
    if not settings.radar_enabled:
        return 0
    try:
        from ..radar.scan import scan_all

        return scan_all()
    except Exception as e:  # noqa: BLE001
        log.exception("radar scan failed: %s", e)
        return 0


def build_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(plan_ahead, "interval", hours=1, id="plan",
                  next_run_time=utcnow() + dt.timedelta(seconds=20),
                  max_instances=1, coalesce=True)
    sched.add_job(promote_and_publish, "interval", minutes=1, id="publish",
                  max_instances=1, coalesce=True)
    sched.add_job(run_radar, "interval", minutes=settings.radar_interval_minutes,
                  id="radar", max_instances=1, coalesce=True)
    return sched
