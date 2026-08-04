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
from .planner import fill_slot, plan_slots_detailed, slot_key, tz

log = logging.getLogger("adforge.sched")

LOOKAHEAD_HOURS = 26  # a bit over a day, so tomorrow's early slots exist tonight
MAX_PUBLISH_ATTEMPTS = 3
# A post whose slot passed more than this long ago is not published. See the
# staleness bound in promote_and_publish for why.
MAX_SLOT_AGE_HOURS = 12


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

            # Enumerate LOCAL calendar days. plan_slots builds its times in
            # the configured zone, so feeding it a UTC date skipped the
            # current evening's slots whenever UTC had already rolled over -
            # from 20:00 onward in America/New_York.
            today_local = now.astimezone(tz()).date()
            for day in (today_local, today_local + dt.timedelta(days=1)):
                for nominal, when in plan_slots_detailed(sched, day, rng):
                    if not (now < when <= horizon):
                        continue
                    # One post per NOMINAL slot, matched exactly.
                    #
                    # This used to be a +/-30 minute window around the jittered
                    # time, which failed both ways: pinned times 20 minutes
                    # apart collided so one slot silently never got a post, and
                    # with jitter above 15 two planning runs could place the
                    # same slot more than 30 minutes apart and produce two.
                    key = slot_key(sched, nominal)
                    exists = (
                        session.query(Post.id)
                        .filter(Post.slot_key == key)
                        .first()
                    )
                    if exists:
                        continue

                    if _published_today(session, sched) >= settings.daily_post_cap:
                        log.info("%s/%s hit the daily cap of %d",
                                 sched.brand, sched.platform, settings.daily_post_cap)
                        break

                    try:
                        post = fill_slot(session, sched, when, account,
                                         rng, slot_key=key)
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


# A claim older than this is assumed dead: the process was killed, the pool was
# shut down mid-send, or the adapter hung. Comfortably longer than the slowest
# adapter's rate-limit floor (Reddit, 1800s) plus its upload time.
STALE_CLAIM_SECONDS = 3600


def reap_stale_claims() -> int:
    """Return posts stuck in PUBLISHING to APPROVED so they can be retried.

    `claim_post` commits PUBLISHING before the network call, which is what
    stops a double send - but nothing moved a post back out if the worker died
    in between. Those posts became invisible to every later tick: never
    published, never marked failed, and showing a status the UI could not act
    on. `attempts` was already incremented, so the retry cap still bounds this.
    """
    cutoff = utcnow() - dt.timedelta(seconds=STALE_CLAIM_SECONDS)
    with session_scope() as session:
        stale = (
            session.query(Post)
            .filter(Post.status == PostStatus.PUBLISHING)
            .filter(Post.updated_at <= cutoff)
            .all()
        )
        for post in stale:
            post.status = PostStatus.APPROVED
            post.error = "publisher died mid-send; requeued"
            log.warning("reaped stale claim on post %s", post.id)
        return len(stale)


def promote_and_publish() -> tuple[int, int]:
    """Expire review windows, then publish anything due."""
    promoted = published = 0
    now = utcnow()

    reap_stale_claims()

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

        # Collect ids only. The publish loop below deliberately runs OUTSIDE
        # this transaction: adapters sleep for their rate-limit floor (Reddit
        # is 1800s) and then do network I/O, and holding a write transaction
        # across that meant a crash mid-run rolled back PUBLISHED state for
        # posts that were already live, which then republished on the next tick.
        # STALENESS BOUND. Without it, a backlog that accumulated while the
        # process was down - or while dry run was on, since a dry publish
        # leaves the post APPROVED and past-due - all fires at once on the
        # first tick after going live. That is a burst of old posts with no
        # fresh review and no daily-cap check, since the cap is only consulted
        # when planning.
        stale_before = now - dt.timedelta(hours=MAX_SLOT_AGE_HOURS)

        due_rows = (
            session.query(Post.id, Post.brand, Post.platform, Post.scheduled_for)
            .filter(Post.status.in_([PostStatus.APPROVED, PostStatus.FAILED]))
            .filter(Post.scheduled_for.isnot(None), Post.scheduled_for <= now)
            .filter(Post.attempts < MAX_PUBLISH_ATTEMPTS)
            .order_by(Post.scheduled_for)
            .all()
        )

        # A schedule switched off, or an account flipped to MANUAL, reads to a
        # user as "stop posting". It did not stop posts already APPROVED - and
        # with 26h of lookahead and a 60-minute review window, tomorrow's are
        # normally approved already. Re-check both at publish time.
        live_schedules = {
            (s.brand, s.platform)
            for s in session.query(Schedule).filter(Schedule.enabled.is_(True)).all()
        }
        manual_accounts = {
            (a.brand, a.platform)
            for a in session.query(Account).filter(Account.mode == PostMode.MANUAL).all()
        }

        due_ids = []
        for pid, brand, platform, when in due_rows:
            post = session.get(Post, pid)
            # SQLite hands datetimes back NAIVE while everything written was
            # UTC, so comparing one directly against an aware value raises
            # TypeError. The audit flagged this as latent; this loop is where
            # it became real.
            if when is not None and when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
            if when < stale_before:
                post.status = PostStatus.REJECTED
                post.error = (
                    f"slot was {int((now - when).total_seconds() // 3600)}h old at "
                    f"publish time (limit {MAX_SLOT_AGE_HOURS}h) - regenerate rather "
                    f"than posting something stale"
                )
                log.info("post %s expired: scheduled %s", pid, when)
                continue
            if (brand, platform) not in live_schedules:
                post.status = PostStatus.REVIEW
                post.mode = PostMode.MANUAL
                post.review_until = None
                post.critic_notes = (
                    f"held: the {brand}/{platform} schedule is switched off; "
                    + (post.critic_notes or "")
                )[:2000]
                log.info("post %s held, schedule disabled", pid)
                continue
            if (brand, platform) in manual_accounts and post.mode == PostMode.AUTO:
                post.mode = PostMode.MANUAL
                post.status = PostStatus.REVIEW
                post.review_until = None
                log.info("post %s held, account switched to MANUAL", pid)
                continue
            due_ids.append(pid)

    for post_id in due_ids:
        if publish_claimed(post_id):
            published += 1

    return promoted, published


def claim_post(post_id: int) -> bool:
    """Atomically move a due post to PUBLISHING. True if this caller won it.

    A compare-and-set on status, committed before any network call. Two
    publishers can select the same row - WAL lets both read it - but only one
    UPDATE matches, so only one transmits.
    """
    with session_scope() as session:
        rows = (
            session.query(Post)
            .filter(
                Post.id == post_id,
                Post.status.in_([PostStatus.APPROVED, PostStatus.FAILED]),
                Post.attempts < MAX_PUBLISH_ATTEMPTS,
            )
            .update(
                {Post.status: PostStatus.PUBLISHING,
                 Post.attempts: Post.attempts + 1},
                synchronize_session=False,
            )
        )
        return rows == 1


def publish_claimed(post_id: int) -> bool:
    """Claim, publish, then record the outcome. Returns True if it went out."""
    if not claim_post(post_id):
        log.debug("post %s already claimed by another publisher", post_id)
        return False

    with session_scope() as session:
        post = session.get(Post, post_id)
        account = session.get(Account, post.account_id) if post.account_id else None
        if account is None or not account.enabled:
            post.error = "no enabled account for this destination"
            post.status = PostStatus.REJECTED
            return False

        # A retryable failure must return to APPROVED, not stay PUBLISHING, or
        # the claim would permanently strand it.
        try:
            result = publish_post(post, account)
        except PublishError as e:
            if post.status == PostStatus.PUBLISHING:
                post.status = (
                    PostStatus.APPROVED if e.retryable else PostStatus.REJECTED
                )
            log.warning("post %s failed: %s", post_id, e)
            return False
        except Exception as e:  # noqa: BLE001
            post.status = PostStatus.APPROVED
            post.error = str(e)[:900]
            log.exception("unexpected error publishing post %s", post_id)
            return False

        if result.dry_run:
            # Dry runs used to leave the post APPROVED and past-due, so it was
            # re-selected every minute until it burned the retry cap - and any
            # backlog transmitted in one burst the moment the user went live.
            post.status = PostStatus.REJECTED
            post.error = "dry run - not transmitted"
        return not result.dry_run


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
