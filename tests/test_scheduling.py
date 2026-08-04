"""
Scheduling and job-tracking regressions.

Every case here is a defect an audit found in working code, reproduced before
it was fixed. All offline - no model, no GPU, no network.
"""

from __future__ import annotations

import datetime as dt
import random

import pytest

from adforge.db import Schedule
from adforge.scheduler.planner import plan_slots, plan_slots_detailed, slot_key, tz

DAY = dt.date(2026, 8, 5)  # a Wednesday


def _sched(**kw):
    base = dict(
        brand="inferix", platform="x", enabled=True, posts_per_day=3,
        days_of_week="0,1,2,3,4,5,6", times="", window_start="09:00",
        window_end="17:00", jitter_minutes=0,
    )
    base.update(kw)
    return Schedule(**base)


# --- slot identity ----------------------------------------------------------
# Dedupe used to compare scheduled_for within +/-30 minutes, which failed both
# ways: it swallowed legitimate slots closer than that, and missed duplicates
# once jitter let two runs place one slot more than 30 minutes apart.

def test_slots_20_minutes_apart_get_distinct_keys():
    """Pinned 09:00/09:20/09:40 used to collapse to one post."""
    s = _sched(times="09:00,09:20,09:40", posts_per_day=3)
    keys = [slot_key(s, n) for n, _ in plan_slots_detailed(s, DAY, random.Random(1))]
    assert len(set(keys)) == 3, keys


def test_slots_exactly_30_minutes_apart_get_distinct_keys():
    """The old window was inclusive, so exactly-30 collided too."""
    s = _sched(times="09:00,09:30", posts_per_day=2)
    keys = [slot_key(s, n) for n, _ in plan_slots_detailed(s, DAY, random.Random(1))]
    assert len(set(keys)) == 2, keys


@pytest.mark.parametrize("jitter", [5, 20, 45, 120])
def test_jitter_never_changes_a_slots_identity(jitter):
    """Two planning runs re-roll jitter; both must map to the same slot.

    With the old time-window dedupe, jitter above 15 could place the same
    nominal slot >30 min apart across runs and produce two posts for it.
    """
    s = _sched(times="12:00", posts_per_day=1, jitter_minutes=jitter)
    a = plan_slots_detailed(s, DAY, random.Random(1))[0]
    b = plan_slots_detailed(s, DAY, random.Random(99))[0]
    assert slot_key(s, a[0]) == slot_key(s, b[0])


def test_slot_key_separates_brands_and_platforms():
    a = _sched(times="12:00", posts_per_day=1)
    b = _sched(times="12:00", posts_per_day=1, brand="vallorix")
    c = _sched(times="12:00", posts_per_day=1, platform="linkedin")
    n = plan_slots_detailed(a, DAY, random.Random(1))[0][0]
    assert len({slot_key(a, n), slot_key(b, n), slot_key(c, n)}) == 3


def test_jitter_actually_moves_the_fire_time():
    """The key must be stable, but the published time must still vary."""
    s = _sched(times="12:00", posts_per_day=1, jitter_minutes=30)
    fires = {plan_slots_detailed(s, DAY, random.Random(i))[0][1] for i in range(12)}
    assert len(fires) > 1


# --- planning basics --------------------------------------------------------

def test_day_not_in_schedule_yields_nothing():
    s = _sched(days_of_week="0,1,2,3,4")  # Mon-Fri
    assert plan_slots(s, dt.date(2026, 8, 8), random.Random(1)) == []  # Saturday


def test_daily_cap_bounds_the_slot_count():
    from adforge.config import settings

    s = _sched(posts_per_day=99)
    assert len(plan_slots(s, DAY, random.Random(1))) <= settings.daily_post_cap


def test_pinned_times_are_honoured_and_remainder_spread():
    s = _sched(times="09:15", posts_per_day=3)
    local = [t.astimezone(tz()).strftime("%H:%M")
             for t in plan_slots(s, DAY, random.Random(1))]
    assert "09:15" in local
    assert len(local) == 3
    assert len(set(local)) == 3  # no stacking on the pinned time


def test_slots_are_returned_in_order():
    s = _sched(times="16:00,09:00,12:00", posts_per_day=3)
    slots = plan_slots(s, DAY, random.Random(1))
    assert slots == sorted(slots)


# --- timestamp rendering ----------------------------------------------------
# The queue printed stored naive UTC as if it were local while the detail page
# converted it, so the same row showed two times four hours apart.

def test_localtime_converts_naive_storage_to_local():
    from adforge.ui.app import _localtime

    stored = dt.datetime(2026, 8, 4, 13, 0)  # naive, but UTC by convention
    expected = stored.replace(tzinfo=dt.timezone.utc).astimezone(tz())
    assert _localtime(stored) == expected.strftime("%d %b %H:%M")


def test_localtime_treats_aware_and_naive_identically():
    from adforge.ui.app import _localtime

    naive = dt.datetime(2026, 8, 4, 13, 0)
    assert _localtime(naive) == _localtime(naive.replace(tzinfo=dt.timezone.utc))


def test_localtime_handles_null():
    from adforge.ui.app import _localtime

    assert _localtime(None) == ""


# --- job tracking -----------------------------------------------------------
# Clearing the panel dropped anything not "running", which included "queued" -
# and the worker then KeyErrored before calling fn, killing the job silently.

def test_clearing_jobs_leaves_queued_work_alone():
    from adforge.ui import app as uiapp

    with uiapp.JOBS_LOCK:
        uiapp.JOBS.clear()
        uiapp.JOBS["q"] = {"id": "q", "name": "queued-job", "state": "queued"}
        uiapp.JOBS["r"] = {"id": "r", "name": "running-job", "state": "running"}
        uiapp.JOBS["d"] = {"id": "d", "name": "done-job", "state": "done"}
        uiapp.JOBS["f"] = {"id": "f", "name": "failed-job", "state": "failed"}

    uiapp.api_jobs_clear()

    with uiapp.JOBS_LOCK:
        left = set(uiapp.JOBS)
        uiapp.JOBS.clear()
    assert left == {"q", "r"}, f"queued/running must survive, got {left}"


def test_a_job_still_runs_if_its_record_disappears():
    """The record can be evicted while the job waits in a 2-worker pool."""
    import time

    from adforge.ui import app as uiapp

    ran = []
    jid = uiapp._job("evict-me", lambda: ran.append(1) or "ok")
    with uiapp.JOBS_LOCK:
        uiapp.JOBS.pop(jid, None)  # simulate eviction mid-queue
    for _ in range(100):
        if ran:
            break
        time.sleep(0.05)
    assert ran, "job did not run after its record was removed"


def test_job_ids_stay_unique_at_the_eviction_cap():
    from adforge.ui import app as uiapp

    with uiapp.JOBS_LOCK:
        uiapp.JOBS.clear()
    ids = {uiapp._job("same-name", lambda: None) for _ in range(25)}
    assert len(ids) == 25


# --- publish-time guards ----------------------------------------------------
# A backlog that built up while dry run was on stays APPROVED and past-due, so
# without a staleness bound the first tick after going live fires all of it.

def _engine_on_scratch(tmp_path):
    """Reload the engine against a throwaway database."""
    import importlib

    from adforge.config import settings

    settings.db_url = f"sqlite:///{tmp_path}/t.db"
    import adforge.db as db

    importlib.reload(db)
    db.init_db()
    import adforge.scheduler.engine as eng

    importlib.reload(eng)
    return db, eng


def _seed(db, brand="inferix", schedule_enabled=True, age_hours=1,
          account_mode=None):
    from adforge.db import (Account, PostMode, PostStatus, Post, Schedule,
                            session_scope, utcnow)

    with session_scope() as s:
        acct = Account(brand=brand, platform="x", enabled=True, dry_run=True,
                       mode=account_mode or PostMode.AUTO,
                       credentials="{}", options="{}")
        s.add(acct)
        s.flush()
        s.add(Schedule(brand=brand, platform="x", enabled=schedule_enabled,
                       posts_per_day=1, days_of_week="0,1,2,3,4,5,6",
                       times="09:00"))
        post = Post(brand=brand, platform="x", account_id=acct.id,
                    status=PostStatus.APPROVED, mode=PostMode.AUTO,
                    body="x" * 80, attempts=0,
                    scheduled_for=utcnow() - dt.timedelta(hours=age_hours))
        s.add(post)
        s.flush()
        return post.id


def test_a_long_stale_slot_is_not_published(tmp_path):
    db, eng = _engine_on_scratch(tmp_path)
    pid = _seed(db, age_hours=eng.MAX_SLOT_AGE_HOURS + 6)
    eng.promote_and_publish()
    with db.session_scope() as s:
        post = s.get(db.Post, pid)
        assert post.status == db.PostStatus.REJECTED
        assert "old at publish time" in post.error


def test_turning_a_schedule_off_holds_already_approved_posts(tmp_path):
    """'Stop posting' must stop posts already through the review window."""
    db, eng = _engine_on_scratch(tmp_path)
    pid = _seed(db, schedule_enabled=False)
    eng.promote_and_publish()
    with db.session_scope() as s:
        post = s.get(db.Post, pid)
        assert post.status == db.PostStatus.REVIEW
        assert post.mode == db.PostMode.MANUAL
        assert "schedule is switched off" in post.critic_notes


def test_switching_an_account_to_manual_holds_approved_posts(tmp_path):
    db, eng = _engine_on_scratch(tmp_path)
    from adforge.db import PostMode

    pid = _seed(db, account_mode=PostMode.MANUAL)
    eng.promote_and_publish()
    with db.session_scope() as s:
        post = s.get(db.Post, pid)
        assert post.mode == db.PostMode.MANUAL
        assert post.status == db.PostStatus.REVIEW


# --- video prompt scrubbing -------------------------------------------------
# SDXL renders digits and product names as text ON the object and gets it
# wrong: "16GB graphics card" came back with a garbled "166" on the fan hub.

@pytest.mark.parametrize(
    "prompt,gone",
    [
        ("one 16GB graphics card on a dark bench, hard key light", "16"),
        ("a single NVIDIA A100 GPU on a dark bench, cyan rim light", "NVIDIA"),
        ("an RTX 4090 beside a 500W power supply on a rack rail", "4090"),
        ("the 5090 and a 500W PSU, 2x fans, on a rack rail", "5090"),
    ],
)
def test_numbers_and_brands_are_stripped_from_visuals(prompt, gone):
    from adforge.media.script import scrub_visual

    assert gone.lower() not in scrub_visual(prompt).lower()


def test_a_clean_visual_is_left_alone():
    from adforge.media.script import scrub_visual

    clean = "a graphics card standing upright on a dark bench, fans sharply lit"
    assert scrub_visual(clean) == clean


@pytest.mark.parametrize(
    "prompt",
    [
        "one 16GB graphics card on a dark bench, hard key light",
        "an RTX 4090 beside a 500W power supply on a rack rail",
        "the 5090 and a 500W PSU, 2x fans, on a rack rail",
    ],
)
def test_scrubbing_never_leaves_a_dangling_connective(prompt):
    """Deleting the subject left prompts like 'an with VRAM beside a supply'."""
    from adforge.media.script import scrub_visual

    out = scrub_visual(prompt)
    first = out.split()[0].lower()
    assert first not in {"with", "and", "or", "beside", "on", "in", "of", "an", "a",
                         "the"} or out.split()[1].lower() not in {
        "with", "and", "or", "beside", "on", "in", "of"}


def test_an_all_numbers_visual_falls_back_to_a_real_subject():
    from adforge.media.script import FALLBACK_SUBJECT, scrub_visual

    assert scrub_visual("RTX 4090 24GB 500W 2x") == FALLBACK_SUBJECT


def test_video_style_is_preferred_over_the_still_style():
    """The still style's 'abstract compute topology' produced wallpaper."""
    from adforge.brands import get_brand

    for key in ("inferix", "vallorix"):
        vis = get_brand(key).visual
        assert vis.get("video_style"), f"{key} has no video_style"
        assert "abstract" not in vis["video_style"].lower()


# --- metrics collection -----------------------------------------------------
# The Metric table existed from the first commit and nothing ever wrote to it,
# which made measuring a campaign decorative.

def test_unsupported_platforms_say_so_rather_than_reporting_zero():
    from adforge.db import Post
    from adforge.platforms.metrics import COLLECTORS, collect

    for platform in ("discord", "slack", "tiktok"):
        assert platform not in COLLECTORS
        s = collect(Post(platform=platform, remote_id="1"), {})
        assert s["fetched"] is False
        assert "no per-post engagement" in s["note"]


def test_a_post_with_no_remote_id_is_not_queried():
    from adforge.db import Post
    from adforge.platforms.metrics import collect

    s = collect(Post(platform="x", remote_id=""), {})
    assert s["fetched"] is False and "no remote id" in s["note"]


def test_collection_never_raises_on_bad_credentials():
    """A revoked token must not break the scheduler loop."""
    from adforge.db import Post
    from adforge.platforms.metrics import collect

    s = collect(Post(platform="x", remote_id="123"), {})  # no keys at all
    assert s["fetched"] is False
    assert s["impressions"] == 0


def test_not_reported_is_distinguishable_from_zero_engagement():
    """A real zero and 'the platform said nothing' must not look the same."""
    from adforge.platforms.metrics import Stats

    silent = Stats.empty("429")
    genuine = Stats.of(impressions=0, likes=0)
    assert silent["fetched"] is False and genuine["fetched"] is True
    assert silent["impressions"] == genuine["impressions"] == 0


# --- route ordering ---------------------------------------------------------

def test_no_shadowed_routes():
    """A parameterised route declared first swallows its literal siblings.

    /schedules/{sched_id} was declared above /schedules/plan-now and
    /schedules/tick, so FastAPI parsed "plan-now" as an id and both buttons
    returned "Input should be a valid integer". Neither had ever worked, and
    nothing failed loudly enough to notice.
    """
    import re

    from adforge.ui.app import app

    routes = [
        (r.path, sorted(getattr(r, "methods", set()) - {"HEAD"}))
        for r in app.routes
        if hasattr(r, "path") and getattr(r, "methods", None)
    ]
    seen_param, shadowed = [], []
    for path, methods in routes:
        if "{" in path:
            seen_param.append((path, methods))
            continue
        for ppath, pmethods in seen_param:
            if set(methods) & set(pmethods) and re.fullmatch(
                re.sub(r"\{[^}]+\}", "[^/]+", ppath), path
            ):
                shadowed.append(f"{path} is unreachable behind {ppath}")
    assert not shadowed, "; ".join(shadowed)
