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
