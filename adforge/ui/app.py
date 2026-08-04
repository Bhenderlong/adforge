"""
AdForge web UI.

FastAPI + Jinja2 + HTMX. No build step, no node_modules, no CDN - the whole
interface is served from this process so it works on an air-gapped box.

Long-running work (generation, rendering, publishing) is dispatched to a
background thread pool rather than executed in the request, because a single
post can take minutes on a 70B model plus a Wan render and no browser will wait
that long. The UI polls for state.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import gpu as gpumod
from ..brands import all_brands, get_brand, reload_brands
from ..config import ASSETS, BRANDS_DIR, ROOT, settings
from ..db import (
    Account,
    Metric,
    Post,
    PostMode,
    PostStatus,
    RadarTarget,
    RadarThread,
    ReplyDraft,
    Schedule,
    Setting,
    init_db,
    session_scope,
    utcnow,
)
from ..llm.client import health as llm_health
from ..media import comfy
from ..platforms.registry import ADAPTERS, account_status, publish_post
from ..platforms.spec import SPECS, spec
from ..scheduler.engine import build_scheduler, plan_ahead, promote_and_publish
from ..scheduler.planner import plan_slots, tz

log = logging.getLogger("adforge.ui")

# Hosts this app answers state-changing requests on. Anything else is either a
# DNS-rebinding attempt or a misconfigured proxy; both should fail loudly.
_ALLOWED_HOSTS = {
    "localhost", "127.0.0.1", "::1", "[::1]",
    settings.host, *(h.strip() for h in settings.extra_hosts.split(",") if h.strip()),
}

HERE = Path(__file__).resolve().parent
app = FastAPI(title="AdForge")


@app.middleware("http")
async def _same_origin_only(request: Request, call_next):
    """Reject cross-origin state changes and unexpected Host headers.

    Binding to 127.0.0.1 does NOT make this safe on its own: any page the
    operator's browser loads can POST here cross-origin. An HTML form POST is a
    "simple request", so there is no preflight and CORS never gets a say on the
    send. Without this an ad or iframe on an unrelated site could enable an
    account, point the Discord webhook at an attacker, switch that account's
    dry-run off, and trigger publish_now - all silently.

    Checking Host as well closes DNS rebinding, where an attacker-controlled
    name resolves to 127.0.0.1 and makes their page same-origin with this app.
    """
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        host = (request.headers.get("host") or "").split(":")[0]
        if host not in _ALLOWED_HOSTS:
            log.warning("rejected %s %s: unexpected Host %r",
                        request.method, request.url.path, host)
            return JSONResponse({"detail": "unexpected Host header"}, status_code=421)

        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        source = origin or referer
        if source is None:
            # curl and the test-suite send neither. A browser always sends at
            # least a Referer for a form POST, so this only admits non-browser
            # clients, which are not the CSRF threat.
            pass
        else:
            try:
                src_host = urlparse(source).hostname or ""
            except ValueError:
                src_host = ""
            if src_host not in _ALLOWED_HOSTS:
                log.warning("rejected cross-origin %s %s from %r",
                            request.method, request.url.path, source)
                return JSONResponse(
                    {"detail": "cross-origin request refused"}, status_code=403
                )
    return await call_next(request)


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

def _localtime(value, fmt: str = "%d %b %H:%M") -> str:
    """Render a stored timestamp in the configured local timezone.

    Every datetime column comes back NAIVE from SQLite while everything written
    to it was UTC, so printing one directly showed UTC labelled as local. The
    queue said 13:00 and the detail page said 09:00 for the same row. Templates
    must use this filter rather than calling strftime on the column.
    """
    if value is None:
        return ""
    if getattr(value, "tzinfo", None) is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(tz()).strftime(fmt)


templates.env.filters["localtime"] = _localtime


# Bounded: two concurrent jobs already saturate the GPU lock, and a deeper
# queue only makes the UI lie about how fast things will finish.
POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="adforge-job")
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
_JOB_SEQ = 0
_scheduler = None


MAX_TRACKED_JOBS = 200


def _jobs_snapshot(limit: int) -> list[dict]:
    """Copies of the most recent job records, taken under the lock.

    Both parts matter. Reading JOBS without the lock races `_job` inserting
    from another thread ("dictionary changed size during iteration" -> a 500 on
    the dashboard). And handing out references to the live inner dicts let a
    poll serialize one while a worker was mutating it, producing torn records
    like state="running" alongside a "finished" timestamp.
    """
    with JOBS_LOCK:
        return [dict(j) for j in list(JOBS.values())[-limit:]]


def _job(name: str, fn, *args, **kw) -> str:
    """Run fn in the pool, tracking its state for the UI."""
    global _JOB_SEQ
    with JOBS_LOCK:
        # A monotonic counter, not len(JOBS): once eviction pins the dict at
        # the cap, len() stops changing and two same-named jobs in the same
        # millisecond collide again.
        _JOB_SEQ += 1
        jid = f"{name}-{_JOB_SEQ}-{int(utcnow().timestamp() * 1000)}"
        JOBS[jid] = {"id": jid, "name": name, "state": "queued",
                     "started": utcnow().isoformat(), "detail": ""}
        # Bounded: only the last handful is ever displayed, so anything older
        # was pure retained garbage in a process meant to run for weeks.
        # Evict only TERMINAL records - see api_jobs_clear for why "queued"
        # must never be dropped.
        while len(JOBS) > MAX_TRACKED_JOBS:
            for old_id, rec in list(JOBS.items()):
                if rec["state"] in ("done", "failed"):
                    JOBS.pop(old_id, None)
                    break
            else:
                break

    def wrapper():
        with JOBS_LOCK:
            # setdefault, not [jid][...]: if the record was evicted or cleared
            # while this job sat in the queue, a bare KeyError here would kill
            # the job before fn ever ran, and die unnoticed inside the Future.
            JOBS.setdefault(jid, {"id": jid, "name": name, "detail": "",
                                  "started": utcnow().isoformat()})
            JOBS[jid]["state"] = "running"
        try:
            result = fn(*args, **kw)
            with JOBS_LOCK:
                JOBS.setdefault(jid, {"id": jid, "name": name}).update(
                    state="done", detail=str(result)[:300],
                    finished=utcnow().isoformat())
        except Exception as e:  # noqa: BLE001 - surface it in the UI
            log.exception("job %s failed", name)
            with JOBS_LOCK:
                JOBS.setdefault(jid, {"id": jid, "name": name}).update(
                    state="failed", detail=f"{type(e).__name__}: {e}"[:300],
                    finished=utcnow().isoformat())

    POOL.submit(wrapper)
    return jid


_ENV_UNSAFE = re.compile(r"[\r\n\x00]")


def _env_safe(value: str) -> str:
    """Strip anything that could forge an extra line in the .env file.

    The file is written one `KEY=value` per line and re-read by
    pydantic-settings, so a newline inside a value injects a whole new setting.
    Values are emitted in sorted order and the last definition wins, which made
    any free-text field sorting after ADFORGE_DRY_RUN - the timezone, the host,
    the model fallback list - able to smuggle in `ADFORGE_DRY_RUN=false` and
    silently defeat the typed GO LIVE confirmation below.
    """
    return _ENV_UNSAFE.sub(" ", value).strip()


def _flash(request: Request, msg: str, kind: str = "ok") -> None:
    request.session_flash = (kind, msg)  # type: ignore[attr-defined]


@app.on_event("startup")
def _startup() -> None:
    global _scheduler
    init_db()
    _seed_defaults()
    if settings.ui_runs_scheduler:
        _scheduler = build_scheduler()
        _scheduler.start()
    else:
        log.info("scheduler disabled in this process (expecting run.py --worker)")
    log.info("adforge ui up on %s:%s", settings.host, settings.port)


@app.on_event("shutdown")
def _shutdown() -> None:
    if _scheduler:
        _scheduler.shutdown(wait=False)
    POOL.shutdown(wait=False)


def _seed_defaults() -> None:
    """Create a disabled account + schedule row per brand/platform.

    Every destination is visible in the UI from first run, all switched off,
    so enabling one is a deliberate act rather than something that happens by
    forgetting to disable it.
    """
    with session_scope() as s:
        for bkey in all_brands():
            for pkey in SPECS:
                if not s.query(Account).filter_by(brand=bkey, platform=pkey).first():
                    s.add(Account(brand=bkey, platform=pkey, enabled=False,
                                  dry_run=True, mode=PostMode.AUTO,
                                  credentials="{}", options="{}"))
                if not s.query(Schedule).filter_by(brand=bkey, platform=pkey).first():
                    s.add(Schedule(brand=bkey, platform=pkey, enabled=False,
                                   posts_per_day=1, days_of_week="0,1,2,3,4",
                                   times="09:30", window_start="09:00",
                                   window_end="17:00", jitter_minutes=12))


# ---------------------------------------------------------------------------
# Template context
# ---------------------------------------------------------------------------


def ctx(request: Request, **kw) -> dict:
    base = {
        "request": request,
        "brands": all_brands(),
        "platforms": SPECS,
        "settings": settings,
        "now": utcnow(),
        # Templates render stored (naive UTC) timestamps in local time.
        "tz": tz(),
        "utc": dt.timezone.utc,
    }
    base.update(kw)
    return base


def _status_counts(s) -> dict:
    out = {}
    for st in PostStatus:
        out[st.value] = s.query(Post.id).filter(Post.status == st).count()
    return out


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    with session_scope() as s:
        counts = _status_counts(s)
        upcoming = (
            s.query(Post)
            .filter(Post.status.in_([PostStatus.REVIEW, PostStatus.APPROVED]))
            .order_by(Post.scheduled_for)
            .limit(12)
            .all()
        )
        recent = (
            s.query(Post)
            .filter(Post.status == PostStatus.PUBLISHED)
            .order_by(Post.published_at.desc())
            .limit(8)
            .all()
        )
        hot = (
            s.query(RadarThread)
            .filter(RadarThread.dismissed.is_(False))
            .order_by(RadarThread.relevance.desc())
            .limit(6)
            .all()
        )
        live_accounts = (
            s.query(Account)
            .filter(Account.enabled.is_(True))
            .all()
        )
        live = [(a, *account_status(a)) for a in live_accounts]

    ok_llm, msg_llm = llm_health()
    return templates.TemplateResponse(
        request, "dashboard.html",
        ctx(request, counts=counts, upcoming=upcoming, recent=recent, hot=hot,
            live=live, llm_ok=ok_llm, llm_msg=msg_llm,
            comfy_ok=comfy.is_up(), vram=gpumod.vram(),
            gpu_mode=gpumod.current_mode(), jobs=_jobs_snapshot(8)),
    )


@app.get("/api/status")
def api_status():
    """Polled by the dashboard for live status without a full reload."""
    ok_llm, msg_llm = llm_health()
    with session_scope() as s:
        counts = _status_counts(s)
    jobs = _jobs_snapshot(8)
    return JSONResponse({
        "llm": {"ok": ok_llm, "detail": msg_llm},
        "comfy": comfy.is_up(),
        "gpu_mode": gpumod.current_mode(),
        "vram": gpumod.vram(),
        "counts": counts,
        "jobs": jobs,
    })


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


@app.get("/queue", response_class=HTMLResponse)
def queue(request: Request, brand: str = "", platform: str = "", status: str = ""):
    with session_scope() as s:
        q = s.query(Post)
        if brand:
            q = q.filter(Post.brand == brand)
        if platform:
            q = q.filter(Post.platform == platform)
        if status:
            q = q.filter(Post.status == PostStatus(status))
        posts = q.order_by(Post.scheduled_for.desc().nullslast(),
                           Post.id.desc()).limit(200).all()
        counts = _status_counts(s)
    return templates.TemplateResponse(
        request, "queue.html",
        ctx(request, posts=posts, counts=counts, f_brand=brand,
            f_platform=platform, f_status=status, statuses=list(PostStatus)),
    )


@app.get("/post/{post_id}", response_class=HTMLResponse)
def post_detail(request: Request, post_id: int):
    with session_scope() as s:
        post = s.get(Post, post_id)
        if not post:
            raise HTTPException(404, "no such post")
        account = s.get(Account, post.account_id) if post.account_id else None
        meta = {}
        try:
            meta = json.loads(post.generation_meta or "{}")
        except json.JSONDecodeError:
            pass
        latest = (
            s.query(Metric)
            .filter(Metric.post_id == post.id)
            .order_by(Metric.captured_at.desc())
            .first()
        )
        stats = {c.name: getattr(latest, c.name) for c in Metric.__table__.columns} if latest else None
    return templates.TemplateResponse(
        request, "post_detail.html",
        ctx(request, post=post, account=account, ps=spec(post.platform), meta=meta,
            stats=stats),
    )


@app.post("/post/{post_id}/edit")
def post_edit(
    post_id: int,
    body: str = Form(""),
    title: str = Form(""),
    hashtags: str = Form(""),
    link: str = Form(""),
    scheduled_for: str = Form(""),
):
    with session_scope() as s:
        post = s.get(Post, post_id)
        if not post:
            raise HTTPException(404, "no such post")
        ps = spec(post.platform)
        if len(body) > ps.max_chars:
            raise HTTPException(
                400, f"{len(body)} chars exceeds the {ps.label} limit of {ps.max_chars}"
            )
        post.body = body
        post.title = title
        post.hashtags = hashtags
        post.link = link
        if scheduled_for:
            try:
                # The form renders local time (see post_detail.html), so parse
                # it back as local. Previously the template displayed the raw
                # stored value - naive UTC - while this parsed it as local, so
                # saving a post without touching the field moved it by the UTC
                # offset. Three edits pushed a post most of a day.
                local = dt.datetime.fromisoformat(scheduled_for).replace(tzinfo=tz())
                post.scheduled_for = local.astimezone(dt.timezone.utc)
            except ValueError:
                raise HTTPException(400, f"bad datetime {scheduled_for!r}") from None
    return RedirectResponse(f"/post/{post_id}", status_code=303)


@app.post("/post/{post_id}/action")
def post_action(post_id: int, action: str = Form(...)):
    with session_scope() as s:
        post = s.get(Post, post_id)
        if not post:
            raise HTTPException(404, "no such post")

        if action == "approve":
            post.status = PostStatus.APPROVED
            post.review_until = None
        elif action == "reject":
            post.status = PostStatus.REJECTED
        elif action == "hold":
            # Pull back into review and clear the timer, so an auto account
            # cannot publish it while the user is still deciding.
            post.status = PostStatus.REVIEW
            post.mode = PostMode.MANUAL
            post.review_until = None
        elif action == "publish_now":
            # `Account.enabled` is the master off-switch and every other send
            # path honours it. This one fetched the account and then never
            # looked at it, so a post could be transmitted from a destination
            # the user had deliberately switched off - while the panel still
            # displayed "live".
            account = s.get(Account, post.account_id) if post.account_id else None
            if account is None or not account.enabled:
                raise HTTPException(
                    400,
                    "this post has no enabled account - enable the destination "
                    "on the Accounts page first",
                )
            post.status = PostStatus.APPROVED
            post.scheduled_for = utcnow()
            post.attempts = 0
            _job(f"publish#{post.id}", _publish_one, post.id)
        elif action == "regenerate":
            _job(f"regen#{post.id}", _regenerate, post.id)
        elif action == "regenerate_media":
            _job(f"media#{post.id}", _regen_media, post.id)
        elif action == "delete":
            s.delete(post)
            return RedirectResponse("/queue", status_code=303)
        else:
            raise HTTPException(400, f"unknown action {action!r}")
    return RedirectResponse(f"/post/{post_id}", status_code=303)


def _publish_one(post_id: int) -> str:
    """Publish one post through the same claim the scheduler uses.

    Going straight to publish_post here raced the timed tick: for the up to a
    minute this job takes, the scheduler saw an APPROVED, due, attempts=0 post
    and sent it as well. The claim makes whichever one gets there first the
    only one that transmits. It also re-checks `enabled`, since the account can
    be switched off between the click and the send.
    """
    from ..scheduler.engine import publish_claimed

    sent = publish_claimed(post_id)
    with session_scope() as s:
        post = s.get(Post, post_id)
        if post is None:
            return "post deleted"
        if sent:
            return post.remote_url or "published"
        return post.error or f"not sent ({post.status.value})"


def _regenerate(post_id: int) -> str:
    from ..llm.copywriter import write_post
    from ..scheduler.planner import recent_bodies

    with session_scope() as s:
        post = s.get(Post, post_id)
        if post is None:
            return "post deleted while the job was running"
        brand = get_brand(post.brand)
        ps = spec(post.platform)
        pillar = brand.pillar(post.pillar) if post.pillar else brand.pick_pillar()
        recent = recent_bodies(s, post.brand, post.platform)
        draft = write_post(brand, ps, pillar, recent=recent)
        post.body = draft.text
        post.hashtags = ",".join(draft.hashtags)
        post.quality_score = draft.score
        post.critic_notes = "; ".join(draft.problems[:6])
        post.status = PostStatus.REVIEW
        post.generation_meta = json.dumps(
            {"angle": draft.angle, "attempts": draft.attempts}
        )

        # Mirror fill_slot's gating. Without this a rewrite inherited the
        # original post's review_until - normally already in the past, since
        # posts are planned up to 26h ahead - so freshly rewritten copy was
        # promoted on the very next 60-second tick with no review window at
        # all. Worse, a rewrite that scored BELOW the bar stayed AUTO and
        # auto-published, which is exactly what the quality gate exists to
        # prevent.
        if not draft.passed:
            post.mode = PostMode.MANUAL
            post.review_until = None
            post.critic_notes = (
                f"below quality bar (score {draft.score:.1f}); {post.critic_notes}"
            )[:2000]
        elif post.mode == PostMode.AUTO:
            post.review_until = utcnow() + dt.timedelta(
                minutes=max(0, settings.review_window_minutes)
            )
        return f"score {draft.score:.1f}{'' if draft.passed else ' (below bar, held for review)'}"


def _regen_media(post_id: int) -> str:
    from ..llm.copywriter import media_prompt
    from ..media.images import generate_image

    with session_scope() as s:
        post = s.get(Post, post_id)
        if post is None:
            return "post deleted while the job was running"
        brand = get_brand(post.brand)
        ps = spec(post.platform)
        prompt = post.media_prompt or media_prompt(brand, post.body, post.pillar)
        post.media_prompt = prompt
        path = generate_image(
            brand, ps, prompt, f"{post.brand}_{post.platform}_{post.id}"
        )
        post.image_path = str(path)
        return path.name


@app.get("/asset/{name}")
def asset(name: str):
    """Serve a generated asset. Path-traversal safe."""
    target = (ASSETS / name).resolve()
    # is_relative_to, not startswith: a prefix match with no separator also
    # accepts a sibling directory such as data/assets_backup/.
    if not target.is_relative_to(ASSETS.resolve()) or not target.exists():
        raise HTTPException(404, "no such asset")
    return FileResponse(target)


# ---------------------------------------------------------------------------
# Composer - manual one-off posts
# ---------------------------------------------------------------------------


@app.get("/compose", response_class=HTMLResponse)
def compose(request: Request):
    return templates.TemplateResponse(
        request, "compose.html", ctx(request))


@app.post("/compose")
def compose_submit(
    brand: str = Form(...),
    platform: str = Form(...),
    pillar: str = Form(""),
    angle: str = Form(""),
    with_media: str = Form(""),
    when: str = Form(""),
):
    _job(
        f"compose:{brand}/{platform}",
        _compose_job, brand, platform, pillar, angle, bool(with_media), when,
    )
    return RedirectResponse("/queue", status_code=303)


def _compose_job(brand_key, platform, pillar_key, angle, with_media, when) -> str:
    from ..llm.copywriter import media_prompt, write_post
    from ..media.images import generate_image
    from ..scheduler.planner import recent_bodies

    brand = get_brand(brand_key)
    ps = spec(platform)
    pillar = brand.pillar(pillar_key) if pillar_key else brand.pick_pillar()

    with session_scope() as s:
        recent = recent_bodies(s, brand_key, platform)
        draft = write_post(brand, ps, pillar, angle=angle or None, recent=recent)
        account = (
            s.query(Account).filter_by(brand=brand_key, platform=platform).first()
        )
        scheduled = utcnow()
        if when:
            try:
                scheduled = (
                    dt.datetime.fromisoformat(when)
                    .replace(tzinfo=tz())
                    .astimezone(dt.timezone.utc)
                )
            except ValueError:
                pass

        post = Post(
            brand=brand_key, platform=platform,
            account_id=account.id if account else None,
            pillar=pillar.key, status=PostStatus.REVIEW, mode=PostMode.MANUAL,
            body=draft.text, hashtags=",".join(draft.hashtags), link=brand.url,
            scheduled_for=scheduled, quality_score=draft.score,
            critic_notes="; ".join(draft.problems[:6]),
            generation_meta=json.dumps({"angle": draft.angle,
                                        "attempts": draft.attempts}),
        )
        if ps.key in ("reddit", "youtube"):
            post.title = draft.text.strip().split("\n")[0][:290]

        # Render BEFORE the insert, and name the asset with a random slug
        # rather than post.id.
        #
        # Flushing first opens a SQLite write transaction, and generate_image
        # then waits on the GPU lock and ComfyUI for up to comfy_timeout (30
        # minutes). Every other writer - the 60-second publish tick, saving an
        # account, approving a post - would block for busy_timeout and then
        # return a 500. This is the same trap already fixed in
        # scheduler/planner.fill_slot; Compose still had it.
        if with_media and ps.supports_image:
            slug = f"{brand_key}_{platform}_{uuid.uuid4().hex[:10]}"
            try:
                prompt = media_prompt(brand, draft.text, draft.angle)
                post.media_prompt = prompt
                post.image_path = str(generate_image(brand, ps, prompt, slug))
            except Exception as e:  # noqa: BLE001 - never lose the copy
                log.warning("compose media failed for %s: %s", slug, e)
                post.critic_notes = (
                    f"media failed: {str(e)[:200]}; {post.critic_notes}"
                )[:2000]

        s.add(post)
        s.flush()
        pid = post.id
    return f"post {pid}, score {draft.score:.1f}"


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


@app.get("/schedules", response_class=HTMLResponse)
def schedules(request: Request):
    with session_scope() as s:
        rows = s.query(Schedule).order_by(Schedule.brand, Schedule.platform).all()
        accounts = {
            (a.brand, a.platform): a for a in s.query(Account).all()
        }
        # Preview the next two days so the cadence is visible, not theoretical.
        preview = {}
        for sched in rows:
            if not sched.enabled:
                continue
            slots = []
            for d in range(3):
                # LOCAL date, matching the planner. Using the UTC date made
                # the preview start at tomorrow from 20:00 local onward, so it
                # hid exactly the slots about to be filled tonight.
                day = utcnow().astimezone(tz()).date() + dt.timedelta(days=d)
                slots += plan_slots(sched, day)
            preview[sched.id] = sorted(slots)[:8]
    return templates.TemplateResponse(
        request, "schedules.html",
        ctx(request, schedules=rows, accounts=accounts, preview=preview, tz=tz()),
    )


@app.post("/schedules/{sched_id}")
def schedule_save(
    sched_id: int,
    enabled: str = Form(""),
    posts_per_day: int = Form(1),
    days_of_week: list[str] = Form([]),
    times: str = Form(""),
    window_start: str = Form("09:00"),
    window_end: str = Form("17:00"),
    jitter_minutes: int = Form(12),
    pillars: list[str] = Form([]),
    attach_media: str = Form(""),
):
    with session_scope() as s:
        sched = s.get(Schedule, sched_id)
        if not sched:
            raise HTTPException(404, "no such schedule")
        sched.enabled = bool(enabled)
        sched.posts_per_day = max(0, min(int(posts_per_day), settings.daily_post_cap))
        sched.days_of_week = ",".join(days_of_week)
        sched.times = times
        sched.window_start = window_start
        sched.window_end = window_end
        sched.jitter_minutes = max(0, int(jitter_minutes))
        sched.pillars = ",".join(pillars)
        sched.attach_media = bool(attach_media)
    return RedirectResponse("/schedules", status_code=303)


@app.post("/schedules/plan-now")
def schedule_plan_now():
    _job("plan", plan_ahead)
    return RedirectResponse("/schedules", status_code=303)


@app.post("/schedules/tick")
def schedule_tick():
    _job("publish-tick", promote_and_publish)
    return RedirectResponse("/queue", status_code=303)


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


@app.get("/accounts", response_class=HTMLResponse)
def accounts(request: Request):
    with session_scope() as s:
        rows = s.query(Account).order_by(Account.brand, Account.platform).all()
        status = {a.id: account_status(a) for a in rows}
        creds = {a.id: a.creds() for a in rows}
        opts = {a.id: a.opts() for a in rows}
    required = {k: a.required_credentials for k, a in ADAPTERS.items()}
    return templates.TemplateResponse(
        request, "accounts.html",
        ctx(request, accounts=rows, status=status, creds=creds, opts=opts,
            required=required, notes={k: SPECS[k].notes for k in SPECS}),
    )


@app.post("/accounts/{account_id}")
async def account_save(request: Request, account_id: int):
    form = await request.form()
    with session_scope() as s:
        acct = s.get(Account, account_id)
        if not acct:
            raise HTTPException(404, "no such account")

        acct.handle = str(form.get("handle", ""))
        acct.enabled = bool(form.get("enabled"))
        acct.mode = PostMode(str(form.get("mode", "AUTO")))
        # Tri-state: "" inherits the global dry-run switch.
        dr = str(form.get("dry_run", ""))
        acct.dry_run = None if dr == "" else (dr == "1")

        # Credential fields arrive as cred_<name>. An empty submission keeps
        # the stored value, so the UI can show masked placeholders without
        # wiping secrets on every save.
        creds = acct.creds()
        for key, value in form.items():
            if key.startswith("cred_"):
                name = key[5:]
                if str(value).strip():
                    creds[name] = str(value).strip()
        acct.credentials = json.dumps(creds)

        opts = acct.opts()
        for key, value in form.items():
            if key.startswith("opt_"):
                name = key[4:]
                v = str(value).strip()
                if v.lower() in ("true", "false"):
                    opts[name] = v.lower() == "true"
                elif v:
                    opts[name] = v
        # Checkbox options only appear in the form when ticked.
        for flag in ("allowed", "links_allowed", "allow_image", "direct_post"):
            if f"optflag_{flag}" in form:
                opts[flag] = bool(form.get(f"optflag_{flag}"))
            elif f"optflag_{flag}_present" in form:
                opts[flag] = False

        # Bind the Reddit rules confirmation to the subreddit it was given for,
        # so retyping the name does not silently carry the tick over to a
        # community nobody vetted. The adapter compares the two.
        if acct.platform == "reddit":
            opts["allowed_for"] = str(opts.get("subreddit", "")) if opts.get("allowed") else ""

        acct.options = json.dumps(opts)
    return RedirectResponse("/accounts", status_code=303)


@app.post("/accounts/{account_id}/clear-credential")
def account_clear_credential(account_id: int, name: str = Form(...)):
    with session_scope() as s:
        acct = s.get(Account, account_id)
        if not acct:
            raise HTTPException(404, "no such account")
        creds = acct.creds()
        creds.pop(name, None)
        acct.credentials = json.dumps(creds)
    return RedirectResponse("/accounts", status_code=303)


@app.post("/accounts/{account_id}/test")
def account_test(account_id: int):
    """Ask the platform whether these credentials actually work.

    Field-presence checking answers the wrong question. This performs a
    READ-ONLY identity call per platform - nothing here can post - and reports
    the authenticated identity, which also catches valid credentials pointing
    at the wrong account.
    """
    from ..platforms.verify import verify

    with session_scope() as s:
        acct = s.get(Account, account_id)
        if not acct:
            raise HTTPException(404, "no such account")
        platform, creds, opts = acct.platform, acct.creds(), acct.opts()

    missing = ADAPTERS[platform].validate(creds, opts)
    if missing:
        return JSONResponse({"account": account_id, "ok": False,
                             "detail": "; ".join(missing)})
    try:
        ok, detail = verify(platform, creds, opts)
    except Exception as e:  # noqa: BLE001 - never 500 a diagnostic button
        ok, detail = False, f"{type(e).__name__}: {e}"[:200]
    return JSONResponse({"account": account_id, "ok": ok, "detail": detail})


# ---------------------------------------------------------------------------
# Radar
# ---------------------------------------------------------------------------


@app.get("/radar", response_class=HTMLResponse)
def radar(request: Request, brand: str = "", min_rel: float = 0.0):
    with session_scope() as s:
        q = s.query(RadarThread).filter(RadarThread.dismissed.is_(False))
        if brand:
            q = q.filter(RadarThread.brand == brand)
        if min_rel:
            q = q.filter(RadarThread.relevance >= min_rel)
        threads = q.order_by(RadarThread.relevance.desc()).limit(120).all()
        targets = s.query(RadarTarget).order_by(RadarTarget.brand).all()
        drafts = {}
        for t in threads:
            d = (
                s.query(ReplyDraft)
                .filter(ReplyDraft.thread_id == t.id)
                .order_by(ReplyDraft.id.desc())
                .first()
            )
            if d:
                drafts[t.id] = d
    return templates.TemplateResponse(
        request, "radar.html",
        ctx(request, threads=threads, targets=targets, drafts=drafts,
            f_brand=brand, f_min=min_rel),
    )


@app.post("/radar/targets/new")
def radar_target_new(
    brand: str = Form(...),
    source: str = Form(...),
    target: str = Form(...),
    keywords: str = Form(""),
    promo_allowed: str = Form(""),
    rules_note: str = Form(""),
    min_relevance: float = Form(0.6),
):
    with session_scope() as s:
        s.add(RadarTarget(
            brand=brand, source=source, target=target.strip().lstrip("r/"),
            keywords=keywords, enabled=True,
            promo_allowed=bool(promo_allowed), rules_note=rules_note,
            min_relevance=min_relevance,
        ))
    return RedirectResponse("/radar", status_code=303)


@app.post("/radar/targets/{target_id}")
def radar_target_save(
    target_id: int,
    keywords: str = Form(""),
    enabled: str = Form(""),
    promo_allowed: str = Form(""),
    rules_note: str = Form(""),
    min_relevance: float = Form(0.6),
    delete: str = Form(""),
):
    with session_scope() as s:
        t = s.get(RadarTarget, target_id)
        if not t:
            raise HTTPException(404, "no such target")
        if delete:
            s.delete(t)
        else:
            t.keywords = keywords
            t.enabled = bool(enabled)
            t.promo_allowed = bool(promo_allowed)
            t.rules_note = rules_note
            t.min_relevance = min_relevance
    return RedirectResponse("/radar", status_code=303)


@app.post("/radar/scan")
def radar_scan_now():
    from ..radar.scan import scan_all

    _job("radar-scan", scan_all)
    return RedirectResponse("/radar", status_code=303)


@app.post("/radar/thread/{thread_id}/draft")
def radar_draft(thread_id: int):
    _job(f"draft#{thread_id}", _draft_job, thread_id)
    return RedirectResponse("/radar", status_code=303)


def _draft_job(thread_id: int) -> str:
    from ..radar.reply import draft_reply

    with session_scope() as s:
        thread = s.get(RadarThread, thread_id)
        if not thread:
            return "thread gone"
        target = (
            s.query(RadarTarget)
            .filter(RadarTarget.brand == thread.brand)
            .filter(RadarTarget.source == thread.source)
            .first()
        )
        links_allowed = bool(
            thread.promo_allowed and target and target.promo_allowed
        )
        draft = draft_reply(thread, links_allowed)
        if draft is None:
            return "model declined - nothing useful to add"
        s.add(draft)
        return "drafted"


@app.post("/radar/thread/{thread_id}/dismiss")
def radar_dismiss(thread_id: int):
    with session_scope() as s:
        t = s.get(RadarThread, thread_id)
        if t:
            t.dismissed = True
    return RedirectResponse("/radar", status_code=303)


@app.post("/radar/draft/{draft_id}")
def radar_draft_action(draft_id: int, action: str = Form(...), body: str = Form("")):
    with session_scope() as s:
        draft = s.get(ReplyDraft, draft_id)
        if not draft:
            raise HTTPException(404, "no such draft")

        if action == "save":
            draft.body = body
        elif action == "reject":
            draft.status = PostStatus.REJECTED
        elif action == "send":
            if body:
                draft.body = body
            _job(f"reply#{draft.id}", _send_reply_job, draft.id)
        else:
            raise HTTPException(400, f"unknown action {action!r}")
    return RedirectResponse("/radar", status_code=303)


def _send_reply_job(draft_id: int) -> str:
    from ..radar import policy
    from ..radar.reply import send_reply

    with session_scope() as s:
        draft = s.get(ReplyDraft, draft_id)
        if draft is None:
            return "draft deleted while the job was running"
        thread = s.get(RadarThread, draft.thread_id)
        if thread is None:
            draft.error = "thread no longer exists"
            return draft.error

        # Enforce the daily reply cap. settings.radar_max_replies_per_day was
        # defined, documented as a safety rail and editable in the UI, but read
        # nowhere - you could set it to 1 and send fifty. Volume is exactly
        # what turns disclosed participation into spam, so a control that
        # claims to bound it has to actually bound it.
        since = utcnow() - dt.timedelta(hours=24)
        sent_today = (
            s.query(ReplyDraft.id)
            .filter(ReplyDraft.brand == draft.brand,
                    ReplyDraft.status == PostStatus.PUBLISHED,
                    ReplyDraft.posted_at.isnot(None),
                    ReplyDraft.posted_at >= since)
            .count()
        )
        if sent_today >= settings.radar_max_replies_per_day:
            draft.error = (
                f"daily reply cap reached ({sent_today}/"
                f"{settings.radar_max_replies_per_day} for {draft.brand} in the "
                f"last 24h) - raise it on Settings or send this tomorrow"
            )
            log.info("reply %s blocked by daily cap", draft.id)
            return draft.error
        account = (
            s.query(Account)
            .filter(Account.brand == draft.brand, Account.platform == thread.source)
            .first()
        )
        if account is None or not account.enabled:
            draft.error = "no enabled account for this source"
            return draft.error

        target = (
            s.query(RadarTarget)
            .filter(RadarTarget.brand == thread.brand,
                    RadarTarget.source == thread.source)
            .first()
        )
        links_allowed = bool(thread.promo_allowed and target and target.promo_allowed)

        # Re-check policy at send time: the body may have been edited in the UI
        # since it was drafted.
        problems = policy.blocking(
            policy.check_reply(draft.body, draft.brand, links_allowed)
        )
        if problems:
            draft.error = "blocked by policy: " + "; ".join(problems)
            draft.status = PostStatus.REJECTED
            return draft.error

        dry = account.dry_run if account.dry_run is not None else settings.dry_run
        try:
            url = send_reply(draft, thread, account, dry)
        except Exception as e:  # noqa: BLE001
            draft.error = str(e)[:600]
            draft.status = PostStatus.FAILED
            return draft.error

        draft.status = PostStatus.PUBLISHED
        draft.posted_at = utcnow()
        draft.remote_url = url
        return url or "dry run"


# ---------------------------------------------------------------------------
# Brands
# ---------------------------------------------------------------------------


@app.get("/brands", response_class=HTMLResponse)
def brands_page(request: Request):
    raw = {}
    for path in sorted(BRANDS_DIR.glob("*.yaml")):
        raw[path.stem] = path.read_text()
    return templates.TemplateResponse(
        request, "brands.html", ctx(request, raw=raw))


@app.post("/brands/{key}")
def brand_save(key: str, content: str = Form(...)):
    import yaml

    path = BRANDS_DIR / f"{key}.yaml"
    if not path.exists():
        raise HTTPException(404, "no such brand")

    # Validate before writing. A brand file that fails to parse takes the
    # scheduler down at the next planning tick, so it must never be saved
    # broken.
    try:
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"invalid YAML: {e}") from None

    backup = path.with_suffix(".yaml.bak")
    backup.write_text(path.read_text())
    path.write_text(content)
    reload_brands()
    try:
        get_brand(key)
    except Exception as e:  # noqa: BLE001 - roll back a semantically bad file
        path.write_text(backup.read_text())
        reload_brands()
        raise HTTPException(400, f"rejected, restored previous: {e}") from None
    return RedirectResponse("/brands", status_code=303)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    from ..llm.client import available_models

    with session_scope() as s:
        overrides = {row.key: row.value for row in s.query(Setting).all()}
    return templates.TemplateResponse(
        request, "settings.html",
        ctx(request, overrides=overrides, models=available_models(),
            checkpoints=comfy.checkpoints(), unets=comfy.unet_models(),
            vaes=comfy.list_options("VAELoader", "vae_name"),
            clips=comfy.list_options("CLIPLoader", "clip_name"),
            env_path=ROOT / ".env"),
    )


@app.post("/settings")
async def settings_save(request: Request):
    """Persist overrides to .env so they survive a restart.

    Written as ADFORGE_<KEY>=value, which pydantic-settings reads on boot. The
    running process is not hot-patched: a restart is required, and the UI says
    so rather than pretending the change took effect.
    """
    form = await request.form()
    env_path = ROOT / ".env"
    existing = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()

    for key, value in form.items():
        if not key.startswith("set_"):
            continue
        name = f"ADFORGE_{key[4:].upper()}"
        existing[name] = _env_safe(str(value))

    # Checkboxes are absent from the form when unticked, so each one carries a
    # hidden `bool_<name>_present` companion. The companion marks that the
    # control was on the page; it is not itself a setting, so drive the loop
    # from the companions and never write them out.
    for key in list(form.keys()):
        if key.startswith("bool_") and key.endswith("_present"):
            name = key[len("bool_"):-len("_present")]
            if name == "dry_run":
                continue  # handled below, it needs a stronger gate
            existing[f"ADFORGE_{name.upper()}"] = (
                "true" if form.get(f"bool_{name}") else "false"
            )

    # Dry run is the one setting whose wrong value publishes to the world, so
    # it is not driven by a checkbox being absent. Turning it OFF requires an
    # explicit typed confirmation; anything else leaves it on. A partial or
    # malformed POST can therefore only ever fail safe.
    #
    # This is not hypothetical: exercising this endpoint during development
    # silently flipped dry run off, because "checkbox missing" and "user wants
    # live publishing" were the same signal.
    if str(form.get("disable_dry_run_confirm", "")).strip().upper() == "GO LIVE":
        existing["ADFORGE_DRY_RUN"] = "false"
        log.warning("global dry run DISABLED via the settings page")
    else:
        existing["ADFORGE_DRY_RUN"] = "true"

    lines = ["# Written by the AdForge settings page. Restart to apply.\n"]
    lines += [f"{k}={v}\n" for k, v in sorted(existing.items())]
    env_path.write_text("".join(lines))
    env_path.chmod(0o600)
    return RedirectResponse("/settings?saved=1", status_code=303)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@app.get("/api/jobs")
def api_jobs():
    return JSONResponse(_jobs_snapshot(30))


@app.post("/api/jobs/clear")
def api_jobs_clear():
    # Only terminal records. This used to drop anything not "running", which
    # included "queued" - and clearing the panel then killed jobs that had not
    # started yet, silently, because the worker KeyErrored before calling fn.
    # With a 2-worker pool and jobs that run for minutes, queued is normal.
    with JOBS_LOCK:
        for jid in [j for j, v in JOBS.items() if v["state"] in ("done", "failed")]:
            JOBS.pop(jid, None)
    return JSONResponse({"ok": True})
