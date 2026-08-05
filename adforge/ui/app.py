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
from urllib.parse import quote, urlparse

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
from ..platforms import credhelp
from ..platforms.registry import (ADAPTERS, account_status, is_dry,
                                  publish_post)
from ..platforms.spec import SPECS, VIDEO_FIRST, spec
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


from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(RequestValidationError)
async def _form_errors(request: Request, exc: RequestValidationError):
    """Render form validation failures as readable text, not raw JSON.

    FastAPI's default is a JSON blob like
    {"detail":[{"type":"missing","loc":["body","target"],...}]}
    which is shown to whoever submitted the form. That is unusable: it names
    the field but not what to do, and it replaces the page.
    """
    missing, bad = [], []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", []) if x not in ("body", "query")]
        field = ".".join(loc) or "a field"
        if err.get("type") == "missing":
            missing.append(field)
        else:
            bad.append(f"{field}: {err.get('msg', 'invalid')}")

    if request.method == "GET" or "text/html" not in request.headers.get("accept", ""):
        return JSONResponse({"detail": exc.errors()}, status_code=422)

    parts = []
    if missing:
        parts.append("Required field(s) not received: " + ", ".join(missing))
    if bad:
        parts.append("; ".join(bad))
    log.warning("form validation failed on %s: %s", request.url.path, exc.errors())
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8>"
        "<style>body{font:15px system-ui;background:#0b0f14;color:#e2e8f0;"
        "padding:2.5rem;max-width:44rem}a{color:#06b6d4}code{color:#8fa3bd}</style>"
        f"<h2>That form could not be saved</h2><p>{'. '.join(parts)}.</p>"
        "<p>If the field looked filled in, the browser did not send it - most "
        "often because the page was left open across a restart. Reload the page "
        "and try again.</p>"
        f'<p><a href="{request.headers.get("referer", "/")}">Back</a></p>',
        status_code=422,
    )


def _validate_setting(annotation, raw: str, field: str = "") -> None:
    """Raise if `raw` cannot become `annotation`. Mirrors what boot will do."""
    if field == "timezone":
        # planner.tz() catches a bad zone and silently returns UTC, so a typo
        # here shifts every scheduled time by hours while the field goes on
        # displaying the value the user typed as though it were in force.
        import zoneinfo

        if raw not in zoneinfo.available_timezones():
            raise ValueError(f"unknown IANA timezone {raw!r}")
        return
    if annotation is bool:
        return  # checkboxes are handled separately and are always well-formed
    if annotation in (int, float):
        annotation(raw)  # raises ValueError on '' or 'abc'
    # str and anything else: pydantic accepts it, so we do too.


def _error_page(request: Request, title: str, detail: str, code: int) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8>"
        "<style>body{font:15px system-ui;background:#0b0f14;color:#e2e8f0;"
        "padding:2.5rem;max-width:44rem}a{color:#06b6d4}</style>"
        f"<h2>{title}</h2><p>{detail}</p>"
        f'<p><a href="{request.headers.get("referer", "/")}">Back</a></p>',
        status_code=code,
    )


@app.exception_handler(StarletteHTTPException)
async def _http_errors(request: Request, exc: StarletteHTTPException):
    """Same treatment for raised HTTPExceptions as for validation errors.

    These are deliberate, well-worded refusals - "this post has no enabled
    account", "401 chars exceeds the X limit of 280" - and every one of them
    was being delivered as a raw JSON body that replaced the page the user was
    working in. The message was already right; only the presentation destroyed
    the context needed to act on it.
    """
    if "text/html" not in request.headers.get("accept", ""):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    titles = {400: "That could not be done", 404: "Not found",
              403: "Refused", 409: "Conflicting change"}
    return _error_page(request, titles.get(exc.status_code, "Something went wrong"),
                       str(exc.detail), exc.status_code)


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


def _target_label(target: str, source: str) -> str:
    """How a radar target is named on screen.

    The template used to print 'r/' ~ target unconditionally, which rendered a
    sitewide target as "r/all". That reads as one literal subreddit, and it
    caused exactly that misreading in practice - a scan that had actually been
    skipped for missing credentials was diagnosed as "it is only searching
    r/all". Sitewide is a different kind of thing from a subreddit and is now
    named as one.
    """
    from ..radar.scan import is_sitewide  # imported lazily, as elsewhere here

    if is_sitewide(target):
        return ("everywhere on Reddit" if source == "reddit"
                else "every channel the bot can read")
    return f"r/{target}" if source == "reddit" else target


templates.env.filters["localtime"] = _localtime
templates.env.globals["target_label"] = _target_label
templates.env.globals["cred_help"] = credhelp.help_for


def _slot_age(value) -> tuple[str, str]:
    """('ok'|'overdue'|'expired', human phrase) for a scheduled time.

    The queue listed these under "Coming up" with a bare "04 Aug 06:26" - no
    year, no relative cue - so a slot 36 hours past its time looked exactly
    like one due this evening. Approving it appeared to work and then the next
    tick rejected it for staleness, which reads as the app losing the post.
    """
    from ..scheduler.engine import MAX_SLOT_AGE_HOURS

    if value is None:
        return "ok", ""
    if getattr(value, "tzinfo", None) is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    hours = (utcnow() - value).total_seconds() / 3600
    if hours <= 0:
        return "ok", ""
    if hours > MAX_SLOT_AGE_HOURS:
        return "expired", (
            f"{hours:.0f}h past its slot (limit {MAX_SLOT_AGE_HOURS}h) - "
            f"publishing will reject this; move the time forward or regenerate")
    return "overdue", f"{hours:.0f}h past its slot"


templates.env.globals["slot_age"] = _slot_age


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
            # An unknown value used to raise ValueError out of the handler and
            # render a 500 for what is only ever a hand-edited query string.
            try:
                q = q.filter(Post.status == PostStatus(status))
            except ValueError:
                raise HTTPException(
                    400, f"{status!r} is not a post status. Valid values: "
                         + ", ".join(p.value for p in PostStatus)) from None
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
    # Validate BEFORE dispatching. The select lists every brand's pillars in
    # optgroups, so picking the wrong brand's pillar is a click away - and it
    # used to raise ValueError inside the worker, where the only trace was a
    # job record on a different page than the one you were redirected to.
    if pillar:
        try:
            get_brand(brand).pillar(pillar)
        except ValueError:
            valid = ", ".join(p.key for p in get_brand(brand).pillars)
            raise HTTPException(
                400, f"{pillar!r} is not a pillar of {brand}. Its pillars are: "
                     f"{valid}. The dropdown lists both brands' pillars, so "
                     f"check the brand selected above it.") from None
    jid = _job(
        f"compose:{brand}/{platform}",
        _compose_job, brand, platform, pillar, angle, bool(with_media), when,
    )
    return RedirectResponse(f"/queue?job={quote(jid)}", status_code=303)


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

        # Why an enabled schedule produces no slots. Empty days_of_week or
        # posts_per_day=0 both save without complaint and both render the same
        # healthy "N/day" pill, and the only symptom is that "Next slots"
        # quietly vanishes - which looks identical to every other cause.
        stalled = {}
        for sched in rows:
            if not sched.enabled or preview.get(sched.id):
                continue
            if not sched.day_list():
                stalled[sched.id] = "no days of the week are ticked"
            elif not int(sched.posts_per_day or 0):
                stalled[sched.id] = "posts per day is 0"
            elif not sched.time_list() and not (sched.window_start and sched.window_end):
                stalled[sched.id] = "no pinned times and no window"
            else:
                stalled[sched.id] = "produces no slots"
        # A schedule with no Account row at all generates posts that can never
        # publish; the existing warning only covered an account that exists and
        # is switched off.
        missing_account = {
            s_.id for s_ in rows
            if s_.enabled and not accounts.get((s_.brand, s_.platform))
        }

        # Warn when the schedules together ask for more than the machine can
        # generate. Slots that cannot be filled fail silently - the planner
        # just runs out of time - so the failure looks like "posts stopped
        # appearing" with nothing to point at.
        daily = 0
        video_daily = 0
        for sched in rows:
            if not sched.enabled:
                continue
            n = min(int(sched.posts_per_day or 0), settings.daily_post_cap)
            daily += n
            if sched.attach_media and sched.platform in VIDEO_FIRST:
                video_daily += n
        # A video post is several renders plus a script, not one generation.
        est_minutes = (daily - video_daily) * settings.minutes_per_post
        est_minutes += video_daily * settings.minutes_per_post * 5
        capacity = {
            "posts": daily,
            "video_posts": video_daily,
            "hours": round(est_minutes / 60, 1),
            "over": est_minutes > 20 * 60,
        }
    return templates.TemplateResponse(
        request, "schedules.html",
        ctx(request, schedules=rows, accounts=accounts, preview=preview,
            stalled=stalled, missing_account=missing_account,
            tz=tz(), capacity=capacity),
    )


# Declared BEFORE /schedules/{sched_id}. FastAPI matches in declaration order,
# so a parameterised route listed first swallows every literal sibling: with
# {sched_id} above, "plan-now" and "tick" were parsed as ids and both buttons
# returned "Input should be a valid integer" instead of doing anything. Neither
# had ever worked. test_no_shadowed_routes guards the ordering.
def _plan_ahead_described() -> str:
    n = plan_ahead()
    if n:
        return f"created {n} post(s)"
    return ("created nothing - every schedule is disabled, already filled for "
            "its window, or capped for today")


def _tick_described() -> str:
    promoted, published = promote_and_publish()
    if not promoted and not published:
        return "nothing was due"
    return f"promoted {promoted}, published {published}"


@app.post("/schedules/plan-now")
def schedule_plan_now():
    # Both of these return the job id in the query string. The work happens on
    # the pool, so the outcome is not known at redirect time - without this the
    # button returned an identical page and "created 5 posts" and "created
    # nothing because everything is disabled" were the same screen.
    jid = _job("plan", _plan_ahead_described)
    return RedirectResponse(f"/schedules?job={quote(jid)}", status_code=303)


@app.post("/schedules/tick")
def schedule_tick():
    jid = _job("publish-tick", _tick_described)
    return RedirectResponse(f"/queue?job={quote(jid)}", status_code=303)


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


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

        was_enabled = bool(acct.enabled)
        acct.handle = str(form.get("handle", ""))
        acct.enabled = bool(form.get("enabled"))
        acct.mode = PostMode(str(form.get("mode", "AUTO")))
        # Tri-state: "" inherits the global dry-run switch.
        dr = str(form.get("dry_run", ""))
        acct.dry_run = None if dr == "" else (dr == "1")

        # Enabling an account must never go live by INHERITANCE.
        #
        # With the global switch off, "inherit" resolves to live - so ticking
        # one checkbox on a form about handles and credentials would start
        # publishing, with no step that was about publishing. Going live should
        # take an act that says so. The account is pinned to dry run instead,
        # and the user can then choose LIVE explicitly from the dropdown, which
        # is a decision rather than a side effect.
        pinned = ""
        if acct.enabled and not was_enabled and acct.dry_run is None \
                and not settings.dry_run:
            acct.dry_run = True
            pinned = f"{acct.brand}/{acct.platform}"
            log.warning(
                "%s/%s enabled while global dry run is off - pinned to dry run; "
                "set Transmission to LIVE explicitly to publish",
                acct.brand, acct.platform,
            )

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
    # The pin is a deliberate safety behaviour, but it silently rewrote a
    # choice the user had made and reported it only to the server log - so the
    # dropdown came back reading "dry run" with no explanation.
    if pinned:
        return RedirectResponse(f"/accounts?pinned={quote(pinned)}", status_code=303)
    return RedirectResponse("/accounts", status_code=303)


@app.post("/accounts/{account_id}/clear-credential")
def account_clear_credential(account_id: int, name: str = Form(...)):
    with session_scope() as s:
        acct = s.get(Account, account_id)
        if not acct:
            raise HTTPException(404, "no such account")
        creds = acct.creds()
        existed = creds.pop(name, None) is not None
        acct.credentials = json.dumps(creds)
        label = f"{acct.brand}/{acct.platform}"
    # Say what happened. A bare redirect back to an identical page is the
    # failure mode this whole endpoint was unreachable behind: deleting a
    # credential and deleting nothing looked exactly the same.
    outcome = "cleared" if existed else "was-not-set"
    return RedirectResponse(
        f"/accounts?{outcome}={quote(name)}&on={quote(label)}", status_code=303)


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

        # Why a scan found nothing. "Scan now" returns 303 and re-renders the
        # page whether it scanned or skipped every target, so a missing
        # credential looked exactly like a scan that found no matches. The
        # reason was only in the log.
        blockers = []
        need = {"reddit": ("client_id", "client_secret", "username", "password"),
                "discord": ("bot_token",)}
        enabled_targets = [t_ for t_ in targets if t_.enabled]
        for tgt in enabled_targets:
            acct = (
                s.query(Account)
                .filter_by(brand=tgt.brand, platform=tgt.source)
                .first()
            )
            if acct is None:
                blockers.append(f"{tgt.brand}/{tgt.source}: no account exists")
                continue
            creds = acct.creds()
            missing = [k for k in need.get(tgt.source, ()) if not str(creds.get(k, "")).strip()]
            if missing:
                blockers.append(
                    f"{tgt.brand}/{tgt.source}: no credentials yet "
                    f"({', '.join(missing)})"
                )
            # NOT a blocker: scan_all never consults account.enabled, so a
            # disabled account scans perfectly well. Reporting it sent the user
            # to Accounts to fix something that was not stopping anything.
            elif not tgt.keyword_list():
                blockers.append(f"{tgt.brand}/{tgt.source}: no keywords")
        blockers = sorted(set(blockers))
        # How many of the enabled targets are actually blocked. The banner used
        # to say "Nothing will be scanned" whenever this list was non-empty,
        # which is wrong and alarming when two of three targets are healthy.
        blocked_n, target_n = len(blockers), len(enabled_targets)
        # Automatic scanning being off is not a per-target problem, but it is
        # the reason a user who waits for the next pass waits forever.
        auto_off = not settings.radar_enabled
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
            blockers=blockers, blocked_n=blocked_n, target_n=target_n,
            auto_off=auto_off,
            f_brand=brand, f_min=min_rel),
    )


@app.post("/radar/targets/new")
def radar_target_new(
    brand: str = Form(...),
    source: str = Form(...),
    # Empty-but-present is the common case (the user left it blank), and it
    # deserves a sentence rather than a validation blob.
    target: str = Form(""),
    keywords: str = Form(""),
    promo_allowed: str = Form(""),
    rules_note: str = Form(""),
    min_relevance: float = Form(0.6),
):
    cleaned = target.strip().lstrip("/").removeprefix("r/")
    if not cleaned:
        raise HTTPException(
            400,
            "Target is required. Use 'all' to search the whole of Reddit, "
            "'*' for every Discord channel your bot can read, or name one "
            "subreddit (without the r/) or one guild_id/channel_id.",
        )
    # "I have read this community's rules" cannot be true of every community.
    # scan.py already forces links off for sitewide hits, but storing the flag
    # as True records a promise nobody could have made and would quietly become
    # load-bearing if that scan logic ever changed.
    from ..radar.scan import is_sitewide

    if is_sitewide(cleaned) and promo_allowed:
        promo_allowed = ""
        log.info("sitewide target %r: ignoring the rules-read tick", cleaned)

    with session_scope() as s:
        # Two identical cards are indistinguishable on screen and double both
        # the scan work and the platform's rate-limit budget for no benefit.
        dupe = (s.query(RadarTarget)
                 .filter(RadarTarget.brand == brand,
                         RadarTarget.source == source,
                         RadarTarget.target == cleaned)
                 .first())
        if dupe:
            raise HTTPException(
                400,
                f"{brand}/{source} already has a target for {cleaned!r} "
                f"(#{dupe.id}). Edit that one instead - a second copy would "
                f"scan the same thing twice and spend twice the rate limit.")
        s.add(RadarTarget(
            brand=brand, source=source, target=cleaned,
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
            from ..radar.scan import is_sitewide

            t.keywords = keywords
            t.enabled = bool(enabled)
            # The create form refuses this tick on a sitewide target; the edit
            # form did not, so the flag could be set afterwards and the card
            # then rendered a rules-read promise nobody could have made.
            t.promo_allowed = bool(promo_allowed) and not is_sitewide(t.target)
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
    # min_relevance was stored, rendered and editable but read by nothing -
    # the module docstring claimed it gated drafting and it did not. Enforce
    # the documented behaviour here, where the click happens, so the number
    # does what the label says.
    with session_scope() as s:
        thread = s.get(RadarThread, thread_id)
        if not thread:
            raise HTTPException(404, "no such thread")
        target = (s.query(RadarTarget)
                   .filter(RadarTarget.brand == thread.brand,
                           RadarTarget.source == thread.source)
                   .first())
        if target and thread.relevance < target.min_relevance:
            raise HTTPException(
                400,
                f"This thread scored {thread.relevance:.2f}, below the "
                f"{target.min_relevance:.2f} minimum relevance set on its "
                f"target. Lower that threshold if you want to reply here.")
    jid = _job(f"draft#{thread_id}", _draft_job, thread_id)
    return RedirectResponse(f"/radar?job={quote(jid)}", status_code=303)


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


# Pillars whose posts assert things no gate can check - hardware behaviour,
# inference characteristics, cost arithmetic. The factcheck flags what it
# believes is wrong, but that is one model's opinion of another's output.
UNATTENDED_RISK_PILLARS = {"tips", "cost", "ai_news"}


def _unattended() -> list[str]:
    """Destinations that would publish with no human involvement at all.

    Enabled account, AUTO mode, actually transmitting, and a review window of
    zero. The combination is legitimate - it is what hands-off means - but each
    setting lives on a different page, so nothing shows the four of them
    lining up.
    """
    if settings.review_window_minutes > 0:
        return []
    out = []
    with session_scope() as s:
        for sched in s.query(Schedule).filter(Schedule.enabled.is_(True)).all():
            acct = (
                s.query(Account)
                .filter_by(brand=sched.brand, platform=sched.platform)
                .first()
            )
            if not (acct and acct.enabled and acct.mode == PostMode.AUTO):
                continue
            if is_dry(acct):
                continue
            pillars = set(sched.pillar_list()) or {
                p.key for p in get_brand(sched.brand).pillars
            }
            risky = pillars & UNATTENDED_RISK_PILLARS
            label = f"{sched.brand}/{sched.platform}"
            if risky:
                label += f" (incl. {', '.join(sorted(risky))})"
            out.append(label)
    return out


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    from ..llm.client import available_models

    with session_scope() as s:
        overrides = {row.key: row.value for row in s.query(Setting).all()}
    return templates.TemplateResponse(
        request, "settings.html",
        ctx(request, overrides=overrides, models=available_models(),
            unattended=_unattended(),
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

    # Validate against the Settings field types BEFORE writing. `settings =
    # Settings()` runs at import, so an unparseable value here does not
    # produce a bad setting - it stops the process from starting at all, on
    # the next restart, after this page has said "Saved." Clearing a number
    # box is enough to do it: an empty string is not an int.
    from ..config import Settings

    fields = Settings.model_fields
    rejected = []
    staged = {}
    for key, value in form.items():
        if not key.startswith("set_"):
            continue
        field = key[4:]
        raw = str(value)
        if field in fields:
            try:
                _validate_setting(fields[field].annotation, raw, field)
            except ValueError as e:
                rejected.append(f"{field}: {e}")
                continue
            except TypeError:
                rejected.append(
                    f"{field}: {raw!r} is not a valid "
                    f"{getattr(fields[field].annotation, '__name__', 'value')}")
                continue
        staged[f"ADFORGE_{field.upper()}"] = _env_safe(raw)

    if rejected:
        return _error_page(
            request, "Nothing was saved",
            "These values were refused, so the whole form was left alone "
            "rather than saving it half-applied: <br><br>"
            + "<br>".join(rejected)
            + "<br><br>An unparseable setting does not merely misbehave - it "
              "stops AdForge starting next time, which is why this refuses "
              "instead of writing it.", 400)
    existing.update(staged)

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
