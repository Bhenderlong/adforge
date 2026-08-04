"""
SQLite schema and session handling.

Status flow for a Post:
    DRAFT -> REVIEW -> APPROVED -> PUBLISHED
                    \\-> REJECTED
                       -> FAILED (retryable)

REVIEW is where the review-window timer runs. On accounts set to auto-post the
scheduler promotes REVIEW -> APPROVED once the window expires; on manual
accounts it waits for a human.
"""

from __future__ import annotations

import datetime as dt
import enum
import json
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from .config import settings


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class PostStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    PUBLISHED = "PUBLISHED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class PostMode(str, enum.Enum):
    AUTO = "AUTO"  # publishes when the review window expires
    MANUAL = "MANUAL"  # requires an explicit human approval


class JSONColumn(Text):
    """Plain JSON in a TEXT column - avoids a JSON1 dependency."""


def _dumps(v) -> str:
    return json.dumps(v, default=str)


class Account(Base):
    """One connected destination: a brand's presence on one platform."""

    __tablename__ = "accounts"
    __table_args__ = (UniqueConstraint("brand", "platform", "handle"),)

    id = Column(Integer, primary_key=True)
    brand = Column(String(32), nullable=False, index=True)
    platform = Column(String(32), nullable=False, index=True)
    handle = Column(String(128), default="")
    enabled = Column(Boolean, default=False)
    # Per-account override of the global dry_run. None = inherit global.
    dry_run = Column(Boolean, nullable=True)
    mode = Column(Enum(PostMode), default=PostMode.AUTO)
    # Credentials are stored here rather than in .env so the UI can manage
    # several accounts per platform. File is chmod 600 by run.sh.
    credentials = Column(Text, default="{}")
    # Platform-specific config: subreddit allowlist, discord channel ids, etc.
    options = Column(Text, default="{}")
    created_at = Column(DateTime, default=utcnow)

    def creds(self) -> dict:
        return json.loads(self.credentials or "{}")

    def opts(self) -> dict:
        return json.loads(self.options or "{}")


class Schedule(Base):
    """A posting cadence for one brand+platform."""

    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True)
    brand = Column(String(32), nullable=False, index=True)
    platform = Column(String(32), nullable=False, index=True)
    enabled = Column(Boolean, default=True)
    posts_per_day = Column(Integer, default=1)
    # "0,1,2,3,4" = Mon-Fri. Python weekday numbering.
    days_of_week = Column(String(32), default="0,1,2,3,4")
    # Explicit local times "09:15,13:40,17:05". If fewer entries than
    # posts_per_day, the remainder are spread across active_window.
    times = Column(String(256), default="09:00")
    # Fallback window when times are not pinned.
    window_start = Column(String(8), default="09:00")
    window_end = Column(String(8), default="17:00")
    # +/- minutes of randomness so a feed does not fire on the exact minute
    # every day, which reads as robotic to humans as much as to platforms.
    jitter_minutes = Column(Integer, default=12)
    # Restrict to specific content pillars; empty = all, weighted.
    pillars = Column(String(256), default="")
    attach_media = Column(Boolean, default=True)

    def day_list(self) -> list[int]:
        return [int(d) for d in self.days_of_week.split(",") if d.strip() != ""]

    def time_list(self) -> list[str]:
        return [t.strip() for t in self.times.split(",") if t.strip()]

    def pillar_list(self) -> list[str]:
        return [p.strip() for p in self.pillars.split(",") if p.strip()]


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    brand = Column(String(32), nullable=False, index=True)
    platform = Column(String(32), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)
    pillar = Column(String(48), default="")

    status = Column(Enum(PostStatus), default=PostStatus.DRAFT, index=True)
    mode = Column(Enum(PostMode), default=PostMode.AUTO)

    title = Column(String(400), default="")  # reddit/youtube need one
    body = Column(Text, default="")
    # X threads and IG carousels: extra parts after the first.
    extra_parts = Column(Text, default="[]")
    hashtags = Column(String(400), default="")
    link = Column(String(400), default="")

    image_path = Column(String(600), default="")
    video_path = Column(String(600), default="")
    media_prompt = Column(Text, default="")

    scheduled_for = Column(DateTime, index=True)
    review_until = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)

    remote_id = Column(String(256), default="")
    remote_url = Column(String(600), default="")
    error = Column(Text, default="")
    attempts = Column(Integer, default=0)

    # Critic scoring, kept for tuning the generator over time.
    quality_score = Column(Float, default=0.0)
    critic_notes = Column(Text, default="")
    generation_meta = Column(Text, default="{}")

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    def parts(self) -> list[str]:
        return json.loads(self.extra_parts or "[]")


class RadarThread(Base):
    """A discovered conversation worth engaging with."""

    __tablename__ = "radar_threads"
    __table_args__ = (UniqueConstraint("source", "external_id"),)

    id = Column(Integer, primary_key=True)
    brand = Column(String(32), index=True)
    source = Column(String(16))  # reddit | discord
    external_id = Column(String(128))
    where = Column(String(200))  # r/LocalLLaMA | guild#channel
    author = Column(String(128), default="")
    title = Column(String(600), default="")
    excerpt = Column(Text, default="")
    url = Column(String(700), default="")
    posted_at = Column(DateTime, nullable=True)

    # Scoring: is this genuinely a place where the product helps?
    relevance = Column(Float, default=0.0)
    score_reason = Column(Text, default="")
    matched_keywords = Column(String(400), default="")
    # Set when the sub/channel rules forbid promotion - draft is suppressed.
    promo_allowed = Column(Boolean, default=True)
    rules_note = Column(String(400), default="")

    seen_at = Column(DateTime, default=utcnow)
    dismissed = Column(Boolean, default=False)

    drafts = relationship("ReplyDraft", back_populates="thread")


class ReplyDraft(Base):
    """A drafted reply. ALWAYS requires human approval before it is sent."""

    __tablename__ = "reply_drafts"

    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey("radar_threads.id"), index=True)
    brand = Column(String(32), index=True)
    body = Column(Text, default="")
    # Stored separately so it cannot be edited away accidentally in the UI;
    # the adapter re-appends it at send time regardless.
    disclosure = Column(String(400), default="")
    status = Column(Enum(PostStatus), default=PostStatus.REVIEW, index=True)
    quality_score = Column(Float, default=0.0)
    critic_notes = Column(Text, default="")
    posted_at = Column(DateTime, nullable=True)
    remote_url = Column(String(700), default="")
    error = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)

    thread = relationship("RadarThread", back_populates="drafts")


class RadarTarget(Base):
    """Where the radar looks, and for what. Configured in the UI."""

    __tablename__ = "radar_targets"

    id = Column(Integer, primary_key=True)
    brand = Column(String(32), index=True)
    source = Column(String(16))  # reddit | discord
    # subreddit name, or "guild_id/channel_id"
    target = Column(String(200))
    keywords = Column(Text, default="")  # comma separated
    enabled = Column(Boolean, default=True)
    # Whether ANY link/product mention is permitted here. Off by default for
    # subreddits until the user confirms the sub's self-promotion rules.
    promo_allowed = Column(Boolean, default=False)
    rules_note = Column(String(400), default="")
    min_relevance = Column(Float, default=0.6)

    def keyword_list(self) -> list[str]:
        return [k.strip().lower() for k in self.keywords.split(",") if k.strip()]


class Metric(Base):
    """Post-publication stats, pulled back on a slow loop."""

    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True)
    captured_at = Column(DateTime, default=utcnow)
    impressions = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    clicks = Column(Integer, default=0)


class Setting(Base):
    """UI-editable runtime settings that override config defaults."""

    __tablename__ = "settings"
    key = Column(String(64), primary_key=True)
    value = Column(Text, default="")


_engine = create_engine(
    settings.db_url,
    connect_args={
        "check_same_thread": False,
        # The UI, the scheduler thread pool and a separate `--worker` process
        # all write this file. Without a busy timeout, a concurrent write
        # raises "database is locked" immediately and the losing transaction
        # is rolled back - which shows up as a generated post silently never
        # appearing in the queue.
        "timeout": 30,
    },
    future=True,
)


@event.listens_for(_engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    """WAL + a busy timeout, set on every connection.

    Default SQLite journalling takes an exclusive lock for the whole write
    transaction, so a reader blocks a writer and two writers conflict. WAL lets
    readers continue during a write and is what makes multi-process access
    (`run.py` plus `run.py --worker`) safe rather than merely usually-fine.
    """
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(_engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
