"""Adapter registry and the single publish entry point."""

from __future__ import annotations

import logging
from pathlib import Path

from ..brands import get_brand
from ..config import settings
from ..db import Account, Post, PostStatus, utcnow
from .base import Payload, PublishError, Result
from .community import RedditAdapter, TikTokAdapter, YouTubeAdapter
from .social import (
    DiscordAdapter,
    FacebookAdapter,
    InstagramAdapter,
    LinkedInAdapter,
    SlackAdapter,
    XAdapter,
)
from .spec import spec

log = logging.getLogger("adforge.publish")

ADAPTERS = {
    a.key: a
    for a in (
        XAdapter(), LinkedInAdapter(), FacebookAdapter(), InstagramAdapter(),
        TikTokAdapter(), YouTubeAdapter(), RedditAdapter(), DiscordAdapter(),
        SlackAdapter(),
    )
}


def adapter_for(platform: str):
    try:
        return ADAPTERS[platform]
    except KeyError:
        raise PublishError(f"no adapter for platform {platform!r}") from None


def payload_from(post: Post, account: Account | None) -> Payload:
    brand = get_brand(post.brand)
    opts = account.opts() if account else {}
    tags = [t.strip() for t in (post.hashtags or "").split(",") if t.strip()]
    return Payload(
        brand=post.brand,
        platform=post.platform,
        body=post.body,
        title=post.title,
        link=post.link or brand.url,
        image=Path(post.image_path) if post.image_path else None,
        video=Path(post.video_path) if post.video_path else None,
        extra_parts=post.parts(),
        options={**opts, "tags": tags},
    )


def is_dry(account: Account | None) -> bool:
    """Per-account override wins; otherwise the global switch.

    Defaults to dry when there is no account at all - an unconfigured
    destination must never transmit.
    """
    if account is None:
        return True
    if account.dry_run is not None:
        return bool(account.dry_run)
    return bool(settings.dry_run)


def publish_post(post: Post, account: Account | None) -> Result:
    """Publish one post. Mutates `post` with the outcome; caller commits."""
    ps = spec(post.platform)
    adapter = adapter_for(post.platform)
    payload = payload_from(post, account)
    dry = is_dry(account)

    # Final gate. Copy is checked at generation time, but it can be edited in
    # the UI afterwards, so the last word on format belongs here.
    if len(payload.body) > ps.max_chars:
        raise PublishError(
            f"body is {len(payload.body)} chars, over the {ps.label} limit of "
            f"{ps.max_chars} - edit it before publishing"
        )

    # Refuse to send anything that already has a remote id.
    #
    # A 5xx or a read timeout is classed retryable and re-queued, but the
    # platform may well have accepted the post before the connection broke.
    # Where the adapter got far enough to record an id, retrying would publish
    # a second copy. Without platform idempotency keys this cannot be made
    # airtight, but it closes the case we can actually detect - and a duplicate
    # post is worse than a missing one, because only the missing one is fixable
    # from the queue.
    if post.remote_id and not dry:
        raise PublishError(
            f"already published as {post.remote_id} - refusing to send again. "
            f"If that post is genuinely missing, clear the remote id first.",
            retryable=False,
        )

    # `attempts` is incremented by the claim in scheduler.engine.claim_post,
    # which is what makes the retry cap meaningful across processes. Doing it
    # again here would double-count and halve the effective cap.
    try:
        creds = account.creds() if account else {}
        result = adapter.publish(payload, creds, dry)
    except PublishError as e:
        post.error = str(e)
        post.status = PostStatus.FAILED if e.retryable else PostStatus.REJECTED
        log.error("%s/%s publish failed (retryable=%s): %s",
                  post.brand, post.platform, e.retryable, e)
        raise

    post.error = ""
    post.remote_id = result.remote_id
    post.remote_url = result.url
    if not result.dry_run:
        post.status = PostStatus.PUBLISHED
        post.published_at = utcnow()
    log.info("%s/%s -> %s %s", post.brand, post.platform,
             "DRY RUN" if result.dry_run else "PUBLISHED", result.url or result.detail)
    return result


def account_status(account: Account) -> tuple[bool, str]:
    """Whether an account is ready to publish, for the UI."""
    try:
        adapter = adapter_for(account.platform)
    except PublishError as e:
        return False, str(e)
    problems = adapter.validate(account.creds(), account.opts())
    if problems:
        return False, "; ".join(problems)
    return True, "dry run" if is_dry(account) else "live"
