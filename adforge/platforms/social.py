"""
Adapters for X, LinkedIn, Facebook, Instagram, Slack and Discord.

All official APIs. Each one documents the specific quirk that bites, because
these are the failures that cost hours: LinkedIn's two-step asset registration,
Instagram's requirement that media be fetched from a public URL, Facebook's
separate page token.
"""

from __future__ import annotations

import contextlib
import json
import logging
import mimetypes
import time
from pathlib import Path

import httpx

from .base import BaseAdapter, NotConfigured, Payload, PublishError, Result

log = logging.getLogger("adforge.publish")


def _raise_for(r: httpx.Response, what: str) -> dict:
    if r.status_code >= 400:
        # 429 and 5xx are worth retrying later; 4xx generally is not.
        retryable = r.status_code == 429 or r.status_code >= 500
        raise PublishError(f"{what} failed [{r.status_code}]: {r.text[:500]}", retryable)
    try:
        return r.json()
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# X
# ---------------------------------------------------------------------------


class XAdapter(BaseAdapter):
    """X API v2 for posting, v1.1 for media upload.

    Media still has to go through the v1.1 upload endpoint - v2 has no
    equivalent - so the OAuth 1.0a user context credentials are required even
    though the tweet itself is a v2 call.
    """

    key = "x"
    required_credentials = (
        "api_key", "api_secret", "access_token", "access_token_secret",
    )
    min_interval_seconds = 120

    def _session(self, creds: dict):
        try:
            from requests_oauthlib import OAuth1Session
        except ImportError as e:
            raise NotConfigured("requests_oauthlib not installed") from e
        return OAuth1Session(
            creds["api_key"],
            client_secret=creds["api_secret"],
            resource_owner_key=creds["access_token"],
            resource_owner_secret=creds["access_token_secret"],
        )

    def _upload_media(self, sess, path: Path) -> str:
        with path.open("rb") as fh:
            r = sess.post(
                "https://upload.twitter.com/1.1/media/upload.json",
                files={"media": fh},
            )
        if r.status_code >= 400:
            raise PublishError(f"X media upload failed: {r.text[:400]}")
        return str(r.json()["media_id_string"])

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        if problems := self.validate(creds, payload.options):
            raise NotConfigured("; ".join(problems))
        if dry_run:
            return self.dry(payload, f"thread parts: {len(payload.extra_parts)}")

        sess = self._session(creds)
        media_ids = []
        media = payload.video or payload.image
        if media and Path(media).exists():
            media_ids.append(self._upload_media(sess, Path(media)))

        self.throttle(f"x:{creds['access_token'][:12]}")

        body: dict = {"text": payload.body}
        if media_ids:
            body["media"] = {"media_ids": media_ids}
        r = sess.post("https://api.x.com/2/tweets", json=body)
        if r.status_code >= 400:
            raise PublishError(
                f"X post failed [{r.status_code}]: {r.text[:400]}",
                retryable=r.status_code in (429, 500, 502, 503),
            )
        data = r.json()["data"]
        root_id = data["id"]

        # Thread continuation.
        reply_to = root_id
        for part in payload.extra_parts:
            time.sleep(3)
            rr = sess.post(
                "https://api.x.com/2/tweets",
                json={"text": part, "reply": {"in_reply_to_tweet_id": reply_to}},
            )
            if rr.status_code >= 400:
                log.warning("thread part failed, stopping: %s", rr.text[:200])
                break
            reply_to = rr.json()["data"]["id"]

        return Result(True, root_id, f"https://x.com/i/web/status/{root_id}")


# ---------------------------------------------------------------------------
# LinkedIn
# ---------------------------------------------------------------------------


class LinkedInAdapter(BaseAdapter):
    """LinkedIn UGC Posts API.

    Media upload is two-step: registerUpload returns a one-time URL that the
    bytes are PUT to, and only then can the asset URN be referenced in a post.
    Posting the URN before the upload completes yields a post with a broken
    image and no error.

    `author_urn` is either "urn:li:person:<id>" or "urn:li:organization:<id>";
    company pages need w_organization_social scope, not w_member_social.
    """

    key = "linkedin"
    required_credentials = ("access_token", "author_urn")
    min_interval_seconds = 300
    API = "https://api.linkedin.com/v2"

    def _headers(self, creds: dict) -> dict:
        return {
            "Authorization": f"Bearer {creds['access_token']}",
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json",
        }

    def _upload_asset(self, client: httpx.Client, creds: dict, path: Path) -> str:
        recipe = (
            "urn:li:digitalmediaRecipe:feedshare-video"
            if path.suffix.lower() in (".mp4", ".mov")
            else "urn:li:digitalmediaRecipe:feedshare-image"
        )
        reg = _raise_for(
            client.post(
                f"{self.API}/assets?action=registerUpload",
                headers=self._headers(creds),
                json={
                    "registerUploadRequest": {
                        "recipes": [recipe],
                        "owner": creds["author_urn"],
                        "serviceRelationships": [
                            {
                                "relationshipType": "OWNER",
                                "identifier": "urn:li:userGeneratedContent",
                            }
                        ],
                    }
                },
            ),
            "linkedin registerUpload",
        )
        value = reg["value"]
        url = value["uploadMechanism"][
            "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
        ]["uploadUrl"]
        # Stream from the file rather than materialising it. A 40s vertical
        # short is hundreds of MB, and this process also serves the web UI and
        # runs a 2-worker job pool.
        with path.open("rb") as fh:
            up = client.put(
                url,
                content=fh,
                headers={"Authorization": f"Bearer {creds['access_token']}"},
                timeout=600,
            )
        if up.status_code >= 400:
            raise PublishError(f"linkedin asset upload failed: {up.status_code}")
        return value["asset"]

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        if problems := self.validate(creds, payload.options):
            raise NotConfigured("; ".join(problems))
        if dry_run:
            return self.dry(payload, "link goes in the first comment")

        with httpx.Client(timeout=120) as client:
            media = payload.video or payload.image
            asset, media_type = None, "NONE"
            if media and Path(media).exists():
                asset = self._upload_asset(client, creds, Path(media))
                media_type = "VIDEO" if Path(media).suffix.lower() in (".mp4", ".mov") else "IMAGE"

            content: dict = {
                "shareCommentary": {"text": payload.body},
                "shareMediaCategory": media_type,
            }
            if asset:
                content["media"] = [{"status": "READY", "media": asset}]

            self.throttle(f"li:{creds['author_urn']}")
            data = _raise_for(
                client.post(
                    f"{self.API}/ugcPosts",
                    headers=self._headers(creds),
                    json={
                        "author": creds["author_urn"],
                        "lifecycleState": "PUBLISHED",
                        "specificContent": {
                            "com.linkedin.ugc.ShareContent": content
                        },
                        "visibility": {
                            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                        },
                    },
                ),
                "linkedin ugcPosts",
            )
            post_id = data.get("id", "")

            # The URL lives in the first comment: LinkedIn demotes posts with a
            # link in the body, so the copy is generated without one.
            if payload.link and post_id:
                try:
                    client.post(
                        f"{self.API}/socialActions/{post_id}/comments",
                        headers=self._headers(creds),
                        json={
                            "actor": creds["author_urn"],
                            "message": {"text": payload.link},
                        },
                    )
                except Exception as e:  # noqa: BLE001 - post already succeeded
                    log.warning("linkedin first comment failed: %s", e)

        return Result(True, post_id, f"https://www.linkedin.com/feed/update/{post_id}")


# ---------------------------------------------------------------------------
# Facebook / Instagram (Meta Graph API)
# ---------------------------------------------------------------------------


class FacebookAdapter(BaseAdapter):
    """Facebook Page posting.

    Requires a PAGE access token, not a user token - a user token returns 200
    on /me but 403 on the page's /feed. Get it from
    /me/accounts and store the page-specific `access_token`.
    """

    key = "facebook"
    required_credentials = ("page_id", "page_access_token")
    min_interval_seconds = 300
    GRAPH = "https://graph.facebook.com/v21.0"

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        if problems := self.validate(creds, payload.options):
            raise NotConfigured("; ".join(problems))
        if dry_run:
            return self.dry(payload)

        pid, token = creds["page_id"], creds["page_access_token"]
        self.throttle(f"fb:{pid}")
        with httpx.Client(timeout=600) as client:
            if payload.video and Path(payload.video).exists():
                with Path(payload.video).open("rb") as fh:
                    data = _raise_for(
                        client.post(
                            f"{self.GRAPH}/{pid}/videos",
                            data={"description": payload.body, "access_token": token},
                            files={"source": fh},
                        ),
                        "facebook video",
                    )
            elif payload.image and Path(payload.image).exists():
                with Path(payload.image).open("rb") as fh:
                    data = _raise_for(
                        client.post(
                            f"{self.GRAPH}/{pid}/photos",
                            data={"message": payload.body, "access_token": token},
                            files={"source": fh},
                        ),
                        "facebook photo",
                    )
            else:
                data = _raise_for(
                    client.post(
                        f"{self.GRAPH}/{pid}/feed",
                        data={"message": payload.body, "link": payload.link or None,
                              "access_token": token},
                    ),
                    "facebook feed",
                )
        rid = str(data.get("post_id") or data.get("id", ""))
        return Result(True, rid, f"https://facebook.com/{rid}")


class InstagramAdapter(BaseAdapter):
    """Instagram Graph API (Business/Creator accounts only).

    The API cannot accept a file upload. Media must be at a PUBLIC HTTPS URL
    that Meta's servers fetch themselves, so `public_media_base` must point at
    somewhere internet-reachable that serves the assets directory. On this
    setup that is an nginx location on the Dell, not localhost.
    """

    key = "instagram"
    required_credentials = ("ig_user_id", "access_token", "public_media_base")
    min_interval_seconds = 600
    GRAPH = "https://graph.facebook.com/v21.0"

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        if problems := self.validate(creds, payload.options):
            raise NotConfigured("; ".join(problems))

        media = payload.video or payload.image
        if not media or not Path(media).exists():
            raise PublishError("instagram requires an image or video")

        base = str(creds["public_media_base"]).rstrip("/")
        media_url = f"{base}/{Path(media).name}"
        if dry_run:
            return self.dry(payload, f"would tell Meta to fetch {media_url}")

        if not base.startswith("https://"):
            raise PublishError(
                "public_media_base must be an https:// URL Meta can reach; "
                f"got {base!r}"
            )

        uid, token = creds["ig_user_id"], creds["access_token"]
        is_video = Path(media).suffix.lower() in (".mp4", ".mov")
        params = {
            "caption": payload.body,
            "access_token": token,
            **({"media_type": "REELS", "video_url": media_url} if is_video
               else {"image_url": media_url}),
        }

        self.throttle(f"ig:{uid}")
        with httpx.Client(timeout=300) as client:
            container = _raise_for(
                client.post(f"{self.GRAPH}/{uid}/media", data=params),
                "instagram container",
            )
            cid = container["id"]

            # Video containers are processed asynchronously; publishing before
            # status_code is FINISHED returns a misleading "media not ready".
            if is_video:
                for _ in range(60):
                    time.sleep(5)
                    st = _raise_for(
                        client.get(f"{self.GRAPH}/{cid}",
                                   params={"fields": "status_code",
                                           "access_token": token}),
                        "instagram status",
                    )
                    code = st.get("status_code")
                    if code == "FINISHED":
                        break
                    if code == "ERROR":
                        raise PublishError("instagram rejected the video")
                else:
                    raise PublishError("instagram video processing timed out", True)

            pub = _raise_for(
                client.post(f"{self.GRAPH}/{uid}/media_publish",
                            data={"creation_id": cid, "access_token": token}),
                "instagram publish",
            )
        rid = str(pub.get("id", ""))
        return Result(True, rid, f"https://www.instagram.com/p/{rid}")


# ---------------------------------------------------------------------------
# Slack / Discord
# ---------------------------------------------------------------------------


class SlackAdapter(BaseAdapter):
    key = "slack"
    required_credentials = ("bot_token", "channel")
    min_interval_seconds = 30

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        if problems := self.validate(creds, payload.options):
            raise NotConfigured("; ".join(problems))
        if dry_run:
            return self.dry(payload)

        token, channel = creds["bot_token"], creds["channel"]
        text = payload.body + (f"\n{payload.link}" if payload.link else "")
        with httpx.Client(timeout=120) as client:
            media = payload.video or payload.image
            if media and Path(media).exists():
                # files.upload is deprecated; the external-upload flow is the
                # supported path and is required for new apps.
                p = Path(media)
                ext = _raise_for(
                    client.get(
                        "https://slack.com/api/files.getUploadURLExternal",
                        params={"filename": p.name, "length": p.stat().st_size},
                        headers={"Authorization": f"Bearer {token}"},
                    ),
                    "slack upload url",
                )
                if not ext.get("ok"):
                    raise PublishError(f"slack upload url: {ext.get('error')}")
                with p.open("rb") as fh:
                    client.post(ext["upload_url"], content=fh, timeout=600)
                done = _raise_for(
                    client.post(
                        "https://slack.com/api/files.completeUploadExternal",
                        headers={"Authorization": f"Bearer {token}"},
                        json={"files": [{"id": ext["file_id"], "title": p.name}],
                              "channel_id": channel, "initial_comment": text},
                    ),
                    "slack complete upload",
                )
                if not done.get("ok"):
                    raise PublishError(f"slack upload: {done.get('error')}")
                return Result(True, ext["file_id"], "")

            data = _raise_for(
                client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "text": text},
                ),
                "slack postMessage",
            )
        if not data.get("ok"):
            raise PublishError(f"slack: {data.get('error')}")
        return Result(True, str(data.get("ts", "")), "")


class DiscordAdapter(BaseAdapter):
    """Discord via webhook (simplest) or bot token.

    A webhook needs no bot, no gateway connection and no privileged intents -
    for posting announcements to a channel you own it is strictly simpler. The
    bot token path exists because the radar's reply flow needs it.
    """

    key = "discord"
    required_credentials = ()
    min_interval_seconds = 15

    def validate(self, creds: dict, options: dict) -> list[str]:
        if url := creds.get("webhook_url"):
            # The webhook URL is the entire destination and is dialled
            # verbatim, so an arbitrary value turns this adapter into an
            # HTTP POST primitive against anything the host can reach -
            # including loopback services and cloud metadata endpoints.
            if not str(url).startswith("https://discord.com/api/webhooks/"):
                return [
                    "webhook_url must start with "
                    "https://discord.com/api/webhooks/"
                ]
            return []
        if creds.get("bot_token") and creds.get("channel_id"):
            return []
        return ["need either webhook_url, or bot_token + channel_id"]

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        if problems := self.validate(creds, payload.options):
            raise NotConfigured("; ".join(problems))
        if dry_run:
            return self.dry(payload)

        text = payload.body + (f"\n{payload.link}" if payload.link else "")
        media = payload.video or payload.image
        path = Path(media) if media and Path(media).exists() else None

        self.throttle(f"discord:{creds.get('webhook_url') or creds.get('channel_id')}")
        # The file handle is opened for the duration of the request and closed
        # by the context manager, so httpx streams it rather than the whole
        # video sitting in memory alongside the web server and job pool.
        with contextlib.ExitStack() as stack, httpx.Client(timeout=300) as client:
            files = None
            if path is not None:
                fh = stack.enter_context(path.open("rb"))
                files = {
                    "files[0]": (
                        path.name, fh,
                        mimetypes.guess_type(path.name)[0]
                        or "application/octet-stream",
                    )
                }
            if url := creds.get("webhook_url"):
                r = client.post(
                    f"{url}?wait=true",
                    data={"payload_json": json.dumps({"content": text})} if files
                    else None,
                    json=None if files else {"content": text},
                    files=files,
                )
            else:
                r = client.post(
                    f"https://discord.com/api/v10/channels/{creds['channel_id']}/messages",
                    headers={"Authorization": f"Bot {creds['bot_token']}"},
                    data={"payload_json": json.dumps({"content": text})} if files
                    else None,
                    json=None if files else {"content": text},
                    files=files,
                )
            data = _raise_for(r, "discord message")
        return Result(True, str(data.get("id", "")), "")
