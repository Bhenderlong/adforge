"""
Reddit, YouTube and TikTok adapters.

Reddit carries an extra safety mechanism the others do not need: an explicit
per-subreddit allowlist. Posting promotional content to a subreddit whose rules
forbid it does not merely get the post removed - repeated offences get the
DOMAIN banned sitewide, after which nobody can link inferix.co anywhere on
Reddit. That is unrecoverable, so the adapter refuses to post to any subreddit
the user has not explicitly enabled, and refuses to include a link where the
user has not confirmed links are permitted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from .base import BaseAdapter, NotConfigured, Payload, PublishError, Result

log = logging.getLogger("adforge.publish")


class RedditAdapter(BaseAdapter):
    key = "reddit"
    required_credentials = (
        "client_id", "client_secret", "username", "password", "user_agent",
    )
    # Reddit's own guidance is that self-promotion should be a small fraction
    # of activity. This floor is per-account across all subreddits.
    min_interval_seconds = 1800

    def _reddit(self, creds: dict):
        try:
            import praw
        except ImportError as e:
            raise NotConfigured("praw not installed") from e
        return praw.Reddit(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            username=creds["username"],
            password=creds["password"],
            user_agent=creds.get("user_agent")
            or f"adforge by u/{creds['username']}",
        )

    def validate(self, creds: dict, options: dict) -> list[str]:
        problems = super().validate(creds, options)
        sub = str(options.get("subreddit") or "").strip()
        if not sub:
            problems.append("no subreddit specified")
            return problems

        # The confirmation is bound to the subreddit name it was granted for.
        # A bare boolean carried over when the field was retyped: vet
        # r/LocalLLaMA, later change the name to r/MachineLearning, and the
        # "I have read the rules" tick silently applied to a community nobody
        # had checked. Given the stated downside is a sitewide domain ban, the
        # grant has to be specific.
        granted = str(options.get("allowed_for") or "").strip()
        if not options.get("allowed") or granted.lower() != sub.lower():
            if granted and granted.lower() != sub.lower():
                problems.append(
                    f"the rules confirmation was given for r/{granted}, not "
                    f"r/{sub} - re-confirm it for this subreddit in the UI"
                )
            else:
                problems.append(
                    f"subreddit r/{sub} is not allowlisted - confirm its "
                    "self-promotion rules and enable it in the UI first"
                )
        return problems

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        opts = payload.options
        if problems := self.validate(creds, opts):
            raise NotConfigured("; ".join(problems))

        sub = opts["subreddit"]
        body = payload.body
        # A subreddit can be allowlisted for participation but not for links.
        if payload.link and opts.get("links_allowed"):
            body = f"{body}\n\n{payload.link}"
        elif payload.link:
            log.info("r/%s: links not permitted, omitting URL", sub)

        if not payload.title:
            raise PublishError("reddit posts require a title")
        if dry_run:
            return self.dry(payload, f"to r/{sub}, links_allowed="
                                     f"{bool(opts.get('links_allowed'))}")

        self.throttle(f"reddit:{creds['username']}")
        reddit = self._reddit(creds)
        try:
            subreddit = reddit.subreddit(sub)
            if payload.image and Path(payload.image).exists() and opts.get("allow_image"):
                submission = subreddit.submit_image(
                    title=payload.title, image_path=str(payload.image)
                )
            else:
                submission = subreddit.submit(title=payload.title, selftext=body)
        except Exception as e:  # noqa: BLE001 - praw raises many shapes
            msg = str(e)
            retryable = "RATELIMIT" in msg.upper() or "503" in msg
            raise PublishError(f"reddit submit to r/{sub} failed: {msg[:300]}",
                               retryable) from e
        return Result(True, submission.id, f"https://reddit.com{submission.permalink}")


class YouTubeAdapter(BaseAdapter):
    """YouTube Data API v3 resumable upload.

    A vertical video of 60s or less is classified as a Short automatically -
    there is no API flag for it, so the format of the file is what decides.
    Uploads from an unverified project are capped at 15 minutes and may be
    locked to private; verify the project in the Google Cloud console first.
    """

    key = "youtube"
    required_credentials = ("client_id", "client_secret", "refresh_token")
    min_interval_seconds = 600

    def _service(self, creds: dict):
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError as e:
            raise NotConfigured("google-api-python-client not installed") from e
        c = Credentials(
            token=None,
            refresh_token=creds["refresh_token"],
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/youtube.upload"],
        )
        return build("youtube", "v3", credentials=c, cache_discovery=False)

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        if problems := self.validate(creds, payload.options):
            raise NotConfigured("; ".join(problems))
        if not payload.video or not Path(payload.video).exists():
            raise PublishError("youtube requires a video file")
        if dry_run:
            return self.dry(payload, "would upload as a Short (vertical, <=60s)")

        from googleapiclient.http import MediaFileUpload

        tags = [t.strip("#") for t in payload.options.get("tags", [])]
        description = payload.body
        if payload.link:
            description = f"{description}\n\n{payload.link}"

        self.throttle(f"yt:{creds['client_id'][:12]}")
        service = self._service(creds)
        media = MediaFileUpload(str(payload.video), chunksize=-1, resumable=True)
        try:
            request = service.videos().insert(
                part="snippet,status",
                body={
                    "snippet": {
                        "title": (payload.title or payload.body[:95])[:100],
                        "description": description[:4900],
                        "tags": tags[:15],
                        "categoryId": payload.options.get("category_id", "28"),
                    },
                    "status": {
                        "privacyStatus": payload.options.get("privacy", "public"),
                        "selfDeclaredMadeForKids": False,
                    },
                },
                media_body=media,
            )
            response = None
            while response is None:
                _, response = request.next_chunk()
        except Exception as e:  # noqa: BLE001
            raise PublishError(f"youtube upload failed: {str(e)[:300]}", True) from e

        vid = response["id"]
        return Result(True, vid, f"https://youtube.com/watch?v={vid}")


class TikTokAdapter(BaseAdapter):
    """TikTok Content Posting API.

    IMPORTANT, and not a limitation of this tool: until a TikTok app passes
    audit, its `video.publish` scope is unavailable and content can only be
    sent to the account's DRAFTS via `/inbox/video/init/`. The user finishes
    publishing in the TikTok app. `direct_post` may be set once the app is
    audited, and the adapter will then publish directly.

    Access tokens expire in 24h; a refresh token is mandatory for unattended
    operation.
    """

    key = "tiktok"
    required_credentials = ("client_key", "client_secret", "refresh_token")
    min_interval_seconds = 900
    API = "https://open.tiktokapis.com/v2"

    def _access_token(self, creds: dict) -> str:
        with httpx.Client(timeout=60) as c:
            r = c.post(
                f"{self.API}/oauth/token/",
                data={
                    "client_key": creds["client_key"],
                    "client_secret": creds["client_secret"],
                    "grant_type": "refresh_token",
                    "refresh_token": creds["refresh_token"],
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if r.status_code >= 400:
            raise PublishError(f"tiktok token refresh failed: {r.text[:300]}")
        data = r.json()
        if "access_token" not in data:
            # Never interpolate `data` - a partial-scope or drifted response
            # carries the refresh_token, and this error is persisted to the DB,
            # written to the log, and rendered on the post page.
            raise PublishError(
                "tiktok token response contained no access_token "
                f"(keys: {sorted(data)})"
            )
        return data["access_token"]

    def publish(self, payload: Payload, creds: dict, dry_run: bool) -> Result:
        if problems := self.validate(creds, payload.options):
            raise NotConfigured("; ".join(problems))
        if not payload.video or not Path(payload.video).exists():
            raise PublishError("tiktok requires a video file")

        direct = bool(payload.options.get("direct_post"))
        mode = "direct publish" if direct else "DRAFT (app not audited)"
        if dry_run:
            return self.dry(payload, f"mode: {mode}")

        path = Path(payload.video)
        size = path.stat().st_size
        token = self._access_token(creds)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
        }
        endpoint = (
            f"{self.API}/post/publish/video/init/" if direct
            else f"{self.API}/post/publish/inbox/video/init/"
        )
        body: dict = {
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": size,
                "chunk_size": size,  # single chunk; files here are a few MB
                "total_chunk_count": 1,
            }
        }
        if direct:
            body["post_info"] = {
                "title": payload.body[:2200],
                "privacy_level": payload.options.get(
                    "privacy_level", "SELF_ONLY"
                ),
                "disable_comment": False,
                "disable_duet": False,
                "disable_stitch": False,
            }

        self.throttle(f"tiktok:{creds['client_key'][:10]}")
        with httpx.Client(timeout=600) as c:
            init = c.post(endpoint, headers=headers, json=body)
            if init.status_code >= 400:
                raise PublishError(f"tiktok init failed: {init.text[:400]}")
            data = init.json().get("data", {})
            upload_url = data.get("upload_url")
            publish_id = data.get("publish_id", "")
            if not upload_url:
                raise PublishError(f"tiktok init returned no upload_url: {data}")

            # Streamed: this is the full rendered mp4.
            with path.open("rb") as fh:
                put = c.put(
                    upload_url,
                    content=fh,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Range": f"bytes 0-{size - 1}/{size}",
                    },
                    timeout=1800,
                )
            if put.status_code >= 400:
                raise PublishError(f"tiktok upload failed: {put.status_code}")

        detail = (
            "published" if direct
            else "sent to TikTok drafts - open the app to finish posting"
        )
        return Result(True, publish_id, "", detail)
