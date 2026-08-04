"""
Read-only credential verification.

"Are the fields filled in" is not the question anyone actually has when they
paste a token. The question is whether it works, and the only honest way to
answer that is to call the platform.

Every check here is a GET against an identity or metadata endpoint - it reads
who you are and nothing else. Nothing in this module can post, delete or
modify. That matters because this runs on demand from a button, and a
verification step that could publish would be a trap.

Each returns (ok, message). The message names the authenticated identity when
it works, so you can catch the other common failure: valid credentials for the
wrong account.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("adforge.verify")

TIMEOUT = 20


def _fail(e: Exception) -> tuple[bool, str]:
    return False, f"{type(e).__name__}: {str(e)[:160]}"


def verify_reddit(creds: dict, opts: dict) -> tuple[bool, str]:
    try:
        import praw

        reddit = praw.Reddit(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            username=creds["username"],
            password=creds["password"],
            user_agent=creds.get("user_agent") or "adforge",
            check_for_async=False,
        )
        me = reddit.user.me()
        if me is None:
            return False, "authenticated but no user returned - check username/password"
        karma = getattr(me, "link_karma", 0) + getattr(me, "comment_karma", 0)
        msg = f"authenticated as u/{me.name} ({karma} karma)"

        # A brand-new or low-karma account is rate-limited hard by Reddit and
        # its posts are often auto-removed, which looks like a bug in this tool.
        if karma < 10:
            msg += " - WARNING: low karma, Reddit heavily rate-limits new accounts"
        if sub := opts.get("subreddit"):
            try:
                s = reddit.subreddit(sub)
                msg += f"; r/{s.display_name} reachable ({s.subscribers:,} members)"
            except Exception as e:  # noqa: BLE001
                msg += f"; but r/{sub} is NOT reachable: {str(e)[:70]}"
                return False, msg
        return True, msg
    except Exception as e:  # noqa: BLE001
        low = str(e).lower()
        if "401" in low or "unauthorized" in low:
            return False, ("401 unauthorized - wrong client id/secret, or the app "
                           "is not type 'script', or the account has 2FA on "
                           "(script auth cannot do the second factor)")
        return _fail(e)


def verify_x(creds: dict, opts: dict) -> tuple[bool, str]:
    try:
        from requests_oauthlib import OAuth1Session

        sess = OAuth1Session(
            creds["api_key"], client_secret=creds["api_secret"],
            resource_owner_key=creds["access_token"],
            resource_owner_secret=creds["access_token_secret"],
        )
        r = sess.get("https://api.x.com/2/users/me", timeout=TIMEOUT)
        if r.status_code == 401:
            return False, "401 - keys rejected. Regenerate the access token AFTER setting the app to Read and Write."
        if r.status_code == 403:
            return False, ("403 - authenticated but not permitted. The app's user "
                           "authentication settings need Read and Write, and the "
                           "access token must be regenerated afterwards.")
        if r.status_code == 429:
            return False, "429 - rate limited; the credentials may be fine, retry shortly"
        if r.status_code >= 400:
            return False, f"{r.status_code}: {r.text[:150]}"
        u = r.json().get("data", {})
        return True, f"authenticated as @{u.get('username', '?')} ({u.get('name', '')})"
    except Exception as e:  # noqa: BLE001
        return _fail(e)


def verify_linkedin(creds: dict, opts: dict) -> tuple[bool, str]:
    try:
        urn = creds.get("author_urn", "")
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.get(
                "https://api.linkedin.com/v2/userinfo",
                headers={"Authorization": f"Bearer {creds['access_token']}"},
            )
        if r.status_code == 401:
            return False, "401 - access token expired or invalid (LinkedIn tokens last 60 days)"
        if r.status_code >= 400:
            return False, f"{r.status_code}: {r.text[:150]}"
        who = r.json().get("name") or r.json().get("sub", "?")
        msg = f"token valid for {who}"
        if urn.startswith("urn:li:organization:"):
            msg += "; posting as an organization needs the w_organization_social scope"
        elif not urn.startswith("urn:li:person:"):
            return False, (f"author_urn {urn!r} is malformed - it must be "
                           f"urn:li:person:<id> or urn:li:organization:<id>")
        return True, msg
    except Exception as e:  # noqa: BLE001
        return _fail(e)


def verify_facebook(creds: dict, opts: dict) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.get(
                f"https://graph.facebook.com/v21.0/{creds['page_id']}",
                params={"fields": "name,category",
                        "access_token": creds["page_access_token"]},
            )
        if r.status_code >= 400:
            body = r.text[:200]
            if "OAuthException" in body:
                return False, ("token rejected. A USER token returns 200 on /me but "
                               "fails here - you need the PAGE token from /me/accounts")
            return False, f"{r.status_code}: {body}"
        d = r.json()
        return True, f"page {d.get('name', '?')} ({d.get('category', '')})"
    except Exception as e:  # noqa: BLE001
        return _fail(e)


def verify_instagram(creds: dict, opts: dict) -> tuple[bool, str]:
    base = str(creds.get("public_media_base", ""))
    if not base.startswith("https://"):
        return False, ("public_media_base must be a public https:// URL - Meta "
                       "fetches the media itself and cannot reach localhost")
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.get(
                f"https://graph.facebook.com/v21.0/{creds['ig_user_id']}",
                params={"fields": "username,account_type",
                        "access_token": creds["access_token"]},
            )
        if r.status_code >= 400:
            return False, f"{r.status_code}: {r.text[:180]}"
        d = r.json()
        kind = d.get("account_type", "")
        msg = f"@{d.get('username', '?')} ({kind})"
        if kind not in ("BUSINESS", "MEDIA_CREATOR", ""):
            return False, msg + " - the API only publishes for Business/Creator accounts"
        return True, msg
    except Exception as e:  # noqa: BLE001
        return _fail(e)


def verify_youtube(creds: dict, opts: dict) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            tok = c.post("https://oauth2.googleapis.com/token", data={
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
                "refresh_token": creds["refresh_token"],
                "grant_type": "refresh_token",
            })
            if tok.status_code >= 400:
                body = tok.text[:200]
                if "invalid_grant" in body:
                    return False, ("invalid_grant - the refresh token was revoked or "
                                   "expired. If the consent screen is still in "
                                   "'Testing' the token dies after 7 days; publish "
                                   "the app. Re-run: run.py --auth youtube")
                return False, f"token refresh failed: {body}"
            access = tok.json()["access_token"]

            r = c.get("https://www.googleapis.com/youtube/v3/channels",
                      params={"part": "snippet,status", "mine": "true"},
                      headers={"Authorization": f"Bearer {access}"})
        if r.status_code >= 400:
            return False, f"{r.status_code}: {r.text[:180]}"
        items = r.json().get("items", [])
        if not items:
            return False, "token works but no channel is attached to this account"
        ch = items[0]["snippet"]
        return True, f"channel {ch.get('title', '?')}"
    except Exception as e:  # noqa: BLE001
        return _fail(e)


def verify_tiktok(creds: dict, opts: dict) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post("https://open.tiktokapis.com/v2/oauth/token/", data={
                "client_key": creds["client_key"],
                "client_secret": creds["client_secret"],
                "grant_type": "refresh_token",
                "refresh_token": creds["refresh_token"],
            }, headers={"Content-Type": "application/x-www-form-urlencoded"})
        if r.status_code >= 400 or "access_token" not in r.text:
            return False, f"token refresh failed: {r.text[:180]}"
        scopes = r.json().get("scope", "")
        direct = bool(opts.get("direct_post"))
        if direct and "video.publish" not in scopes:
            return False, ("'direct post' is enabled but this app lacks the "
                           f"video.publish scope (has: {scopes or 'none'}). "
                           "Uploads will fail until TikTok audits the app.")
        return True, f"token refreshed; scopes: {scopes or 'unknown'}"
    except Exception as e:  # noqa: BLE001
        return _fail(e)


def verify_discord(creds: dict, opts: dict) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            if url := creds.get("webhook_url"):
                # GET on a webhook returns its metadata without posting.
                r = c.get(url)
                if r.status_code >= 400:
                    return False, f"webhook rejected ({r.status_code}) - it may have been deleted"
                d = r.json()
                return True, f"webhook '{d.get('name', '?')}' in channel {d.get('channel_id', '?')}"
            r = c.get("https://discord.com/api/v10/users/@me",
                      headers={"Authorization": f"Bot {creds['bot_token']}"})
            if r.status_code >= 400:
                return False, f"bot token rejected ({r.status_code})"
            return True, f"bot {r.json().get('username', '?')}"
    except Exception as e:  # noqa: BLE001
        return _fail(e)


def verify_slack(creds: dict, opts: dict) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post("https://slack.com/api/auth.test",
                       headers={"Authorization": f"Bearer {creds['bot_token']}"})
        d = r.json()
        if not d.get("ok"):
            return False, f"slack: {d.get('error', 'unknown error')}"
        return True, f"{d.get('user', '?')} in {d.get('team', '?')}"
    except Exception as e:  # noqa: BLE001
        return _fail(e)


VERIFIERS = {
    "reddit": verify_reddit,
    "x": verify_x,
    "linkedin": verify_linkedin,
    "facebook": verify_facebook,
    "instagram": verify_instagram,
    "youtube": verify_youtube,
    "tiktok": verify_tiktok,
    "discord": verify_discord,
    "slack": verify_slack,
}


def verify(platform: str, creds: dict, opts: dict) -> tuple[bool, str]:
    fn = VERIFIERS.get(platform)
    if fn is None:
        return False, f"no verifier for {platform}"
    log.info("verifying %s credentials (read-only)", platform)
    return fn(creds, opts)
