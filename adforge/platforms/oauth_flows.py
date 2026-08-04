"""
Local authorisation-code flows for LinkedIn, Meta and TikTok.

Same problem as YouTube, three more times: none of these platforms shows you a
usable token in a settings page. LinkedIn's developer console shows a client id
and secret; Meta's shows a short-lived user token that expires in an hour and
is the wrong kind anyway; TikTok's shows nothing. In every case the credential
AdForge stores is the output of an authorisation you perform once.

This runs a throwaway HTTP server on localhost, opens the consent page, catches
the redirect, and exchanges the code. Nothing is written to disk - the values
are printed for you to paste into Accounts, which stores them in the 0600
database.

    python run.py --auth linkedin --client-id ... --client-secret ...
    python run.py --auth meta     --client-id ... --client-secret ...
    python run.py --auth tiktok   --client-id ... --client-secret ...
"""

from __future__ import annotations

import http.server
import secrets
import threading
import urllib.parse
import webbrowser

import httpx

CALLBACK_PORT = 8791
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"


class _Catcher(http.server.BaseHTTPRequestHandler):
    """Single-shot handler that captures the authorisation code."""

    result: dict = {}

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        _Catcher.result = {
            k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        ok = "code" in _Catcher.result
        self.wfile.write(
            b"<html><body style='font-family:system-ui;background:#0b0f14;"
            b"color:#e2e8f0;padding:3rem'><h2>"
            + (b"Authorised." if ok else b"Authorisation failed.")
            + b"</h2><p>Close this tab and return to the terminal.</p>"
            b"</body></html>"
        )

    def log_message(self, *a):  # silence the default stderr logging
        pass


def _await_code(timeout: int = 300) -> dict:
    """Serve exactly one callback and return its query parameters."""
    _Catcher.result = {}
    server = http.server.HTTPServer(("127.0.0.1", CALLBACK_PORT), _Catcher)
    server.timeout = timeout
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    t.join(timeout)
    server.server_close()
    if not _Catcher.result:
        raise SystemExit("timed out waiting for the browser redirect")
    if "error" in _Catcher.result:
        raise SystemExit(
            f"authorisation refused: {_Catcher.result.get('error_description') or _Catcher.result['error']}"
        )
    return _Catcher.result


def _authorise(auth_url: str, params: dict) -> dict:
    state = secrets.token_urlsafe(16)
    params["state"] = state
    url = f"{auth_url}?{urllib.parse.urlencode(params)}"
    print(f"\nOpening:\n  {url}\n")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - headless box, the URL is printed above
        pass
    got = _await_code()
    # The state check is what stops a different site from feeding us its own
    # authorisation code and silently binding the wrong account.
    if got.get("state") != state:
        raise SystemExit("state mismatch - discarding this response")
    return got


LINKEDIN_SETUP = """\
LinkedIn credentials come from an app, not from your profile.

  1. https://www.linkedin.com/developers/apps/new
     Create an app and associate it with the Company Page you want to post as.

  2. Products tab -> request "Share on LinkedIn" and "Sign In with LinkedIn
     using OpenID Connect". To post as the COMPANY rather than yourself you
     also need "Advertising API" or Community Management API access, which
     LinkedIn reviews - posting as a person works immediately.

  3. Auth tab -> add this exact redirect URL:
       %s

  4. Auth tab -> copy the Client ID and Client Secret, then run:
       python run.py --auth linkedin --client-id ... --client-secret ...
""" % REDIRECT_URI


def linkedin_auth(client_id: str, client_secret: str) -> dict:
    got = _authorise(
        "https://www.linkedin.com/oauth/v2/authorization",
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            # openid/profile identify the member; w_member_social is what
            # actually permits posting.
            "scope": "openid profile w_member_social",
        },
    )
    with httpx.Client(timeout=60) as c:
        r = c.post(
            "https://www.linkedin.com/oauth/v2/accessToken",
            data={
                "grant_type": "authorization_code",
                "code": got["code"],
                "redirect_uri": REDIRECT_URI,
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        if r.status_code >= 400:
            raise SystemExit(f"token exchange failed: {r.text[:300]}")
        token = r.json()["access_token"]

        me = c.get(
            "https://api.linkedin.com/v2/userinfo",
            headers={"Authorization": f"Bearer {token}"},
        )
        if me.status_code >= 400:
            raise SystemExit(f"could not read your profile: {me.text[:200]}")
        sub = me.json().get("sub", "")

    print(f"\nAuthorised as {me.json().get('name', '?')}.")
    print("NOTE: LinkedIn access tokens expire after ~60 days. Re-run this "
          "command when posting starts failing with a 401.")
    return {"access_token": token, "author_urn": f"urn:li:person:{sub}"}


META_SETUP = """\
Meta credentials are per-PAGE, and the token the console shows you is the wrong
one - it is a short-lived USER token that expires in about an hour.

  1. https://developers.facebook.com/apps -> Create app -> "Business".

  2. Add the "Facebook Login" product.
     Settings -> Valid OAuth Redirect URIs -> add exactly:
       %s

  3. For Instagram you also need the Instagram account converted to a
     Business or Creator account and linked to the Facebook Page. The API
     cannot publish to a personal Instagram account at all.

  4. Settings -> Basic -> copy the App ID and App Secret, then run:
       python run.py --auth meta --client-id <app id> --client-secret <app secret>

This exchanges for a long-lived token and then reads your Pages, printing the
PAGE token for each - that is the one AdForge needs.
""" % REDIRECT_URI


def meta_auth(client_id: str, client_secret: str) -> dict:
    got = _authorise(
        "https://www.facebook.com/v21.0/dialog/oauth",
        {
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": ",".join([
                "pages_show_list",
                "pages_manage_posts",
                "pages_read_engagement",
                "instagram_basic",
                "instagram_content_publish",
                "business_management",
            ]),
        },
    )
    graph = "https://graph.facebook.com/v21.0"
    with httpx.Client(timeout=60) as c:
        r = c.get(f"{graph}/oauth/access_token", params={
            "client_id": client_id, "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI, "code": got["code"],
        })
        if r.status_code >= 400:
            raise SystemExit(f"token exchange failed: {r.text[:300]}")
        short = r.json()["access_token"]

        # Short-lived tokens last about an hour, which is useless for a
        # scheduler. This exchange makes it ~60 days, and the PAGE tokens
        # derived from a long-lived user token do not expire at all.
        r = c.get(f"{graph}/oauth/access_token", params={
            "grant_type": "fb_exchange_token", "client_id": client_id,
            "client_secret": client_secret, "fb_exchange_token": short,
        })
        long_lived = r.json().get("access_token", short)

        pages = c.get(f"{graph}/me/accounts", params={
            "access_token": long_lived,
            "fields": "name,id,access_token,instagram_business_account{id,username}",
        })
        if pages.status_code >= 400:
            raise SystemExit(f"could not list pages: {pages.text[:250]}")
        data = pages.json().get("data", [])

    if not data:
        raise SystemExit(
            "no Pages returned. The account must be an admin of at least one "
            "Facebook Page, and you must have granted pages_show_list."
        )

    print(f"\nFound {len(data)} page(s):\n")
    out: dict = {}
    for i, page in enumerate(data):
        ig = page.get("instagram_business_account") or {}
        print(f"  [{i + 1}] {page['name']}")
        print(f"      page_id           {page['id']}")
        print(f"      page_access_token {page['access_token']}")
        if ig:
            print(f"      ig_user_id        {ig.get('id')}  (@{ig.get('username','?')})")
        else:
            print("      (no Instagram Business account linked to this page)")
        print()
        if i == 0:
            out = {"page_id": page["id"],
                   "page_access_token": page["access_token"]}
            if ig:
                out["ig_user_id"] = ig.get("id")
                out["ig_access_token"] = page["access_token"]
    print("Facebook uses page_id + page_access_token.")
    print("Instagram uses ig_user_id + access_token (the same page token), plus "
          "a public https public_media_base.\n")
    return out


TIKTOK_SETUP = """\
TikTok shows no token anywhere; it only exists after you authorise.

  1. https://developers.tiktok.com/apps -> create an app.

  2. Add the "Content Posting API" product.
     Redirect URI -> add exactly:
       %s

  3. Scopes: video.upload is enough to send to DRAFTS, which is all an
     unaudited app can do. video.publish requires TikTok to audit the app -
     until then leave "direct post" OFF in Accounts or every upload fails.

  4. Copy the Client Key and Client Secret, then run:
       python run.py --auth tiktok --client-id <client key> --client-secret <client secret>
""" % REDIRECT_URI


def tiktok_auth(client_id: str, client_secret: str) -> dict:
    got = _authorise(
        "https://www.tiktok.com/v2/auth/authorize/",
        {
            "client_key": client_id,
            "response_type": "code",
            "scope": "user.info.basic,video.upload,video.publish",
            "redirect_uri": REDIRECT_URI,
        },
    )
    with httpx.Client(timeout=60) as c:
        r = c.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            data={
                "client_key": client_id,
                "client_secret": client_secret,
                "code": got["code"],
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if r.status_code >= 400 or "refresh_token" not in r.text:
            raise SystemExit(f"token exchange failed: {r.text[:300]}")
        data = r.json()

    scopes = data.get("scope", "")
    print(f"\nAuthorised. Granted scopes: {scopes or 'unknown'}")
    if "video.publish" not in scopes:
        print("video.publish was NOT granted - uploads go to drafts only. That "
              "is expected until TikTok audits the app; leave 'direct post' off.")
    return {
        "client_key": client_id,
        "client_secret": client_secret,
        "refresh_token": data["refresh_token"],
    }


FLOWS = {
    "linkedin": (linkedin_auth, LINKEDIN_SETUP),
    "meta": (meta_auth, META_SETUP),
    "facebook": (meta_auth, META_SETUP),
    "instagram": (meta_auth, META_SETUP),
    "tiktok": (tiktok_auth, TIKTOK_SETUP),
}
