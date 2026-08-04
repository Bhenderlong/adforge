"""
One-time OAuth flows that mint the long-lived tokens AdForge stores.

Some platforms hand you credentials in a settings page; others require you to
complete a consent flow once and keep what comes back. YouTube is the second
kind, and it is a common place to get stuck: YouTube Studio shows a Channel ID
and a User ID, neither of which is a credential, and no page anywhere displays
a refresh token. It only exists as the result of an authorisation you perform.

Run:  python run.py --auth youtube
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Upload plus read access, so the adapter can publish and later read back stats.
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

SETUP_STEPS = """\
Before running this you need an OAuth client from Google Cloud. YouTube Studio
does not have one - the Channel ID and User ID it shows are not credentials.

  1. https://console.cloud.google.com/projectcreate
     Create a project (any name).

  2. APIs & Services -> Library -> search "YouTube Data API v3" -> ENABLE.
     Nothing works until this is on.

  3. APIs & Services -> OAuth consent screen
       User type: External
       Fill in app name and your email, then Save
       Audience -> Test users -> ADD USERS -> add the Google account that
       owns the channel.
     While the app is in Testing this is the only account that can authorise,
     and the token lasts 7 days. Click PUBLISH APP to make it permanent - you
     do NOT need Google's verification review for your own channel.

  4. APIs & Services -> Credentials -> CREATE CREDENTIALS
       -> OAuth client ID
       -> Application type: Desktop app        <- important, not "Web"
     Download the JSON, or copy the client ID and secret.

Then run this command again and a browser will open for you to approve it.
"""


def _client_config(client_id: str, client_secret: str) -> dict:
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def youtube_auth(client_id: str = "", client_secret: str = "",
                 secrets_file: str = "") -> dict:
    """Run the consent flow and return the credentials AdForge needs.

    Accepts either the downloaded client_secret JSON or the id/secret pair.
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("google-auth-oauthlib is not installed:\n"
              "  ./venv/bin/pip install google-auth-oauthlib")
        raise SystemExit(1) from None

    if secrets_file:
        path = Path(secrets_file).expanduser()
        if not path.exists():
            print(f"no such file: {path}")
            raise SystemExit(1)
        data = json.loads(path.read_text())
        # Google exports these under "installed" or "web" depending on type.
        block = data.get("installed") or data.get("web")
        if not block:
            print(f"{path} is not an OAuth client file (no 'installed' key). "
                  "Create a DESKTOP APP client and download that one.")
            raise SystemExit(1)
        if "web" in data:
            print("WARNING: that is a Web application client. The desktop flow "
                  "needs an 'installed' client, and Google will reject the "
                  "loopback redirect. Create a Desktop app client instead.")
        flow = InstalledAppFlow.from_client_config(
            _client_config(block["client_id"], block["client_secret"]),
            YOUTUBE_SCOPES,
        )
        client_id, client_secret = block["client_id"], block["client_secret"]
    else:
        if not (client_id and client_secret):
            print(SETUP_STEPS)
            raise SystemExit(1)
        flow = InstalledAppFlow.from_client_config(
            _client_config(client_id, client_secret), YOUTUBE_SCOPES
        )

    print("\nOpening a browser to authorise. If it does not open, copy the URL "
          "printed below.\n")
    # prompt="consent" is required: without it Google returns a refresh token
    # only on the FIRST ever authorisation for a client, and every retry after
    # that comes back with refresh_token=None for no visible reason.
    creds = flow.run_local_server(
        port=0, access_type="offline", prompt="consent",
        authorization_prompt_message="Open this URL to authorise:\n{url}",
        success_message="Authorised. You can close this tab and return to the terminal.",
    )

    if not creds.refresh_token:
        print("\nGoogle returned no refresh token. That happens when this "
              "client was already authorised for this account - revoke it at "
              "https://myaccount.google.com/permissions and run this again.")
        raise SystemExit(1)

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": creds.refresh_token,
    }


def print_result(platform: str, creds: dict) -> None:
    print("\n" + "=" * 68)
    print(f"  {platform} credentials - paste these into Accounts -> {platform}")
    print("=" * 68)
    for key, value in creds.items():
        print(f"  {key:16} {value}")
    print("=" * 68)
    print("\nThese are secrets. They are not written to disk by this command;\n"
          "AdForge stores them in data/adforge.db (mode 0600) once you save\n"
          "the account in the UI.\n")
