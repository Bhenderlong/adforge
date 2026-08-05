"""Where each credential actually comes from.

A field labelled `client_id` with an empty box next to it is not guidance. The
Reddit client id in particular is the single most-missed value on the whole
setup path, because it is the one string on that page with no label printed
next to it - people paste the app name instead and get a 401 that says nothing
useful.

Anything here that is a gotcha rather than a location is marked with a warning,
because those are the ones that cost an hour: Reddit script apps only work for
the account that created them and break silently under 2FA; a Discord webhook
can post but can never read, so the radar needs the bot token instead.

Keep these short. They are hints under an input, not documentation.
"""

from __future__ import annotations

# (platform, field) -> (where to get it, gotcha or None)
CRED_HELP: dict[tuple[str, str], tuple[str, str | None]] = {
    # --- Reddit -------------------------------------------------------------
    ("reddit", "client_id"): (
        "old.reddit.com/prefs/apps - the UNLABELLED string just under "
        "\"personal use script\", about 14 characters.",
        "Not the app name. That is the usual mistake and it fails as a 401.",
    ),
    ("reddit", "client_secret"): (
        "Same app, the value actually labelled \"secret\".", None,
    ),
    ("reddit", "username"): (
        "The account this posts and replies as, without the u/.",
        "A script app only works for the account that CREATED it.",
    ),
    ("reddit", "password"): (
        "That account's password.",
        "2FA must be off. Script auth has no second-factor step, and it fails "
        "with a 401 identical to a wrong password.",
    ),
    ("reddit", "user_agent"): (
        "Any descriptive string. Reddit asks for "
        "python:adforge:v1.0 (by /u/yourname).",
        "Reddit rate-limits generic agents harder. Do not send the default.",
    ),
    # --- Discord ------------------------------------------------------------
    ("discord", "webhook_url"): (
        "Server Settings -> Integrations -> Webhooks -> New Webhook -> "
        "Copy Webhook URL.",
        "Post-only. A webhook can never READ, so the radar cannot use it.",
    ),
    ("discord", "bot_token"): (
        "discord.com/developers/applications -> your app -> Bot -> Reset "
        "Token. Invite it to the server with Send Messages.",
        "For the radar to read messages it needs the MESSAGE CONTENT intent, "
        "toggled on that same Bot page.",
    ),
    ("discord", "channel_id"): (
        "Turn on User Settings -> Advanced -> Developer Mode, then "
        "right-click the channel -> Copy Channel ID.",
        None,
    ),
    # --- everything else ----------------------------------------------------
    ("x", "api_key"): ("developer.x.com -> your project -> Keys and tokens.", None),
    ("x", "api_secret"): ("Shown once, beside the API key.", None),
    ("x", "access_token"): (
        "Same page, \"Access Token and Secret\".",
        "The app must be set to Read and write BEFORE generating these, or "
        "posting 403s with a read-only token.",
    ),
    ("x", "access_token_secret"): ("Shown once, beside the access token.", None),
    ("linkedin", "access_token"): (
        "Run: python run.py --auth linkedin", None,
    ),
    ("linkedin", "author_urn"): (
        "urn:li:person:XXXX for a personal profile, or "
        "urn:li:organization:NNN for a company page.", None,
    ),
    ("facebook", "page_id"): ("Page -> About -> Page ID.", None),
    ("facebook", "page_access_token"): (
        "Run: python run.py --auth facebook",
        "You want a PAGE token, not a user token.",
    ),
    ("instagram", "ig_user_id"): (
        "The Instagram Business account id linked to your Facebook page.",
        "Must be a Business or Creator account. Personal accounts cannot "
        "publish through the API at all.",
    ),
    ("instagram", "access_token"): ("Run: python run.py --auth instagram", None),
    ("instagram", "public_media_base"): (
        "A public https base URL where rendered images are reachable.",
        "Instagram fetches the image from this URL itself, so localhost and "
        "a LAN address cannot work.",
    ),
    ("tiktok", "client_key"): ("developers.tiktok.com -> your app.", None),
    ("tiktok", "client_secret"): ("Same page as the client key.", None),
    ("tiktok", "refresh_token"): ("Run: python run.py --auth tiktok", None),
    ("youtube", "client_id"): (
        "console.cloud.google.com -> Credentials -> OAuth client ID.",
        "Create it as a DESKTOP app. A Web client rejects the local callback.",
    ),
    ("youtube", "client_secret"): ("Downloaded with the OAuth client.", None),
    ("youtube", "refresh_token"): (
        "Run: python run.py --auth youtube",
        "Not the channel id or user id - those are not credentials.",
    ),
    ("slack", "bot_token"): (
        "api.slack.com/apps -> OAuth & Permissions -> Bot User OAuth Token.",
        "Starts xoxb-. Needs the chat:write scope.",
    ),
    ("slack", "channel"): ("Channel id (C…) or #name the bot has joined.", None),
}


def help_for(platform: str, field: str) -> tuple[str, str | None]:
    return CRED_HELP.get((platform, field), ("", None))
