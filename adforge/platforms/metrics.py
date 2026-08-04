"""
Pulling engagement back from the platforms.

The Metric table existed from the first commit and nothing ever wrote to it,
which made the whole idea of measuring a campaign decorative. Without this you
are choosing pillars, times and phrasing by taste.

Every call here is READ-ONLY, same contract as verify.py - nothing in this
module can post or modify. Collection is best-effort by design: a platform that
rate-limits, revokes a token or drops an endpoint must not break the scheduler,
so failures are logged per post and skipped rather than raised.

Not every platform gives the same fields. What is missing is stored as 0 and
distinguished from a real zero by `fetched`, so the UI can say "not reported"
instead of implying a post got no engagement.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("adforge.metrics")

TIMEOUT = 25


class Stats(dict):
    """Engagement for one post. `fetched` False means the platform said nothing."""

    @classmethod
    def empty(cls, note: str = "") -> Stats:
        return cls(fetched=False, note=note, impressions=0, likes=0,
                   comments=0, shares=0, clicks=0)

    @classmethod
    def of(cls, **kw) -> Stats:
        base = dict(fetched=True, note="", impressions=0, likes=0,
                    comments=0, shares=0, clicks=0)
        base.update(kw)
        return cls(base)


def _x(post, creds) -> Stats:
    from requests_oauthlib import OAuth1Session

    sess = OAuth1Session(
        creds["api_key"], client_secret=creds["api_secret"],
        resource_owner_key=creds["access_token"],
        resource_owner_secret=creds["access_token_secret"],
    )
    r = sess.get(
        f"https://api.x.com/2/tweets/{post.remote_id}",
        params={"tweet.fields": "public_metrics,non_public_metrics"},
        timeout=TIMEOUT,
    )
    if r.status_code >= 400:
        return Stats.empty(f"{r.status_code}")
    m = (r.json().get("data") or {}).get("public_metrics", {})
    # impression_count is only present on tiers that expose non_public_metrics;
    # absent is not zero.
    non_pub = (r.json().get("data") or {}).get("non_public_metrics", {})
    return Stats.of(
        likes=m.get("like_count", 0),
        comments=m.get("reply_count", 0),
        shares=m.get("retweet_count", 0) + m.get("quote_count", 0),
        impressions=m.get("impression_count", non_pub.get("impression_count", 0)),
        clicks=non_pub.get("url_link_clicks", 0),
    )


def _linkedin(post, creds) -> Stats:
    with httpx.Client(timeout=TIMEOUT) as c:
        r = c.get(
            "https://api.linkedin.com/v2/socialActions/" + post.remote_id,
            headers={"Authorization": f"Bearer {creds['access_token']}",
                     "X-Restli-Protocol-Version": "2.0.0"},
        )
    if r.status_code >= 400:
        return Stats.empty(f"{r.status_code}")
    d = r.json()
    # Impressions need the organizationalEntityShareStatistics endpoint and an
    # org page; the social actions endpoint has engagement only.
    return Stats.of(
        likes=(d.get("likesSummary") or {}).get("totalLikes", 0),
        comments=(d.get("commentsSummary") or {}).get("totalFirstLevelComments", 0),
    )


def _facebook(post, creds) -> Stats:
    with httpx.Client(timeout=TIMEOUT) as c:
        r = c.get(
            f"https://graph.facebook.com/v21.0/{post.remote_id}",
            params={"fields": "insights.metric(post_impressions,post_clicks),"
                              "likes.summary(true),comments.summary(true),"
                              "shares",
                    "access_token": creds["page_access_token"]},
        )
    if r.status_code >= 400:
        return Stats.empty(f"{r.status_code}")
    d = r.json()
    ins = {i["name"]: (i.get("values") or [{}])[0].get("value", 0)
           for i in (d.get("insights") or {}).get("data", [])}
    return Stats.of(
        impressions=ins.get("post_impressions", 0),
        clicks=ins.get("post_clicks", 0),
        likes=(d.get("likes") or {}).get("summary", {}).get("total_count", 0),
        comments=(d.get("comments") or {}).get("summary", {}).get("total_count", 0),
        shares=(d.get("shares") or {}).get("count", 0),
    )


def _instagram(post, creds) -> Stats:
    with httpx.Client(timeout=TIMEOUT) as c:
        r = c.get(
            f"https://graph.facebook.com/v21.0/{post.remote_id}",
            params={"fields": "like_count,comments_count,"
                              "insights.metric(reach,saved)",
                    "access_token": creds["access_token"]},
        )
    if r.status_code >= 400:
        return Stats.empty(f"{r.status_code}")
    d = r.json()
    ins = {i["name"]: (i.get("values") or [{}])[0].get("value", 0)
           for i in (d.get("insights") or {}).get("data", [])}
    return Stats.of(
        impressions=ins.get("reach", 0),
        likes=d.get("like_count", 0),
        comments=d.get("comments_count", 0),
        shares=ins.get("saved", 0),
    )


def _youtube(post, creds) -> Stats:
    with httpx.Client(timeout=TIMEOUT) as c:
        tok = c.post("https://oauth2.googleapis.com/token", data={
            "client_id": creds["client_id"], "client_secret": creds["client_secret"],
            "refresh_token": creds["refresh_token"], "grant_type": "refresh_token",
        })
        if tok.status_code >= 400:
            return Stats.empty("token refresh failed")
        r = c.get("https://www.googleapis.com/youtube/v3/videos",
                  params={"part": "statistics", "id": post.remote_id},
                  headers={"Authorization": f"Bearer {tok.json()['access_token']}"})
    if r.status_code >= 400:
        return Stats.empty(f"{r.status_code}")
    items = r.json().get("items", [])
    if not items:
        return Stats.empty("video not found")
    s = items[0].get("statistics", {})
    return Stats.of(
        impressions=int(s.get("viewCount", 0)),
        likes=int(s.get("likeCount", 0)),
        comments=int(s.get("commentCount", 0)),
    )


def _reddit(post, creds) -> Stats:
    import praw

    reddit = praw.Reddit(
        client_id=creds["client_id"], client_secret=creds["client_secret"],
        username=creds["username"], password=creds["password"],
        user_agent=creds.get("user_agent") or "adforge", check_for_async=False,
    )
    sub = reddit.submission(id=post.remote_id)
    return Stats.of(
        likes=sub.score,
        comments=sub.num_comments,
        # Reddit reports views only to the owning moderator and often not at
        # all; upvote_ratio is more useful and has nowhere else to go.
        impressions=getattr(sub, "view_count", None) or 0,
        clicks=0,
    )


COLLECTORS = {
    "x": _x,
    "linkedin": _linkedin,
    "facebook": _facebook,
    "instagram": _instagram,
    "youtube": _youtube,
    "reddit": _reddit,
    # Discord, Slack and TikTok deliberately absent. Discord and Slack expose
    # no per-message engagement to a bot, and TikTok's query endpoint needs a
    # scope unaudited apps do not have. Claiming to measure them would be worse
    # than saying nothing.
}


def collect(post, creds: dict) -> Stats:
    fn = COLLECTORS.get(post.platform)
    if fn is None:
        return Stats.empty("platform reports no per-post engagement")
    if not post.remote_id:
        return Stats.empty("no remote id recorded")
    try:
        return fn(post, creds)
    except Exception as e:  # noqa: BLE001 - collection must never break the loop
        log.warning("metrics for post %s (%s) failed: %s", post.id, post.platform, e)
        return Stats.empty(f"{type(e).__name__}")
