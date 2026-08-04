"""
Safety-property tests: the things whose failure posts to the internet.

Covers dry-run resolution, the radar's anti-astroturf policy, and the critic's
minimum-dimension scoring. All offline.
"""

from __future__ import annotations

import json

import pytest

from adforge.config import settings
from adforge.db import Account, PostMode
from adforge.llm.copywriter import _score_of
from adforge.platforms.base import NotConfigured
from adforge.platforms.registry import ADAPTERS, is_dry
from adforge.radar import policy


# --- dry run ----------------------------------------------------------------

def test_no_account_always_means_dry_run():
    """An unconfigured destination must never transmit."""
    assert is_dry(None) is True


def test_account_override_beats_the_global(monkeypatch):
    live = Account(brand="inferix", platform="x", dry_run=False)
    dry = Account(brand="inferix", platform="x", dry_run=True)

    monkeypatch.setattr(settings, "dry_run", True)
    assert is_dry(live) is False  # explicit live overrides a global dry run
    assert is_dry(dry) is True

    monkeypatch.setattr(settings, "dry_run", False)
    assert is_dry(dry) is True  # explicit dry survives a global live
    assert is_dry(live) is False


def test_null_account_setting_inherits_the_global(monkeypatch):
    inherit = Account(brand="inferix", platform="x", dry_run=None)
    monkeypatch.setattr(settings, "dry_run", True)
    assert is_dry(inherit) is True
    monkeypatch.setattr(settings, "dry_run", False)
    assert is_dry(inherit) is False


# --- adapters fail closed ---------------------------------------------------

@pytest.mark.parametrize("key", sorted(ADAPTERS))
def test_every_adapter_refuses_without_credentials(key):
    adapter = ADAPTERS[key]
    problems = adapter.validate({}, {})
    assert problems, f"{key} accepted empty credentials"


def test_reddit_refuses_a_subreddit_that_is_not_allowlisted():
    creds = dict(client_id="c", client_secret="s", username="u",
                 password="p", user_agent="ua")
    adapter = ADAPTERS["reddit"]
    # Full credentials, real subreddit, but the user never ticked "allowed".
    problems = adapter.validate(creds, {"subreddit": "LocalLLaMA"})
    assert any("allowlist" in p for p in problems)
    # The tick alone is not enough - it must name the subreddit it was for.
    assert adapter.validate(creds, {"subreddit": "LocalLLaMA", "allowed": True})
    assert adapter.validate(
        creds,
        {"subreddit": "LocalLLaMA", "allowed": True, "allowed_for": "LocalLLaMA"},
    ) == []


def test_reddit_consent_does_not_carry_to_a_renamed_subreddit():
    """Vetting r/LocalLLaMA then retyping the field must not stay approved."""
    creds = dict(client_id="c", client_secret="s", username="u",
                 password="p", user_agent="ua")
    problems = ADAPTERS["reddit"].validate(
        creds,
        {"subreddit": "MachineLearning", "allowed": True,
         "allowed_for": "LocalLLaMA"},
    )
    assert any("was given for r/LocalLLaMA" in p for p in problems)


# --- radar policy -----------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "I just used Inferix last week and it completely solved all my problems, "
        "you should check it out, it has been a lifesaver for our whole team.",
        "Highly recommend Inferix, it is the best GPU platform I have ever used "
        "and it changed how our entire team ships models to production daily.",
        "DM me and use my referral code for Inferix, it is a total game-changer "
        "for anyone running inference workloads on rented hardware right now.",
    ],
)
def test_astroturf_phrasing_is_blocked(text):
    """The exact behaviour the original brief asked for, refused in code."""
    problems = policy.blocking(policy.check_reply(text, "inferix", True))
    assert problems, f"astroturf passed the policy gate: {text[:60]}"


def test_a_substantive_disclosed_reply_is_allowed():
    text = (
        "The bottleneck there is KV cache, not raw FLOPs. Work out "
        "2 x layers x kv_heads x head_dim x seq_len x 2 bytes for your real "
        "context length and you will see where it stops fitting. For what it "
        "is worth this does not help with third-party marketplaces at all. "
        "(disclosure: I work on Inferix)"
    )
    assert policy.blocking(policy.check_reply(text, "inferix", True)) == []


def test_a_reply_that_is_only_a_link_is_blocked():
    problems = policy.blocking(policy.check_reply("https://inferix.co", "inferix", True))
    assert problems


def test_links_are_blocked_where_the_community_forbids_them():
    text = (
        "Genuinely useful and sufficiently long technical answer about how KV "
        "cache scaling works in practice for this workload. https://inferix.co"
    )
    problems = policy.check_reply(text, "inferix", links_allowed=False)
    assert any("does not permit links" in p for p in problems)


def test_disclosure_is_appended_and_idempotent():
    body = "A helpful reply that says something substantive about the problem."
    once = policy.ensure_disclosure(body, "inferix")
    assert policy.has_disclosure(once, "inferix")
    assert policy.ensure_disclosure(once, "inferix") == once  # never doubles up


def test_disclosure_survives_a_human_edit_that_removed_it():
    edited = "Reply text a human rewrote and stripped the disclosure line from."
    assert policy.has_disclosure(policy.ensure_disclosure(edited, "inferix"), "inferix")


# --- critic scoring ---------------------------------------------------------

def test_score_is_the_worst_dimension_not_the_average():
    """A post with one fatal weakness used to average out to a passing score.

    The real case: 'LoRA reduces VRAM usage. Inferix offers verified hardware.'
    scored 8.0 because five adequate dimensions outvoted a coherence of 2.
    """
    verdict = {"hook": 8, "specificity": 9, "human": 8, "value": 8,
               "brand_fit": 9, "coherence": 2, "overall": 8}
    assert _score_of(verdict) == 2.0


def test_score_falls_back_to_overall_when_dimensions_are_missing():
    assert _score_of({"overall": 7}) == 7.0
    assert _score_of({}) == 5.0


# --- republish guard --------------------------------------------------------

def test_a_post_with_a_remote_id_is_never_sent_again():
    """A 5xx after the platform accepted the post must not duplicate it.

    Retryable failures are re-queued, but the platform may have accepted the
    post before the connection broke. Where an id was recorded, sending again
    would publish a second copy - and a duplicate is worse than a miss, since
    only the miss is fixable from the queue.
    """
    import json

    from adforge.db import Post
    from adforge.platforms.base import PublishError
    from adforge.platforms.registry import publish_post

    acct = Account(brand="inferix", platform="discord", dry_run=False,
                   credentials=json.dumps({"webhook_url": "https://discord.com/api/webhooks/1/abc"}),
                   options="{}")
    post = Post(brand="inferix", platform="discord",
                body="Body copy long enough to clear the length gate here.",
                remote_id="1234567890")
    with pytest.raises(PublishError, match="already published"):
        publish_post(post, acct)


def test_the_republish_guard_does_not_block_a_dry_run():
    """Dry runs must stay repeatable - nothing was transmitted."""
    import json

    from adforge.db import Post
    from adforge.platforms.registry import publish_post

    acct = Account(brand="inferix", platform="discord", dry_run=True,
                   credentials=json.dumps({"webhook_url": "https://discord.com/api/webhooks/1/abc"}),
                   options="{}")
    post = Post(brand="inferix", platform="discord",
                body="Body copy long enough to clear the length gate here.",
                remote_id="1234567890")
    assert publish_post(post, acct).dry_run is True


def test_reddit_refuses_when_no_subreddit_is_named():
    """Empty subreddit with the tick set satisfied both halves of the check.

    sub="" and allowed_for="" compare equal, and allowed=True clears the other
    half, so validate() returned no problems and the adapter would have tried
    to submit to nothing. Found by mutation: removing the no-subreddit guard
    broke no test.
    """
    creds = dict(client_id="c", client_secret="s", username="u",
                 password="p", user_agent="ua")
    for opts in (
        {},
        {"allowed": True},
        {"allowed": True, "allowed_for": ""},
        {"subreddit": "   ", "allowed": True, "allowed_for": "   "},
    ):
        problems = ADAPTERS["reddit"].validate(creds, opts)
        assert any("subreddit" in p for p in problems), f"accepted {opts!r}"
