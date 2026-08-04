"""
Engagement policy. This module is the reason the radar is safe to run.

The original brief asked for replies that read as an unaffiliated user who
"just used Inferix" - marketing that does not look like marketing. That is not
built here, on purpose:

  - FTC Endorsement Guides (16 CFR Part 255) require disclosure of a material
    connection between an endorser and the product. A company's own principals
    posting undisclosed testimonials is a deceptive practice, and the FTC's
    2024 rule on fake reviews and testimonials attaches civil penalties per
    violation. This is a legal exposure, not a platform-rules technicality.
  - Reddit's content policy prohibits undisclosed commercial promotion and
    vote/comment manipulation. Enforcement escalates from comment removal to
    account bans to a SITEWIDE DOMAIN BAN, after which no one can link
    inferix.co anywhere on Reddit. That outcome is effectively permanent and
    would cost far more traffic than astroturfing could ever produce.
  - Discord's ToS prohibits the same behaviour, per-server rules usually go
    further, and server bans are trivially easy for mods to apply.

What is built instead: find the conversations where the product genuinely
answers the question, draft a reply that is useful on its own merits, attach a
disclosure, and require a human to approve it. Disclosed, substantive founder
replies are explicitly welcome in most technical subreddits and consistently
outperform astroturf, because they survive instead of being removed.

These constraints are enforced in code below rather than left to the prompt.
"""

from __future__ import annotations

import re

# Appended to every drafted reply at send time, after any human editing, so it
# cannot be removed by accident in the UI.
DISCLOSURES = {
    "inferix": "(disclosure: I work on Inferix)",
    "vallorix": "(disclosure: I work on Vallorix)",
}

DEFAULT_DISCLOSURE = "(disclosure: I work on this product)"


def disclosure_for(brand_key: str) -> str:
    return DISCLOSURES.get(brand_key, DEFAULT_DISCLOSURE)


def has_disclosure(text: str, brand_key: str) -> bool:
    lowered = text.lower()
    return "disclosure:" in lowered or bool(
        re.search(
            r"\b(i work (on|at)|i'?m (the|a) (founder|dev|maintainer)|"
            r"i built|full disclosure|my (company|team|product))\b",
            lowered,
        )
    )


def ensure_disclosure(text: str, brand_key: str) -> str:
    """Idempotent: never appends a second disclosure."""
    body = text.rstrip()
    if has_disclosure(body, brand_key):
        return body
    return f"{body}\n\n{disclosure_for(brand_key)}"


# Language that turns a helpful reply into an advertisement. Blocked outright -
# these are the exact phrasings that get comments removed and users reported.
SHILL_PATTERNS = [
    (r"(?i)\bi (?:just )?(?:used|tried|switched to|discovered|found)\b[^.!?]{0,40}"
     r"\b(?:inferix|vallorix)\b", "fake_personal_testimonial"),
    (r"(?i)\b(?:you should|you'?ve got to|definitely) (?:check out|try|use)\b",
     "hard_sell"),
    (r"(?i)\b(?:highly recommend|can'?t recommend .{0,20} enough)\b", "hard_sell"),
    (r"(?i)\bgame[- ]chang(?:er|ing)\b", "hype"),
    (r"(?i)\b(?:best|greatest) .{0,25}\b(?:i'?ve (?:ever )?(?:used|seen|tried))\b",
     "fake_personal_testimonial"),
    (r"(?i)\bhas been a lifesaver\b", "fake_personal_testimonial"),
    (r"(?i)\bsolved all (?:my|our) problems\b", "fake_personal_testimonial"),
    (r"(?i)\buse (?:my |the )?(?:code|link|referral)\b", "referral_spam"),
    (r"(?i)\bDM me\b", "dm_solicitation"),
]

# A reply that is only a link is spam on every platform in scope.
MIN_SUBSTANCE_CHARS = 120


def check_reply(text: str, brand_key: str, links_allowed: bool) -> list[str]:
    """Problems that block sending. Empty list means the draft may be queued."""
    problems: list[str] = []
    body = text.strip()

    if len(body) < MIN_SUBSTANCE_CHARS:
        problems.append(
            f"too thin ({len(body)} chars) - a reply must answer the question "
            "on its own merits before it mentions anything"
        )

    for pat, code in SHILL_PATTERNS:
        if m := re.search(pat, body):
            problems.append(f"{code}: {m.group(0)!r}")

    urls = re.findall(r"https?://\S+", body)
    if urls and not links_allowed:
        problems.append(
            "this target does not permit links; the reply must stand without one"
        )
    if len(urls) > 1:
        problems.append(f"{len(urls)} links - at most one, and only if it answers the question")

    # Strip the disclosure before measuring how much of the reply is product
    # talk, otherwise the disclosure itself inflates the ratio.
    without = re.sub(r"\(disclosure:[^)]*\)", "", body, flags=re.I)
    mentions = len(re.findall(r"(?i)\b(?:inferix|vallorix)\b", without))
    if mentions > 2:
        problems.append(
            f"names the product {mentions} times - once is enough in a reply"
        )

    if not has_disclosure(body, brand_key):
        # Not fatal: ensure_disclosure() adds it at send time. Surfaced so the
        # reviewer can see it was machine-added rather than written.
        problems.append("no disclosure written by the model (one will be appended)")

    return problems


def blocking(problems: list[str]) -> list[str]:
    """Problems that must prevent sending, as opposed to advisory notes."""
    return [p for p in problems if not p.startswith("no disclosure written")]
