"""
Deterministic copy checks.

These run BEFORE the LLM critic because they are cheap, and because some of
them (the Vallorix certification claims) are legally load-bearing and must not
depend on a model's judgement.

Two severities:
  BLOCK - copy is discarded and regenerated. Never published.
  WARN  - counts against the quality score, surfaced in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import reference
from ..brands import Brand
from ..platforms.spec import PlatformSpec


@dataclass
class Violation:
    severity: str  # BLOCK | WARN
    code: str
    detail: str


# Phrases that mark text as machine-written to anyone who reads a lot of it.
# This list is the single highest-leverage part of the whole generator.
AI_TELLS = [
    r"\bin today'?s (?:fast[- ]paced |ever[- ]evolving |rapidly changing )?world\b",
    r"\bin the (?:fast[- ]paced|ever[- ]evolving|rapidly changing) (?:world|landscape)\b",
    r"\bdelve into\b",
    r"\bit'?s worth noting\b",
    r"\bnavigat(?:e|ing) the (?:complex|complexities|landscape|world)\b",
    r"\bunlock(?:ing)? the (?:power|potential|secrets)\b",
    r"\bharness(?:ing)? the power\b",
    r"\bgame[- ]chang(?:er|ing)\b",
    r"\brevolutioniz(?:e|ing|es)\b",
    r"\bcutting[- ]edge\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bseamless(?:ly)?\b",
    r"\bempower(?:ing|s)? (?:you|teams|developers|businesses)\b",
    r"\btake your .{3,30} to the next level\b",
    r"\bwhether you'?re a .{3,40}, ",
    r"\blook no further\b",
    r"\bthe future of .{3,30} is here\b",
    r"\bdive (?:deep )?into\b",
    r"\bembark on\b",
    r"\bplethora\b",
    r"\btapestry\b",
    r"\brobust(?:ly)? (?:solution|platform|framework)\b",
    r"\bleverag(?:e|ing) (?:the )?(?:power|capabilities)\b",
    r"\bin conclusion\b",
    r"\bmoreover,",
    r"\bfurthermore,",
    r"\bthat'?s a game[- ]changer\b",
    r"\bsupercharge\b",
    r"\bturbocharge\b",
    r"\bunleash\b",
    r"\bskyrocket\b",
    r"\bno[- ]brainer\b",
    r"\bsay goodbye to\b",
    r"\bthink again\b",
    r"\bhere'?s the kicker\b",
    r"\bthe best part\?",
    r"\bbut here'?s the thing\b",
    r"\blet that sink in\b",
    r"\bmind[- ]blowing\b",
    r"\bstop scrolling\b",
    r"\byou'?re doing .{3,25} wrong\b",
    r"\bnobody (?:is )?talking about\b",
    r"\bexcited to (?:announce|share)\b",
    r"\bthrilled to (?:announce|share)\b",
    r"\bproud to announce\b",
    r"\bwe'?re on a mission to\b",
    # Motivational filler. A real short's caption survived every gate with
    # "train smarter, not harder! scaling your ai projects just got easier" -
    # two sentences that say nothing and could sit under any product on earth.
    r"(?i)\b(?:work|train|build|scale|ship)\s+smarter,?\s+not\s+harder\b",
    r"(?i)\bjust got (?:easier|simpler|better|faster)\b",
    r"(?i)\btake the guesswork out of\b",
    r"(?i)\bmade (?:easy|simple)\b",
    r"(?i)\bthe smart way to\b",
    r"(?i)\blevel up your\b",
]

# "It's not X, it's Y" and em-dash-heavy antithesis are the most recognisable
# LLM rhetorical tics currently in the wild.
STRUCTURAL_TELLS = [
    # "It's not X, it's Y" - the single most recognisable LLM rhetorical tic.
    # Matches the contracted and uncontracted forms of both clauses.
    (
        r"(?i)\bit(?:'s| is) not (?:just |merely |only )?(?:about )?"
        r"\w[\w \-]{2,40}[.,;] it(?:'s| is)\b",
        "antithesis_cliche",
    ),
    (
        r"(?i)\bthis (?:isn'?t|is not) (?:just |merely |only )?"
        r"\w[\w \-]{2,40}[.,;] (?:this |it )?is\b",
        "antithesis_cliche",
    ),
    (r"(?i)^\s*ever (?:wonder|thought|felt|noticed)\b", "rhetorical_opener"),
    (r"(?i)^\s*what if (?:i told you|you could)\b", "rhetorical_opener"),
    (r"(?i)^\s*(?:are|do|did|have) you (?:ever )?\w+.{0,60}\?", "rhetorical_opener"),
    (r"(?i)^\s*imagine\b", "imagine_opener"),
]

VALLORIX_FORBIDDEN = [
    (
        r"(?i)\b(?:we|vallorix|our (?:platform|product|company))\b[^.!?]{0,60}\b"
        r"(?:is|are|am|'re)\s+(?:now\s+)?(?:soc\s*2|iso\s*27001|hipaa|iso\s*42001|"
        r"gdpr|nis2|dora)[^.!?]{0,20}\b(?:certified|compliant|attested)",
        "claims_own_certification",
    ),
    (
        r"(?i)\b(?:soc\s*2|iso\s*27001|hipaa)[- ]certified\s+(?:platform|product|tool|solution)\b",
        "claims_own_certification",
    ),
    (
        r"(?i)\bwe (?:hold|have|achieved|earned|maintain)\b[^.!?]{0,40}\b"
        r"(?:soc\s*2|iso\s*27001|iso\s*42001)\b",
        "claims_own_certification",
    ),
    (
        r"(?i)\b(?:get|achieve|earn|be)\s+(?:soc\s*2|iso\s*27001|hipaa)[^.!?]{0,30}"
        r"\bin\s+(?:just\s+)?\d+\s*(?:days?|weeks?|months?)",
        "promises_audit_timeline",
    ),
    (
        r"(?i)\b(?:guarantee[ds]?|guaranteed to)\b[^.!?]{0,40}\b(?:pass|certif|audit)",
        "guarantees_audit_outcome",
    ),
    (
        r"(?i)\breplaces? (?:your |an? )?auditor\b",
        "replaces_auditor",
    ),
]

# Quantified performance claims the model invented.
#
# Local models will happily assert "boosts throughput by 25%" when nothing in
# the brand profile says so. A made-up benchmark is worse than no benchmark:
# it is trivially falsified by any reader who tests it, and it is the fastest
# way to lose a technical audience. We block the unsourced quantified claim and
# push the copy toward the mechanism instead, which is better marketing anyway.
FABRICATED_METRIC = [
    r"(?i)\b(?:boost|improve|increase|reduce|cut|drop|raise|lower|speed up|slash)"
    r"\w*\s+(?:\w+\s+){0,4}?by\s+(?:up to\s+)?\d+(?:\.\d+)?\s*(?:%|percent|x\b)",
    r"(?i)\b\d+(?:\.\d+)?\s*(?:%|percent)\s+(?:faster|slower|cheaper|higher|lower|"
    r"better|more|less|improvement|reduction|increase|gain|savings?)\b",
    r"(?i)\b\d+(?:\.\d+)?x\s+(?:faster|cheaper|better|higher|more|throughput|"
    r"speedup|performance)\b",
    r"(?i)\b(?:up to|as much as|over)\s+\d+(?:\.\d+)?\s*(?:%|percent|x\b)",
    r"(?i)\bsaves?\s+(?:you\s+)?\d+(?:\.\d+)?\s*(?:%|percent|hours|days|weeks)\b",
    # Money. "batching 10 requests saves $0.06 per hour on an RTX 3080" is a
    # benchmark nobody ran, and a costing claim invites someone to hold you to
    # it. Prices traceable to proof_points still pass via _cited_number.
    r"(?i)(?:saves?|costs?|cuts?|reduces?)\s+(?:you\s+)?[$£€]\s?\d",
    r"(?i)[$£€]\s?\d+(?:\.\d+)?\s*(?:per|/|a)\s*(?:hour|hr|month|year|day|"
    r"request|token|1k|million)",
]

# Hedged or self-measured framings are acceptable - they make no claim about a
# result the reader will get.
METRIC_HEDGE = re.compile(
    r"(?i)\b(?:measure|benchmark|in my|in our|on my|on our|your mileage|depends on|"
    r"varies|test it|we saw|i saw|we measured|i measured|for this workload|"
    r"on this hardware)\b"
)


def _cited_number(match: str, brand: Brand) -> bool:
    """True when the exact figure appears in the brand's approved claims."""
    nums = re.findall(r"\d+(?:\.\d+)?", match)
    corpus = " ".join(brand.proof_points) + " " + brand.about
    return bool(nums) and all(n in corpus for n in nums)


# A company account inventing a personal war story. The radar's reply policy
# has blocked this since it was written; posts never did, and a live run opened
# a Vallorix tweet with "I once spent hours tracking a mystery daemon crash".
# It reads as a lie because it is one - nobody at the company did that.
FABRICATED_ANECDOTE = [
    (r"(?i)\bI (?:once|recently|just)\s+(?:spent|found|tried|built|debugged|"
     r"discovered|hit|ran into|chased)\b", "fabricated_anecdote"),
    (r"(?i)\b(?:last|this)\s+(?:week|month|night|year)\s+I\s+\w+", "fabricated_anecdote"),
    (r"(?i)\bI\s+(?:remember|learned this the hard way|got burned)\b", "fabricated_anecdote"),
    (r"(?i)\ba client of (?:mine|ours)\b", "fabricated_anecdote"),
    # Subject-dropped first person, which is ordinary tweet style and slipped
    # every pattern above: "Ran a 5B sparse model on a 16GB GPU and watched it
    # die." No "I", same fabricated war story. Anchored to the start of the
    # text or a sentence so it does not fire on "Ran" mid-clause.
    (r"(?i)(?:^|[.!?]\s+)(?:ran|tried|spent|built|debugged|tested|deployed|"
     r"benchmarked|watched|hit|chased|swapped)\s+"
     r"(?:a|an|the|our|my|this|that|these|those|it|\d)\b",
     "fabricated_anecdote"),
    (r"(?i)(?:^|[.!?]\s+)(?:turns out|learned that|found out)\b",
     "fabricated_anecdote"),
]


# Claims nobody may make on either brand - unverifiable social proof.
UNVERIFIABLE = [
    (r"(?i)\b(?:trusted|used) by (?:\d[\d,]*\+?|thousands|hundreds|millions|top)\b", "social_proof"),
    # "40% of our users batch inference requests" scored 8.0 and shipped past
    # the old rule, which required the number to sit immediately before the
    # noun. A proportion-of-customers claim is a survey result nobody ran, and
    # on a commercial account it is a false-advertising problem rather than
    # just a credibility one.
    (r"(?i)\b\d+\s*(?:%|percent)\s+of\s+(?:our|the|all|inferix'?s?|vallorix'?s?)?\s*"
     r"(?:users|customers|teams|developers|companies|clients|accounts)\b",
     "invented_user_statistic"),
    (r"(?i)\b(?:most|many|half|a (?:third|quarter)|\d+ in \d+)\s+of\s+(?:our|my)\s+"
     r"(?:users|customers|teams|developers|companies|clients)\b",
     "invented_user_statistic"),
    (r"(?i)\b\d[\d,]*\+?\s+(?:happy\s+)?(?:customers|users|teams|developers|companies)\b", "user_counts"),
    (r"(?i)(?:#1|\bnumber one\b|\bthe leading\b|\bmarket[- ]leading\b|"
     r"\bbest[- ]in[- ]class\b)", "superlative_claim"),
    (r"(?i)\b(?:\d+x|\d+%)\s+(?:faster|cheaper|better)\s+than\s+\w+", "unbenchmarked_comparison"),
    (r"(?i)\bbacked by\b[^.!?]{0,30}\b(?:vc|investors|ycombinator|y combinator)\b", "funding_claim"),
]


# Hashtags are matched as one token, so a typo is a tag nobody follows and the
# post reaches nobody through it. A real caption shipped "#gpucould". Checking
# every word against a dictionary would be overreach; checking that a tag built
# from the brand's own vocabulary is spelled the way the brand spells it is not.
def _misspelt_hashtags(text: str, brand: Brand) -> list[str]:
    import difflib

    vocab = set()
    for kw in brand.keywords:
        # Both the individual words AND the concatenated form, because that is
        # how a multi-word keyword becomes a hashtag: "GPU cloud" -> #gpucloud.
        # Splitting only gave {gpu, cloud}, so the compound tag was never in
        # the vocabulary and "#gpucould" had nothing close to match against.
        vocab |= {re.sub(r"[^a-z0-9]", "", w.lower()) for w in kw.split()}
        vocab.add(re.sub(r"[^a-z0-9]", "", kw.lower()))
    vocab.add(re.sub(r"[^a-z0-9]", "", brand.name.lower()))
    vocab = {v for v in vocab if len(v) > 3}

    bad = []
    for tag in re.findall(r"(?<!\w)#(\w+)", text):
        low = tag.lower()
        if low in vocab:
            continue
        # Only flag a near-miss: one edit away from a word the brand uses.
        close = difflib.get_close_matches(low, vocab, n=1, cutoff=0.86)
        if close and close[0] != low:
            bad.append(f"#{tag} (did you mean #{close[0]}?)")
    return bad


def _hashtag_count(text: str) -> int:
    return len(re.findall(r"(?<!\w)#\w+", text))


def _emoji_count(text: str) -> int:
    return len(
        re.findall(
            "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]", text
        )
    )


def check(text: str, brand: Brand, ps: PlatformSpec) -> list[Violation]:
    """Run every deterministic gate. Empty list means the copy is clean."""
    v: list[Violation] = []
    stripped = text.strip()

    if not stripped:
        return [Violation("BLOCK", "empty", "no copy produced")]

    # --- length -----------------------------------------------------------
    if len(stripped) > ps.max_chars:
        v.append(
            Violation(
                "BLOCK",
                "too_long",
                f"{len(stripped)} chars exceeds {ps.label} limit of {ps.max_chars}",
            )
        )
    if len(stripped) < 40:
        v.append(Violation("BLOCK", "too_short", f"only {len(stripped)} chars"))

    # --- model leakage ----------------------------------------------------
    for pat, code in [
        (
            r"(?i)^\s*(?:here'?s|here is|below is|this is)\s+(?:a|an|the|your)\s+"
            r"(?:\w+[\s-]+){0,4}(?:post|caption|tweet|thread|copy|draft|version)\b",
            "meta_preamble",
        ),
        (r"(?i)\b(?:as an ai|language model|i cannot|i'?m unable to)\b", "assistant_voice"),
        (r"(?i)^\s*(?:sure[,!]|certainly[,!]|absolutely[,!])", "assistant_voice"),
        (r"(?i)\[(?:insert|your|company|product|link|url|name)[^\]]*\]",
         "unfilled_placeholder"),
        (r"\{\{?\s*\w+\s*\}?\}", "unfilled_placeholder"),
        (r"(?i)\b(?:option|version|draft)\s*[12]\s*:", "multiple_drafts"),
        (r"(?i)^\s*(?:caption|post|tweet|copy)\s*:", "labelled_output"),
    ]:
        if re.search(pat, stripped):
            v.append(Violation("BLOCK", code, f"matched {pat}"))

    # --- AI tells ---------------------------------------------------------
    for pat in AI_TELLS:
        m = re.search(pat, stripped, re.I)
        if m:
            v.append(Violation("BLOCK", "ai_tell", f"cliche phrase: {m.group(0)!r}"))
    for pat, code in STRUCTURAL_TELLS:
        m = re.search(pat, stripped)
        if m:
            v.append(Violation("WARN", code, f"{m.group(0)[:60]!r}"))

    # --- brand-agnostic forbidden claims ----------------------------------
    for pat, code in UNVERIFIABLE:
        m = re.search(pat, stripped)
        if m:
            v.append(Violation("BLOCK", code, f"unverifiable claim: {m.group(0)!r}"))

    for pat, code in FABRICATED_ANECDOTE:
        m = re.search(pat, stripped)
        if m:
            v.append(
                Violation(
                    "BLOCK", code,
                    f"first-person anecdote on a brand account: {m.group(0)!r} - "
                    f"nobody at the company did this",
                )
            )

    # --- invented benchmarks ----------------------------------------------
    for pat in FABRICATED_METRIC:
        for m in re.finditer(pat, stripped):
            phrase = m.group(0)
            if _cited_number(phrase, brand):
                continue  # figure is traceable to the brand's approved claims
            if METRIC_HEDGE.search(stripped):
                v.append(
                    Violation(
                        "WARN",
                        "hedged_metric",
                        f"quantified claim {phrase!r} is hedged but still unsourced",
                    )
                )
                continue
            v.append(
                Violation(
                    "BLOCK",
                    "fabricated_metric",
                    f"unsourced performance number {phrase!r} - state the "
                    "mechanism, or tell the reader to measure it themselves",
                )
            )

    # --- brand hard rules -------------------------------------------------
    if brand.key == "vallorix":
        for pat, code in VALLORIX_FORBIDDEN:
            m = re.search(pat, stripped)
            if m:
                v.append(
                    Violation(
                        "BLOCK", code, f"Vallorix legal rule violated: {m.group(0)!r}"
                    )
                )

    # --- framework control citations --------------------------------------
    # Applies to any brand that cites one, but matters most for Vallorix:
    # misciting a criterion to an audience of GRC leads is more damaging than
    # a dull post, and the model does it confidently.
    for cid in reference.unknown(stripped):
        if re.fullmatch(r"A\.\d{1,2}\.\d{1,2}\.\d{1,2}", cid):
            detail = (
                f"{cid} is ISO 27001:2013 numbering, retired by the 2022 "
                "revision - use the current Annex A control"
            )
        else:
            detail = f"{cid} is not a real control in any supported framework"
        v.append(Violation("BLOCK", "unknown_control", detail))

    # --- platform texture -------------------------------------------------
    lo, hi = ps.hashtags
    n = _hashtag_count(stripped)
    if n > hi:
        v.append(Violation("BLOCK", "too_many_hashtags", f"{n} > {hi} for {ps.label}"))
    elif n < lo:
        v.append(Violation("WARN", "too_few_hashtags", f"{n} < {lo} for {ps.label}"))

    for bad in _misspelt_hashtags(stripped, brand):
        v.append(
            Violation("BLOCK", "misspelt_hashtag",
                      f"{bad} - a mistyped tag is followed by nobody")
        )

    if _emoji_count(stripped) > 4:
        v.append(Violation("WARN", "emoji_spam", "more than 4 emoji"))

    # Emoji used as list bullets is the strongest visual AI tell on LinkedIn.
    if len(re.findall(r"(?m)^\s*[\U0001f300-\U0001faff☀-➿]", stripped)) >= 3:
        v.append(Violation("BLOCK", "emoji_bullets", "emoji used as list bullets"))

    if re.search(r"https?://", stripped):
        if ps.link_policy == "unclickable":
            v.append(
                Violation(
                    "BLOCK",
                    "link_not_allowed",
                    f"{ps.label} renders body links as plain text; use the "
                    "profile/bio CTA instead",
                )
            )
        elif ps.link_policy == "demoted":
            v.append(
                Violation(
                    "WARN",
                    "link_demoted",
                    f"{ps.label} demotes posts with body links; the adapter will "
                    "move this URL to the first comment",
                )
            )

    # An em-dash rate well above human prose is a reliable model signature.
    if stripped.count("—") >= 3:
        v.append(Violation("WARN", "em_dash_heavy", f"{stripped.count('—')} em dashes"))

    return v


def blocking(vs: list[Violation]) -> list[Violation]:
    return [x for x in vs if x.severity == "BLOCK"]


def summarize(vs: list[Violation]) -> str:
    return "; ".join(f"[{x.severity}] {x.code}: {x.detail}" for x in vs)
