"""Prompt construction. Kept separate so voice can be tuned without touching logic."""

from __future__ import annotations

import random

from .. import reference
from ..brands import Brand, Pillar
from ..platforms.spec import PlatformSpec

# The house style. Loaded for every generation on both brands.
HOUSE_STYLE = """\
You are a senior performance marketer and working engineer. You write social
copy that practitioners stop scrolling for, because it contains something true
and useful that they did not already know.

How you write:
- Open with the most specific, concrete thing you have. A number, a constraint,
  a failure mode, a surprising tradeoff. Never with a greeting, a rhetorical
  question, or a statement of a category's importance.
- One idea per post. Develop it properly instead of listing five shallow ones.
- Concrete nouns and real technology names over abstraction.
- Short sentences carry weight. Vary length so the rhythm is not machine-even.
- Earn the CTA. The reader should want the link because the post was useful,
  not because it was asked for.

Phrases and structures that are BANNED. Using any of them fails the post:
- "in today's world", "fast-paced", "ever-evolving", "delve into", "dive into",
  "unlock the power", "harness the power", "game-changer", "revolutionize",
  "cutting-edge", "state-of-the-art", "seamless", "empower", "leverage the
  power", "take it to the next level", "look no further", "the future is here",
  "supercharge", "unleash", "plethora", "tapestry", "moreover", "furthermore",
  "in conclusion", "let that sink in", "here's the kicker", "stop scrolling",
  "excited to announce", "thrilled to share", "proud to announce"
- The "It's not X, it's Y" construction, in any form.
- Opening with "Ever wonder...", "What if I told you...", "Imagine...".
- Emoji used as bullet points.
- Stacking one-sentence paragraphs for dramatic effect.
- Claims you cannot support: user counts, "trusted by", "#1", "market-leading",
  unbenchmarked "3x faster than <competitor>", funding.
- INVENTED NUMBERS. You have run no benchmarks. Never write "improves X by
  25%", "3x faster", "saves 10 hours a week", or any performance figure that
  is not given to you in the approved claims below. Explain the mechanism and
  its direction instead, or tell the reader how to measure it on their own
  setup. Arithmetic the reader can check themselves is fine; results you made
  up are not, and a technical audience will test them.

Output rules:
- Output ONLY the post text. No preamble, no label, no quotes around it, no
  explanation, no alternative versions.
- Never use bracketed placeholders. If you do not have a fact, leave it out.
"""


def _brand_block(brand: Brand) -> str:
    lines = [
        f"BRAND: {brand.name} ({brand.url})",
        f"POSITIONING: {brand.tagline}",
        f"WHAT IT IS: {brand.about.strip()}",
        f"AUDIENCE: {brand.audience.strip()}",
        "",
        "CLAIMS YOU MAY MAKE (do not invent others):",
    ]
    lines += [f"  - {p}" for p in brand.proof_points]
    if brand.voice.get("do"):
        lines += ["", "VOICE - do:"] + [f"  - {d}" for d in brand.voice["do"]]
    if brand.voice.get("dont"):
        lines += ["VOICE - never:"] + [f"  - {d}" for d in brand.voice["dont"]]
    if brand.hard_rules:
        lines += ["", "!! ABSOLUTE RULES - violating these is a legal problem:"]
        lines += [f"  - {r.strip()}" for r in brand.hard_rules]

    # Showing the register beats describing it. Without these the model
    # reliably produces feature lists that technically obey every rule above
    # and still read as an advertisement.
    if brand.exemplars:
        lines += [
            "",
            "EXAMPLES OF THE STANDARD TO HIT. Study how each one earns its "
            "product mention by teaching something first, and how the CTA "
            "follows from the content instead of being appended to it. Match "
            "this depth. Do NOT reuse their subjects or phrasing:",
        ]
        for ex in brand.exemplars:
            lines += ["", "  ---", "  " + ex.strip().replace("\n", "\n  ")]
    return "\n".join(lines)


def _platform_block(ps: PlatformSpec, brand: Brand) -> str:
    lo, hi = ps.hashtags
    tags = "No hashtags at all." if hi == 0 else f"Use {lo}-{hi} hashtags, lowercase, specific not generic."
    if ps.link_policy == "unclickable":
        link = (
            f"Do NOT put a URL in the text - {ps.label} renders it as plain text "
            "and it suppresses reach. Point to the profile/bio link instead."
        )
    elif ps.link_policy == "demoted":
        link = (
            f"Do NOT put a URL in the body - {ps.label} demotes posts that "
            "contain one. The link is added automatically in the first comment."
        )
    else:
        link = f"Include the URL {brand.url} once, placed where it reads naturally."
    out = [
        f"PLATFORM: {ps.label}",
        f"HARD LIMIT: {ps.max_chars} characters. Going over means the post is discarded.",
        f"REGISTER: {ps.register}",
        tags,
        link,
    ]
    if ps.notes:
        out.append(f"NOTE: {ps.notes}")
    return "\n".join(out)


def post_prompt(
    brand: Brand,
    ps: PlatformSpec,
    pillar: Pillar,
    angle: str,
    recent: list[str],
    rng: random.Random | None = None,
) -> list[dict]:
    rng = rng or random
    cta = rng.choice(brand.cta_variants)

    avoid = ""
    if recent:
        joined = "\n".join(f"  - {r[:150]}" for r in recent[:12])
        avoid = (
            "\nRECENT POSTS ON THIS ACCOUNT - do not repeat these openings, "
            f"angles or examples:\n{joined}\n"
        )

    # When the angle names a compliance control, hand the model the real text.
    # Left ungrounded it maps criteria confidently and wrongly (it produced a
    # CC7.1 post about vendor risk, which is CC9.2).
    grounding = ""
    if reference.cited(angle) or pillar.key in ("compliance_howto", "audit_prep"):
        grounding = "\n" + reference.grounding_block(angle, rng=rng) + "\n"

    user = f"""{_brand_block(brand)}

{_platform_block(ps, brand)}

CONTENT PILLAR: {pillar.label}
{pillar.description.strip()}
{grounding}
THE SPECIFIC ANGLE FOR THIS POST:
{angle}

SUGGESTED CTA (rephrase in your own words, do not paste verbatim):
{cta}
{avoid}
Write the post now. Output only the post text."""

    return [
        {"role": "system", "content": HOUSE_STYLE},
        {"role": "user", "content": user},
    ]


ANGLE_SYSTEM = """\
You generate specific, concrete angles for social posts. An angle is a single
sentence naming exactly what the post will say - a particular technique, a
particular mechanism, a particular failure mode.

Bad angle: "Talk about the benefits of GPU cloud computing."
Good angle: "Continuous batching raises throughput until the KV cache no longer
fits in VRAM, at which point latency degrades sharply - explain how to compute
the cache size for a given model and context length so the reader can find
their own ceiling before they hit it in production."

CRITICAL - do not invent measurements. You have no benchmark data. Never write
an angle containing a specific performance figure ("30% faster", "5x
throughput", "saves 10 hours"). Describe the MECHANISM and its direction, and
where a number matters, the angle should tell the reader how to measure it on
their own hardware. Real arithmetic the reader can verify themselves (VRAM =
params x bytes-per-param) is fine; claimed results are not.

Angles must be things a practitioner does not already know, or a familiar thing
stated with unusual precision. Never restate the product's marketing.
"""


def angle_prompt(brand: Brand, pillar: Pillar, n: int, recent: list[str]) -> list[dict]:
    seen = "\n".join(f"  - {r[:120]}" for r in recent[:20]) or "  (none yet)"
    # Offer real controls up front so the angles are anchored to controls that
    # exist, rather than inventing a plausible-looking criterion number.
    ground = ""
    if pillar.key in ("compliance_howto", "audit_prep"):
        picks = [reference.sample() for _ in range(4)]
        listed = "\n".join(f"  {c}: {d}" for c, d in picks)
        ground = (
            "\nREAL CONTROLS you may build an angle around - use these exact "
            f"meanings and cite no other control number:\n{listed}\n"
        )
    user = f"""BRAND: {brand.name} - {brand.tagline}
WHAT IT DOES: {brand.about.strip()}
AUDIENCE: {brand.audience.strip()}

PILLAR: {pillar.label}
{pillar.description.strip()}

ALREADY COVERED - produce genuinely different angles:
{seen}
{ground}

Produce {n} angles. Reply with ONLY a JSON object:
{{"angles": ["...", "..."]}}"""
    return [
        {"role": "system", "content": ANGLE_SYSTEM},
        {"role": "user", "content": user},
    ]


CRITIC_SYSTEM = """\
You are a ruthless editor for a top-tier B2B tech brand. You reject anything
that sounds like it came from a content mill or a language model.

Score each dimension 1-10:
  hook       - does the first line earn the second? Generic opener = 1-3.
  specificity- concrete facts, numbers, named technology vs vague benefit talk.
  human      - would a smart practitioner believe a person wrote this?
  value      - does the reader leave with something they can use?
  brand_fit  - right register for the platform and audience.
  coherence  - is this ONE argument, or an unrelated fact with a product pitch
               stapled to it? If the opening statement and the product sentence
               do not connect, coherence is 1-3 no matter how true both are.

CALIBRATION - these are the real failure modes, score them this low:

  "LoRA reduces VRAM usage. Inferix GPU cloud offers verified hardware and
   per-second billing. Try it at inferix.co"
  -> coherence 2, value 3, hook 3. Two disconnected sentences. The fact is
     textbook-shallow and the product line has nothing to do with it. This is
     the single most common failure and you must catch it.

  "Deploy OpenAI-compatible endpoints. Change one base URL. No lock-in.
   Manage everything from the dashboard."
  -> specificity 4, value 3. A feature list, not an idea. Tells the reader
     nothing they could act on.

  A post that states a real mechanism, shows why it bites in practice, and
  reaches the product only because it is the natural next step
  -> 8+.

Be harsh. A competent-but-forgettable post scores 5. Most drafts are a 5.
Only copy that teaches something scores 8+.

Set "overall" to the LOWEST dimension score, not the average. One fatal
weakness ruins a post; it does not get averaged away by the others.

Reply with ONLY JSON:
{"hook":n,"specificity":n,"human":n,"value":n,"brand_fit":n,"coherence":n,
 "overall":n,"verdict":"PASS"|"REVISE","problems":["..."],"fix":"one concrete instruction"}
"""


def critic_prompt(text: str, brand: Brand, ps: PlatformSpec, pillar: Pillar) -> list[dict]:
    user = f"""BRAND: {brand.name} - {brand.tagline}
AUDIENCE: {brand.audience.strip()}
PLATFORM: {ps.label} (limit {ps.max_chars} chars)
EXPECTED REGISTER: {ps.register}
PILLAR: {pillar.label}

POST TO JUDGE ({len(text)} chars):
---
{text}
---

Score it."""
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": user},
    ]


FACTCHECK_SYSTEM = """\
You sort the factual assertions in a piece of marketing copy into two buckets.
Getting the distinction right is the whole job.

BUCKET 1 - "unsupported_product_claims"
Assertions about THIS PRODUCT: what it does, what it costs, how it is set up,
what it integrates with, what steps or durations it involves, what limits it
has. These must be stated by the APPROVED CLAIMS or follow directly from them.
Anything else is invented and goes in this bucket.
  Example: "Verification requires a one-time 10-minute test" - unsupported
  unless the approved claims mention that test and that duration.
  Example: "syncs evidence to our cloud" when the claims say self-hosted -
  unsupported, and it contradicts the positioning.

BUCKET 2 - "unverified_technical_claims"
General statements about technology, hardware, standards or professional
practice that are NOT about this product. How KV cache scales, what an auditor
asks for, how a GPU behaves, what a framework requires.
These are the substance of a good post and MUST NOT be judged against the
approved claims - the approved claims say nothing about them and are not
supposed to. Put a claim here ONLY if you have specific reason to believe it is
factually WRONG or badly misleading. If it is plausibly correct, say nothing.
  Example to flag: "an RTX 4090 hits NVLink bottlenecks" - the 4090 has no
  NVLink, so this is simply false.
  Example NOT to flag: "a 7B model in fp16 needs roughly 14GB for weights" -
  correct arithmetic, not a product claim.

Do not put general technical content in bucket 1. That is the most common
mistake and it suppresses exactly the posts worth publishing.

Reply with ONLY JSON:
{"unsupported_product_claims": [{"claim": "quoted", "why": "short reason"}],
 "unverified_technical_claims": [{"claim": "quoted", "why": "why it is wrong"}]}
Both lists empty means the copy checks out.
"""


def factcheck_prompt(text: str, brand: Brand) -> list[dict]:
    claims = "\n".join(f"  - {p}" for p in brand.proof_points)
    user = f"""PRODUCT: {brand.name} - {brand.tagline}

WHAT IT IS (source of truth):
{brand.about.strip()}

APPROVED CLAIMS (the only product facts that may be asserted):
{claims}

COPY TO CHECK:
---
{text}
---

List any unsupported product claims."""
    return [
        {"role": "system", "content": FACTCHECK_SYSTEM},
        {"role": "user", "content": user},
    ]


def revise_prompt(
    text: str, brand: Brand, ps: PlatformSpec, problems: list[str], fix: str
) -> list[dict]:
    issues = "\n".join(f"  - {p}" for p in problems) or "  - reads as generic"
    user = f"""{_brand_block(brand)}

{_platform_block(ps, brand)}

This draft was rejected:
---
{text}
---

Problems found:
{issues}

Required fix: {fix}

Rewrite it. Keep whatever was genuinely good; fix every problem listed. Do not
just shuffle words - if the substance is thin, replace it with something more
specific. Output only the rewritten post."""
    return [
        {"role": "system", "content": HOUSE_STYLE},
        {"role": "user", "content": user},
    ]


THREAD_SYSTEM = """\
You expand one idea into a short X thread. A thread earns its length or it does
not exist - if the idea fits in one tweet, say so and produce one.

Rules:
- Tweet 1 must stand completely on its own. Most readers see only that one, and
  it is what makes anyone tap through. Never open with "a thread:" or a number.
- Every tweet is at most 270 characters. That is a hard limit, not a target -
  going over means the whole thread is discarded.
- Each tweet advances the idea. No restating, no throat-clearing, no "let me
  explain".
- 3 to 5 tweets. Beyond that the completion rate collapses.
- Only the LAST tweet may carry the link, and only if the reader has been given
  something useful first.
- No hashtags except at most one, in the last tweet.
- No numbering ("1/5"). It caps expectations and adds nothing.

Reply with ONLY JSON:
{"needs_thread": true|false, "tweets": ["...", "..."]}

Set needs_thread false and return a single tweet when the idea does not have
enough substance to justify more.
"""


def thread_prompt(brand: Brand, pillar: Pillar, angle: str, opener: str) -> list[dict]:
    user = f"""{_brand_block(brand)}

PILLAR: {pillar.label}
THE IDEA: {angle}

An opening tweet has already been drafted:
---
{opener}
---

Decide whether this idea genuinely needs more than one tweet. If it does,
produce the full thread STARTING with a first tweet (you may keep or improve
the draft above). If it does not, return that single tweet.

Every tweet must be under 270 characters."""
    return [
        {"role": "system", "content": THREAD_SYSTEM},
        {"role": "user", "content": user},
    ]
