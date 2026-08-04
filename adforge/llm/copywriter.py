"""
Post generation with a quality loop.

    angle -> draft -> deterministic rules -> LLM critic -> revise -> repeat

A draft only escapes the loop when it has zero BLOCK violations AND the critic
scores it at or above the threshold. If the loop exhausts its attempts the best
scoring candidate is returned but marked so the UI can flag it - we never
silently publish something that failed the bar.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field

from ..brands import Brand, Pillar
from ..config import settings
from ..platforms.spec import PlatformSpec
from . import prompts, rules
from ..gpu import TEXT, gpu
from .client import LLMError, chat_json, chat_with_fallback

log = logging.getLogger("adforge.copy")

PASS_SCORE = 7.0
MAX_ATTEMPTS = 4


@dataclass
class Draft:
    text: str
    score: float = 0.0
    passed: bool = False
    notes: str = ""
    problems: list[str] = field(default_factory=list)
    violations: list[rules.Violation] = field(default_factory=list)
    attempts: int = 0
    angle: str = ""
    hashtags: list[str] = field(default_factory=list)


def _clean(raw: str) -> str:
    """Strip the wrappers local models add despite being told not to."""
    t = raw.strip()
    # Whole reply wrapped in one pair of quotes.
    if len(t) > 2 and t[0] in "\"'" and t[-1] == t[0] and t.count(t[0]) == 2:
        t = t[1:-1].strip()
    t = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", t).strip()
    # Leading label the model added anyway.
    t = re.sub(
        r"(?i)^(?:here(?:'s| is)[^\n:]{0,40}:|post|caption|tweet|copy|output)\s*:\s*",
        "",
        t,
    ).strip()
    # Trailing self-commentary after the post body.
    t = re.split(r"\n\s*(?:---+|===+)\s*\n", t)[0].strip()
    t = re.sub(
        r"(?is)\n\s*(?:note|explanation|rationale|why this works)\s*:.*$", "", t
    ).strip()
    # Collapse 3+ blank lines - a common model artifact.
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def extract_hashtags(text: str) -> list[str]:
    return re.findall(r"(?<!\w)#(\w+)", text)


def generate_angles(
    brand: Brand, pillar: Pillar, n: int = 5, recent: list[str] | None = None
) -> list[str]:
    try:
        data = chat_json(
            prompts.angle_prompt(brand, pillar, n, recent or []),
            models=settings.writer_chain,
            temperature=1.0,
            max_tokens=900,
        )
        angles = [str(a).strip() for a in data.get("angles", []) if str(a).strip()]
        if angles:
            return angles
    except LLMError as e:
        log.warning("angle generation failed for %s/%s: %s", brand.key, pillar.key, e)
    # Fallback keeps the scheduler alive if the model is having a bad day.
    return [f"{pillar.label}: {pillar.description.strip()}"]


def critique(text: str, brand: Brand, ps: PlatformSpec, pillar: Pillar) -> dict:
    try:
        return chat_json(
            prompts.critic_prompt(text, brand, ps, pillar),
            models=settings.critic_chain,
            temperature=0.2,
            max_tokens=700,
        )
    except LLMError as e:
        log.warning("critic failed: %s", e)
        # Neutral score - the deterministic rules still gate publication.
        return {"overall": 6.0, "verdict": "PASS", "problems": [], "fix": ""}


def factcheck(text: str, brand: Brand) -> list[str]:
    """Product claims in the copy that the brand profile does not support.

    The deterministic gate catches invented *numbers* ("25% faster"), but not
    invented *mechanics* - a real generation asserted "verification requires a
    one-time 10-minute test", which is a plausible-sounding process that does
    not exist. Only a model comparing the copy against the approved claims
    catches that class of error.
    """
    try:
        data = chat_json(
            prompts.factcheck_prompt(text, brand),
            models=settings.critic_chain,
            temperature=0.1,
            max_tokens=600,
        )
    except LLMError as e:
        log.warning("factcheck failed, allowing through: %s", e)
        return []

    out = []
    for item in data.get("unsupported", []) or []:
        if isinstance(item, dict):
            claim = str(item.get("claim", "")).strip()
            why = str(item.get("why", "")).strip()
            if claim:
                out.append(f"unsupported claim {claim!r}: {why}" if why
                           else f"unsupported claim {claim!r}")
        elif str(item).strip():
            out.append(f"unsupported claim: {str(item).strip()}")
    return out


def _score_of(verdict: dict) -> float:
    if isinstance(verdict.get("overall"), (int, float)):
        return float(verdict["overall"])
    dims = ["hook", "specificity", "human", "value", "brand_fit"]
    vals = [float(verdict[d]) for d in dims if isinstance(verdict.get(d), (int, float))]
    return sum(vals) / len(vals) if vals else 5.0


def write_post(
    brand: Brand,
    ps: PlatformSpec,
    pillar: Pillar,
    angle: str | None = None,
    recent: list[str] | None = None,
    rng: random.Random | None = None,
) -> Draft:
    with gpu(TEXT):
        return _write_post(brand, ps, pillar, angle, recent, rng)


def _write_post(brand, ps, pillar, angle, recent, rng) -> Draft:
    rng = rng or random
    recent = recent or []
    if angle is None:
        angle = rng.choice(generate_angles(brand, pillar, 5, recent))

    best = Draft(text="", score=-1.0, angle=angle)
    text = ""
    problems: list[str] = []
    fix = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt == 1:
            msgs = prompts.post_prompt(brand, ps, pillar, angle, recent, rng)
            temp = 0.9
        else:
            msgs = prompts.revise_prompt(text, brand, ps, problems, fix)
            temp = 0.75

        try:
            raw, used = chat_with_fallback(
                msgs,
                settings.writer_chain,
                temperature=temp,
                max_tokens=min(2000, ps.max_chars // 2 + 400),
            )
            text = _clean(raw)
        except LLMError as e:
            log.error("generation failed on attempt %d: %s", attempt, e)
            break

        vs = rules.check(text, brand, ps)
        blockers = rules.blocking(vs)

        if blockers:
            problems = [f"{b.code}: {b.detail}" for b in blockers]
            fix = (
                "Remove every banned phrase and unsupported claim listed. Stay "
                f"under {ps.max_chars} characters."
            )
            cand = Draft(
                text=text,
                score=0.0,
                passed=False,
                notes=rules.summarize(vs),
                problems=problems,
                violations=vs,
                attempts=attempt,
                angle=angle,
            )
            if cand.score > best.score:
                best = cand
            log.info("attempt %d blocked: %s", attempt, "; ".join(problems[:3]))
            continue

        # Claims the brand profile does not support. Treated as blocking: a
        # confident invented specific is worse than a bland post, because a
        # reader who acts on it finds the product does not work that way.
        unsupported = factcheck(text, brand)
        if unsupported:
            problems = unsupported
            fix = (
                "Remove or replace every unsupported claim listed. State only "
                "what the approved claims support; if you do not have a "
                "specific, leave it out rather than inventing one."
            )
            cand = Draft(
                text=text, score=0.0, passed=False,
                notes="; ".join(unsupported[:4]), problems=problems,
                violations=vs, attempts=attempt, angle=angle,
            )
            if cand.score > best.score:
                best = cand
            log.info("attempt %d unsupported: %s", attempt, "; ".join(unsupported[:2]))
            continue

        verdict = critique(text, brand, ps, pillar)
        score = _score_of(verdict)
        warns = [f"{w.code}: {w.detail}" for w in vs]
        cand = Draft(
            text=text,
            score=score,
            passed=score >= PASS_SCORE and verdict.get("verdict") != "REVISE",
            notes="; ".join(filter(None, [rules.summarize(vs), str(verdict.get("fix", ""))])),
            problems=[str(p) for p in verdict.get("problems", [])] + warns,
            violations=vs,
            attempts=attempt,
            angle=angle,
            hashtags=extract_hashtags(text),
        )
        if cand.score > best.score:
            best = cand
        if cand.passed:
            log.info("passed on attempt %d, score %.1f", attempt, score)
            return cand

        problems = cand.problems
        fix = str(verdict.get("fix") or "Make the opening more specific and concrete.")
        log.info("attempt %d scored %.1f, revising", attempt, score)

    # Nothing cleared the bar. Return the best, still marked failed, and let the
    # caller decide - the scheduler routes these to REVIEW for a human.
    best.hashtags = extract_hashtags(best.text)
    return best


def media_prompt(brand: Brand, post_text: str, angle: str) -> str:
    """Turn a post into an image prompt unique to that post."""
    vis = brand.visual
    sys = (
        "You write prompts for an SDXL image model. Output ONE prompt line: "
        "comma-separated visual descriptors only. Describe a scene or abstract "
        "composition, never a text overlay - the model cannot render legible "
        "words. No people's faces. No logos. Output only the prompt."
    )
    user = f"""HOUSE VISUAL STYLE (obey it):
{vis.get('style', '').strip()}

The image accompanies this post:
---
{post_text[:700]}
---
Underlying idea: {angle[:300]}

Write one image prompt that visually expresses THIS specific post's idea -
not a generic brand image. Concrete subject first, then style, then lighting
and quality descriptors."""
    try:
        raw, _ = chat_with_fallback(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            settings.fast_chain,
            temperature=0.9,
            max_tokens=220,
        )
        p = _clean(raw)
        p = p.replace("\n", ", ").strip(" ,")
        if len(p) > 20:
            return p
    except LLMError as e:
        log.warning("media prompt failed: %s", e)
    return f"{vis.get('style', 'clean technical composition')}, abstract, high detail"
