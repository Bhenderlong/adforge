"""
Short-form video scripting.

The model writes a beat sheet: per scene a caption (burned in, so it must be
readable in about a second) and a visual direction. Beat length is chosen so
the whole thing lands inside the platform's ideal duration, because watch-time
completion is the dominant ranking signal on both TikTok and Shorts and a video
that runs long gets throttled regardless of how good it is.
"""

from __future__ import annotations

import logging
import re

from ..brands import Brand, Pillar
from ..config import settings
from ..gpu import TEXT, gpu
from ..llm import rules
from ..llm.client import LLMError, chat_json
from ..platforms.spec import PlatformSpec
from .video import Scene, Script

log = logging.getLogger("adforge.script")

SYSTEM = """\
You write short-form vertical video scripts (TikTok, Reels, Shorts) for a
technical brand. Your scripts teach one useful thing and earn the follow.

Rules that decide whether the video performs:
- Scene 1 must land the payoff or a sharp problem in under 1.5 seconds. Never
  open on a logo, a greeting, or "in this video we will".
- Captions are BURNED IN and read while scrolling. Maximum 8 words per scene.
  Plain language. No semicolons, no nested clauses.
- Each scene advances the idea. No scene may restate the previous one.
- The final scene is the call to action, and it is the only place the brand is
  named.
- Visual directions describe a SCENE, never text on screen - the image model
  cannot render legible words. No people's faces. No logos.
- Never invent statistics or benchmark results.

VISUAL DIRECTIONS - the single biggest failure mode:
Every visual MUST name ONE concrete physical object as its subject, and that
object must fill the frame. Abstract prompts produce dark textured wallpaper
that nobody stops scrolling for.
  BAD:  "abstract memory lattice filling with cyan light, layered depth"
  BAD:  "compute topology, flowing data, volumetric glow"
  GOOD: "a single graphics card standing upright on a dark bench, fans and
         heatsink fins sharply lit from one side, shallow depth of field"
  GOOD: "one open server chassis viewed from above, RAM sticks in a row,
         cables running off frame"
Name the object first, then how it is lit and framed. One object per scene,
not a collection. Prefer real hardware you could photograph: a GPU, a fan
stack, a rack rail, a cable bundle, a drive tray, a padlock, a filing drawer,
a stamped document.

NO NUMBERS, CAPACITIES OR BRAND NAMES in a visual direction. The image model
tries to render any it sees as text on the object and produces garbled
nonsense - "16GB graphics card" came back with a fake "166" stamped on the fan
hub. Write "a graphics card", not "a 16GB graphics card"; "a GPU", not "an
NVIDIA A100". Naming a real manufacturer's product is also a trademark problem you
do not need. The number belongs in the CAPTION, which is real text rendered
properly.

FRAMING - this is a 9:16 vertical video with a caption bar across the bottom
third. Put the subject in the UPPER TWO THIRDS of the frame and keep the lower
third simple, or the caption will sit on top of the detail that matters.

Reply with ONLY this JSON:
{"title":"...","scenes":[{"caption":"<=8 words","visual":"image prompt",
"motion":"camera move","seconds":number}],"hashtags":["...","..."]}
"""


# Used when scrubbing leaves nothing usable. Deliberately plain: a clean
# recognisable object beats an incoherent one.
FALLBACK_SUBJECT = (
    "a single graphics card standing upright on a dark bench, fans and "
    "heatsink fins sharply lit from one side"
)

_UNIT_NUM = re.compile(
    r"\b\d+\s*(?:gb|tb|mb|gib|tib|b|bit|w|watt|watts|hz|ghz|mhz|x|k)?\b",
    re.I,
)
_BRANDS = re.compile(
    r"\b(?:nvidia|geforce|rtx|gtx|amd|radeon|intel|arc|a100|h100|h200|"
    r"v100|t4|l40s?|4090|4080|3090|5090|apple|m[1-4]\s*(?:pro|max|ultra)?)\b",
    re.I,
)


def scrub_visual(prompt: str) -> str:
    """Remove numbers and hardware brand names from an image prompt.

    SDXL renders any digits or product names it sees as text ON the object,
    and gets it wrong: "16GB graphics card" produced a card with a garbled
    "166" stamped on the fan hub. The prompt asks the model not to include
    them; this makes sure. The figure belongs in the caption, which is real
    text drawn by Pillow.
    """
    out = _BRANDS.sub("", prompt)
    out = _UNIT_NUM.sub("", out)

    # Deleting words leaves orphans: "an RTX 4090 with 24GB VRAM" becomes
    # "an with VRAM", which is a worse prompt than the original. Drop articles
    # and quantifiers that no longer have a noun to attach to.
    out = re.sub(
        r"\b(?:a|an|the|one|single)\s+(?=(?:with|and|or|beside|on|in|of|at|"
        r"for|from|to|by)\b)",
        "",
        out,
        flags=re.I,
    )
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.])", r"\1", out)
    out = re.sub(r"(?:,\s*){2,}", ", ", out)
    # A leading connective is left over when the subject itself was the thing
    # removed ("an RTX 4090 with 24GB" -> "with VRAM").
    out = re.sub(
        r"^(?:with|and|or|beside|on|in|of|at|for|from|to|by)\s+", "", out.strip(" ,."),
        flags=re.I,
    ).strip(" ,.")

    # If the direction was mostly numbers and product names there is nothing
    # coherent left, and word salad makes a worse image than a plain subject.
    # Say so in the log rather than silently rendering something odd.
    if len(out) < 25:
        log.warning(
            "visual direction was almost entirely numbers/brands (%r) - "
            "falling back to a generic subject", prompt[:80]
        )
        return FALLBACK_SUBJECT
    return out


def _sanitize(text: str) -> str:
    """Captions are drawn by ffmpeg; strip what drawtext cannot render."""
    t = re.sub(r"[\U0001f300-\U0001faff\U00002600-\U000027bf]", "", text)
    return re.sub(r"\s+", " ", t).strip()


def write_script(
    brand: Brand,
    ps: PlatformSpec,
    pillar: Pillar,
    angle: str,
    target_seconds: int | None = None,
) -> Script:
    target = target_seconds or ps.video_seconds
    # 4-5s per beat: shorter feels frantic and the caption cannot be read;
    # longer loses the scroll. Cap the count so render time stays sane.
    n_scenes = max(3, min(8, round(target / 4.5)))

    # Video uses video_style when the brand defines one: the still style's
    # composition words ("abstract compute topology") fight the concrete
    # subject a short needs.
    style = (brand.visual.get("video_style")
             or brand.visual.get("style", "")).strip()
    rules_block = ""
    if brand.hard_rules:
        rules_block = "\n!! ABSOLUTE RULES:\n" + "\n".join(
            f"  - {r.strip()}" for r in brand.hard_rules
        )

    user = f"""BRAND: {brand.name} ({brand.url}) - {brand.tagline}
WHAT IT IS: {brand.about.strip()}
AUDIENCE: {brand.audience.strip()}

CLAIMS YOU MAY MAKE (invent no others):
{chr(10).join(f'  - {p}' for p in brand.proof_points)}
{rules_block}

HOUSE VISUAL STYLE - every visual direction must obey this:
{style}

PLATFORM: {ps.label}, vertical {ps.video_size[0]}x{ps.video_size[1]}, target {target}s
PILLAR: {pillar.label}
IDEA: {angle}

Write exactly {n_scenes} scenes totalling about {target} seconds."""

    with gpu(TEXT):
        data = chat_json(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            models=settings.writer_chain,
            temperature=0.9,
            max_tokens=1600,
        )

    scenes = []
    for s in data.get("scenes", []):
        cap = _sanitize(str(s.get("caption", "")))
        vis = str(s.get("visual", "")).strip()
        if not vis:
            continue
        try:
            secs = float(s.get("seconds", 4.5))
        except (TypeError, ValueError):
            secs = 4.5
        scenes.append(
            Scene(
                caption=cap,
                visual=f"{scrub_visual(vis)}. {style}",
                seconds=max(2.0, min(8.0, secs)),
                motion=str(s.get("motion") or "slow push in, subtle parallax"),
            )
        )

    if not scenes:
        raise LLMError("script had no usable scenes")

    script = Script(title=_sanitize(str(data.get("title", angle[:80]))), scenes=scenes)
    log.info(
        "script %r: %d scenes, %.1fs", script.title[:50], len(scenes), script.duration
    )
    return script


def caption_for(brand: Brand, ps: PlatformSpec, script: Script) -> str:
    """The post text that accompanies the video."""
    from ..llm.copywriter import _clean
    from ..llm.client import chat_with_fallback

    beats = " / ".join(s.caption for s in script.scenes if s.caption)
    tags = ps.hashtags[1]
    user = f"""Write the caption for a {ps.label} video.

VIDEO TITLE: {script.title}
WHAT THE VIDEO SHOWS: {beats}
BRAND: {brand.name} - {brand.tagline}

The caption must add context the video does not already state - not a summary
of it. Under {min(300, ps.max_chars)} characters. {tags} hashtags, lowercase.
{"Do not include a URL; it is not clickable here." if ps.link_policy != "ok" else ""}
Output only the caption."""

    with gpu(TEXT):
        raw, _ = chat_with_fallback(
            [{"role": "user", "content": user}],
            settings.writer_chain,
            temperature=0.85,
            max_tokens=400,
        )
    text = _clean(raw)

    # Same gate as a normal post - a video caption is still published copy.
    for attempt in range(2):
        if not rules.blocking(rules.check(text, brand, ps)):
            break
        problems = rules.summarize(rules.check(text, brand, ps))
        with gpu(TEXT):
            raw, _ = chat_with_fallback(
                [{"role": "user", "content": f"{user}\n\nPrevious attempt was "
                                             f"rejected: {problems}\nFix it."}],
                settings.writer_chain,
                temperature=0.7,
                max_tokens=400,
            )
        text = _clean(raw)
    return text
