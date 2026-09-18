"""Template-driven shorts.

A template is an HTML file that renders itself from `window.__DATA` and exposes
`window.__frame(t)`. Copy is generated per post, so nothing here hardcodes a
sentence; the template supplies layout, motion and brand, the data supplies
words.

Why templates rather than free-form scenes: a short that works is a structure -
hook, demonstration, proof, call to action - and the structure is what makes it
read as an ad rather than a slideshow with text on it. Handing a model a blank
canvas produced eight loosely related captions over drifting b-roll, which is
the thing this replaces.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..motion import MotionError, render

log = logging.getLogger("adforge.shorts")

HERE = Path(__file__).parent


@dataclass(frozen=True)
class Template:
    key: str
    label: str
    seconds: float
    # What the model must produce. Kept next to the template so the two cannot
    # drift apart unnoticed.
    fields: tuple[str, ...]
    brief: str

    @property
    def path(self) -> Path:
        return HERE / f"{self.key}.html"


TEMPLATES: dict[str, Template] = {
    "endpoint_swap": Template(
        key="endpoint_swap",
        label="One line changes (code swap)",
        seconds=15.0,
        fields=("hook_a", "hook_b", "hook_dim", "eyebrow", "old_line",
                "new_line", "swap_note", "output", "proof_a", "proof_b",
                "claims", "tagline"),
        brief=(
            "A code-level before/after. hook_* are three SHORT lines (max 22 "
            "chars each) naming the friction. old_line/new_line are the single "
            "line of code that changes. claims is 3-4 items, each at most 32 "
            "characters, and every one must come from the brand's proof points."
        ),
    ),
    "statement": Template(
        key="statement",
        label="Statement (no product UI)",
        seconds=13.0,
        fields=("hook_a", "hook_b", "hook_dim", "eyebrow", "points",
                "kicker_a", "kicker_b", "claims", "tagline"),
        brief=(
            "Typography only, for a claim with nothing to show on screen. "
            "points is 3 short lines carrying the argument. kicker_* is the "
            "payoff in two lines. claims is 3-4 items from the proof points."
        ),
    ),
}


def brand_theme(brand) -> dict:
    """Colours and marks per brand, so one template serves both companies."""
    themes = {
        "inferix": {"accent": "#22d3ee", "accent_deep": "#0e7490",
                    "wash_a": "#0891b2", "wash_b": "#155e75", "mark": "I"},
        "vallorix": {"accent": "#34d399", "accent_deep": "#047857",
                     "wash_a": "#059669", "wash_b": "#065f46", "mark": "V"},
    }
    t = themes.get(brand.key, themes["inferix"])
    return {**t, "name": brand.name, "url": brand.url.replace("https://", ""),
            "tagline": brand.tagline}


def render_short(
    brand, template_key: str, data: dict, out: Path,
    *, size: tuple[int, int] = (1080, 1920), fps: int = 30,
) -> Path:
    """Render one short. `data` fills the template; missing keys use its own."""
    tpl = TEMPLATES.get(template_key)
    if tpl is None:
        raise MotionError(f"no such short template: {template_key!r} "
                          f"(have {', '.join(sorted(TEMPLATES))})")
    if not tpl.path.exists():
        raise MotionError(f"template file missing: {tpl.path}")

    payload = {**data, "brand": brand_theme(brand)}
    html = tpl.path.read_text()
    # Injected immediately before the template's own script, so the template
    # can read it at parse time and the file still opens standalone in a
    # browser with its built-in defaults.
    inject = f"<script>window.__DATA = {json.dumps(payload)};</script>"
    if "<!--DATA-->" in html:
        html = html.replace("<!--DATA-->", inject)
    else:
        raise MotionError(f"{tpl.path.name} has no <!--DATA--> injection point")

    w, h = size
    log.info("short %s (%s) %dx%d %.1fs", out.name, template_key, w, h, tpl.seconds)
    return render(html, out, seconds=tpl.seconds, fps=fps, width=w, height=h)
