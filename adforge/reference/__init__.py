"""
Framework control reference data.

The writer model is fluent about compliance but unreliable on which control
number covers what, and it emits ISO 27001:2013 numbering that the 2022
revision retired. Both failure modes are invisible to a general critic and
obvious to the audience Vallorix sells to, so control IDs are grounded against
real data instead of trusted.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path

import yaml

REF_DIR = Path(__file__).resolve().parent

# SOC 2:  CC6.1, A1.2, PI1.3, P6.4
# ISO:    A.8.24, A5.7 (dot after A optional, as people write it both ways)
# HIPAA:  164.312(a)(1)
CONTROL_RE = re.compile(
    r"\b("
    r"CC\d{1,2}\.\d{1,2}"
    r"|A1\.\d{1,2}|C1\.\d{1,2}|PI1\.\d{1,2}|P\d\.\d{1,2}"
    # Third segment matched deliberately: ISO 27001:2013 used A.9.2.3 style,
    # and capturing it in full lets the block message name the real problem
    # (retired numbering) instead of reporting a mangled two-segment ID.
    r"|A\.?\d{1,2}\.\d{1,2}(?:\.\d{1,2})?"
    r"|164\.\d{3}\([a-z]\)(?:\(\d+\))?"
    r")\b"
)


@lru_cache(maxsize=None)
def _load(name: str) -> dict[str, str]:
    data = yaml.safe_load((REF_DIR / name).read_text()) or {}
    flat: dict[str, str] = {}
    for group in data.values():
        if isinstance(group, dict):
            flat.update({str(k): str(v) for k, v in group.items()})
    return flat


@lru_cache(maxsize=None)
def controls() -> dict[str, str]:
    """Every known control ID -> its description, across all frameworks."""
    merged: dict[str, str] = {}
    merged.update(_load("soc2.yaml"))
    merged.update(_load("iso27001.yaml"))
    return merged


def _normalise(cid: str) -> str:
    """ISO controls get written both 'A.8.24' and 'A8.24'."""
    c = cid.strip().upper()
    if re.fullmatch(r"A\d{1,2}\.\d{1,2}", c):
        c = "A." + c[1:]
    return c


def lookup(cid: str) -> str | None:
    return controls().get(_normalise(cid))


def cited(text: str) -> list[str]:
    """Control IDs referenced in a piece of copy, de-duplicated in order."""
    out, seen = [], set()
    for m in CONTROL_RE.finditer(text):
        c = _normalise(m.group(1))
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def unknown(text: str) -> list[str]:
    """Cited IDs that do not exist in any framework we know.

    HIPAA CFR citations are matched by the regex but not carried in a data
    file, so they are treated as known rather than blocked.
    """
    return [
        c
        for c in cited(text)
        if lookup(c) is None and not c.startswith("164.")
    ]


def sample(framework: str = "any", rng: random.Random | None = None) -> tuple[str, str]:
    """A real control to ground a post on, so the model does not invent one."""
    rng = rng or random
    pool = {
        "soc2": _load("soc2.yaml"),
        "iso27001": _load("iso27001.yaml"),
    }.get(framework, controls())
    cid = rng.choice(sorted(pool))
    return cid, pool[cid]


def grounding_block(text: str = "", framework: str = "any",
                    rng: random.Random | None = None) -> str:
    """Prompt fragment carrying the true text of the relevant controls."""
    ids = [c for c in cited(text) if lookup(c)]
    if not ids:
        cid, desc = sample(framework, rng)
        ids, pairs = [cid], [(cid, desc)]
    else:
        pairs = [(c, lookup(c)) for c in ids]
    lines = "\n".join(f"  {c}: {d}" for c, d in pairs)
    return (
        "AUTHORITATIVE CONTROL TEXT - if you reference a control, use these "
        "exact meanings. Do not cite any other control number, and do not use "
        "ISO 27001:2013 numbering (the 2022 revision renumbered every "
        f"control):\n{lines}"
    )
