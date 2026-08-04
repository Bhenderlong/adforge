"""Brand profile loading. One YAML per brand in brands/."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .config import BRANDS_DIR


@dataclass
class Pillar:
    key: str
    label: str
    description: str
    weight: int = 1


@dataclass
class Brand:
    key: str
    name: str
    url: str
    tagline: str
    about: str
    audience: str
    proof_points: list[str]
    pillars: list[Pillar]
    keywords: list[str]
    cta_variants: list[str]
    visual: dict
    voice: dict
    hard_rules: list[str] = field(default_factory=list)

    def pick_pillar(self, rng: random.Random | None = None) -> Pillar:
        rng = rng or random
        return rng.choices(self.pillars, weights=[p.weight for p in self.pillars])[0]

    def pillar(self, key: str) -> Pillar:
        for p in self.pillars:
            if p.key == key:
                return p
        raise ValueError(f"{self.key}: no pillar {key!r}")


_STR_LISTS = ("proof_points", "keywords", "cta_variants", "hard_rules")


def _load(path: Path) -> Brand:
    raw = yaml.safe_load(path.read_text())

    # An unquoted colon inside a YAML list item silently becomes a dict
    # ("- Docs and free tier: inferix.co" -> {"Docs and free tier": ...}).
    # That reaches the prompt as a stringified dict and corrupts the copy, so
    # fail loudly here instead - the UI surfaces this when a brand is edited.
    for key in _STR_LISTS:
        for i, item in enumerate(raw.get(key) or []):
            if not isinstance(item, str):
                raise ValueError(
                    f"{path.name}: {key}[{i}] parsed as {type(item).__name__}, not a "
                    f"string ({item!r}). A colon in a YAML list item needs the whole "
                    f'item quoted: - "text: more text"'
                )

    raw["pillars"] = [Pillar(**p) for p in raw.get("pillars", [])]
    return Brand(**raw)


@lru_cache(maxsize=None)
def all_brands() -> dict[str, Brand]:
    return {p.stem: _load(p) for p in sorted(BRANDS_DIR.glob("*.yaml"))}


def get_brand(key: str) -> Brand:
    brands = all_brands()
    try:
        return brands[key]
    except KeyError:
        raise ValueError(f"unknown brand {key!r}; known: {sorted(brands)}") from None


def reload_brands() -> None:
    """Called after the UI edits a brand YAML."""
    all_brands.cache_clear()
