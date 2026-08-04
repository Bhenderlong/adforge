"""Per-post image generation, sized for each platform."""

from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

from ..brands import Brand
from ..config import ASSETS, settings
from ..gpu import MEDIA, gpu
from ..platforms.spec import PlatformSpec
from . import comfy

log = logging.getLogger("adforge.images")

# SDXL is trained at ~1MP. Generating directly at 1080x1350 or 1600x900 gives
# duplicated limbs and warped geometry, so we generate at the nearest supported
# bucket with the right aspect and let ffmpeg do the final resize.
SDXL_BUCKETS = [
    (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
    (1344, 768), (768, 1344), (1536, 640), (640, 1536),
]


def _bucket_for(w: int, h: int) -> tuple[int, int]:
    target = w / h
    return min(SDXL_BUCKETS, key=lambda b: abs(b[0] / b[1] - target))


def _resize(src: Path, dst: Path, w: int, h: int) -> Path:
    """Cover-crop to the exact platform dimensions."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    return dst


def generate_image(
    brand: Brand,
    ps: PlatformSpec,
    prompt: str,
    slug: str,
    seed: int | None = None,
    checkpoint: str | None = None,
) -> Path:
    """Render one post-specific image at the platform's exact dimensions."""
    seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    w, h = ps.image_size
    gw, gh = _bucket_for(w, h)

    negative = brand.visual.get("negative", "").strip() or "low quality, blurry"
    graph = comfy.build_txt2img(
        checkpoint or settings.comfy_checkpoint,
        prompt,
        negative,
        seed=seed,
        width=gw,
        height=gh,
    )

    with gpu(MEDIA):
        files = comfy.run(graph)

    raw = ASSETS / f"{slug}_raw.png"
    comfy.fetch(files[0], raw)
    out = _resize(raw, ASSETS / f"{slug}.png", w, h)
    raw.unlink(missing_ok=True)
    log.info("image %s -> %s (%dx%d, seed %d)", slug, out.name, w, h, seed)
    return out
