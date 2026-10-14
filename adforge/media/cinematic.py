"""Cinematic ads: AI footage, cut short, graded, with typography over it.

The format matters more than the model. A real tech commercial is a handful of
short graded shots with text on top - not one long generated clip, and not
typography on a flat background. Both halves already work here; this joins
them.

Three decisions carry most of the quality:

1. SHOT LENGTH. Wan holds together for about two seconds and then begins to
   morph - fingers merge, lettering crawls, straight edges bend. Cutting at
   2.2s is not a workaround, it is how ads are edited anyway, and it removes
   the single most recognisable AI tell.

2. SUBJECT CHOICE. The model breaks on hands, faces and text. It is good at
   hardware, architecture, light, liquid and abstraction. The prompt builder
   below steers towards the second set and names the first set in the negative
   prompt, every time.

3. GRADE. An ungraded render reads as a render. A film curve, a little
   saturation, a vignette and real grain do more for perceived production
   value than any number of sampling steps.
"""

from __future__ import annotations

import logging
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ASSETS, settings
from . import comfy
from ..gpu import MEDIA, gpu

log = logging.getLogger("adforge.cinematic")


class CinematicError(RuntimeError):
    pass


# Camera and lighting language the plate model responds to. Chosen from the
# vocabulary of commercial cinematography rather than of stock photography.
LOOKS = [
    "anamorphic 35mm, shallow depth of field, cyan practical lighting, "
    "volumetric haze, high contrast, deep shadows",
    "macro detail shot, razor-thin focus plane, cold rim light against warm "
    "bounce, matte black surfaces, subtle specular highlights",
    "wide establishing shot, long lens compression, rows receding into "
    "darkness, pools of light, atmospheric fog",
    "overhead top-down composition, geometric symmetry, single hard key light, "
    "long shadows across a dark floor",
]

MOVES = [
    "slow dolly push in, steady, almost imperceptible",
    "gentle lateral tracking move, parallax between foreground and background",
    "slow vertical crane down, revealing depth",
    "static locked-off frame, only light and particles moving",
]

# What Wan cannot do convincingly. Named explicitly on every render, because
# one bad hand in one frame is what makes a viewer say "that is AI".
NEGATIVE = (
    "hands, fingers, people, faces, human figures, crowds, text, letters, "
    "words, lettering, typography, brand names, product names, labels, "
    "signage, logos, numbers, ui, watermark, subtitles, "
    "morphing, warping, melting, flickering, jitter, duplicated objects, "
    "extra limbs, distorted geometry, cartoon, illustration, cgi render, "
    "lowres, oversaturated, blown highlights"
)


@dataclass
class Shot:
    """One cut of the ad."""

    subject: str          # what is in frame, in plain words
    seconds: float = 2.2  # cut before the model starts to morph
    look: str = ""        # overrides the rotating LOOKS entry
    move: str = ""        # overrides the rotating MOVES entry
    plate: Path | None = None
    clip: Path | None = None


@dataclass
class Ad:
    slug: str
    shots: list[Shot] = field(default_factory=list)
    width: int = 1080
    height: int = 1920
    fps: int = 24


def plate_prompt(shot: Shot, i: int) -> str:
    """The still that the motion is generated from.

    Quality of the finished cut is mostly decided here: i2v inherits the
    composition, lighting and grain of its first frame, so a flat or generic
    plate cannot be rescued by the video model.
    """
    look = shot.look or LOOKS[i % len(LOOKS)]
    # No brand or product names anywhere in here. "shot on ARRI Alexa" put
    # the word ARRI onto the side of a graphics card in the first scheduled
    # ad - the plate model renders names it is given as lettering in the
    # image, and lettering is exactly what it cannot spell.
    return (f"cinematic commercial still, {shot.subject}, {look}, "
            f"large format digital cinema camera, photoreal, filmic colour "
            f"grade, fine film grain, professional advertising photography, "
            f"unbranded hardware, blank surfaces, no people")


def render_shot(shot: Shot, i: int, slug: str, w: int, h: int,
                fps: int) -> Path:
    """Plate, then motion, then trimmed to the cut length."""
    seed = random.randint(1, 2**31 - 1)
    plate = ASSETS / f"{slug}_s{i}_plate.png"

    graph = comfy.build_txt2img(
        settings.comfy_checkpoint, plate_prompt(shot, i), NEGATIVE,
        width=w, height=h, steps=30, cfg=5.0, seed=seed)
    with gpu(MEDIA):
        files = comfy.run(graph)
    comfy.fetch(files[0], plate)
    shot.plate = plate

    # 4n+1 frames, a little longer than the cut so the trim can drop the tail
    # where morphing starts.
    frames = int(shot.seconds * fps) + 9
    frames -= (frames - 1) % 4
    move = shot.move or MOVES[i % len(MOVES)]

    handle = comfy.upload_image(plate)
    vg = comfy.build_i2v(
        handle, f"{move}, {shot.subject}", NEGATIVE, seed=seed,
        high_noise=settings.wan_high_noise, low_noise=settings.wan_low_noise,
        vae=settings.wan_vae, text_encoder=settings.wan_text_encoder,
        lora_high=settings.wan_lora_high, lora_low=settings.wan_lora_low,
        lora_strength=settings.wan_lora_strength,
        steps=settings.wan_steps, cfg=settings.wan_cfg,
        length=frames, width=w, height=h, fps=fps)
    with gpu(MEDIA):
        files = comfy.run(vg)
    raw = ASSETS / f"{slug}_s{i}_raw.webm"
    comfy.fetch(files[0], raw)
    shot.clip = raw
    log.info("shot %d: %d frames -> cut to %.1fs", i, frames, shot.seconds)
    return raw


# A film look, applied once to the assembled cut rather than per shot so the
# whole ad shares one grade - mismatched grades between cuts is the thing that
# reads as "assembled from clips".
GRADE = (
    "curves=all='0/0.02 0.25/0.21 0.5/0.52 0.75/0.80 1/0.98',"   # filmic S-curve
    "eq=saturation=1.12:contrast=1.06,"
    "unsharp=5:5:0.5:5:5:0.0,"                                    # gentle crispen
    "vignette=PI/5,"
    "noise=alls=5:allf=t+u"                                       # real grain
)


def assemble(ad: Ad, overlay_dir: Path | None, out: Path) -> Path:
    """Trim, concat, grade, then composite the typography over the top."""
    if not ad.shots or not all(s.clip for s in ad.shots):
        raise CinematicError("every shot must be rendered before assembly")

    parts = []
    for i, s in enumerate(ad.shots):
        p = ASSETS / f"{ad.slug}_s{i}_cut.mp4"
        subprocess.run([
            settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(s.clip),
            "-t", f"{s.seconds:.2f}",
            "-vf", f"scale={ad.width}:{ad.height}:force_original_aspect_ratio=increase,"
                   f"crop={ad.width}:{ad.height},fps={ad.fps}",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(p),
        ], capture_output=True, check=False)
        parts.append(p)

    lst = ASSETS / f"{ad.slug}_concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    joined = ASSETS / f"{ad.slug}_joined.mp4"
    subprocess.run([settings.ffmpeg, "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)],
                   capture_output=True, check=False)

    if overlay_dir is not None:
        cmd = [settings.ffmpeg, "-y", "-loglevel", "error",
               "-i", str(joined),
               "-framerate", str(ad.fps), "-i", str(overlay_dir / "%05d.png"),
               "-filter_complex",
               f"[0:v]{GRADE}[g];[g][1:v]overlay=0:0:format=auto[v]",
               "-map", "[v]", "-c:v", "libx264", "-crf", "17",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    else:
        cmd = [settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(joined),
               "-vf", GRADE, "-c:v", "libx264", "-crf", "17",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise CinematicError(f"assembly failed: {(r.stderr or '')[:400]}")

    for p in parts + [joined, lst]:
        p.unlink(missing_ok=True)
    log.info("ad %s -> %s (%.1f MB)", ad.slug, out.name, out.stat().st_size / 1e6)
    return out
