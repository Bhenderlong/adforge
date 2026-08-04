"""
Short-form video assembly.

Wan 2.2 produces roughly three seconds of real motion per clip, which is far
short of a 30-40s TikTok. So a short is built as a sequence of scenes:

    for each beat of the script:
        SDXL still  ->  Wan i2v clip  ->  slow zoom hold to fill the beat
    concatenate with cross-fades, burn in captions, add the outro card

Captions are burned in rather than left to the platform's auto-captions: most
short-form video is watched muted, and platform captions are unreliable on
technical vocabulary ("KV cache", "SOC 2") which is exactly the vocabulary
these videos depend on.
"""

from __future__ import annotations

import json
import logging
import random
import shlex
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from ..brands import Brand
from ..config import ASSETS, settings
from ..gpu import MEDIA, gpu
from . import comfy
from .images import generate_image

log = logging.getLogger("adforge.video")


class VideoError(RuntimeError):
    pass


@dataclass
class Scene:
    """One beat of a short."""

    caption: str  # burned-in text, kept short enough to read at a glance
    visual: str  # image prompt for this beat
    seconds: float = 4.0
    motion: str = "slow push in, subtle parallax, cinematic"
    still: Path | None = None
    clip: Path | None = None


@dataclass
class Script:
    title: str
    scenes: list[Scene] = field(default_factory=list)
    outro: str = ""

    @property
    def duration(self) -> float:
        return sum(s.seconds for s in self.scenes)


def _run(cmd: list[str], timeout: int = 900) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise VideoError(
            f"ffmpeg failed ({p.returncode}): {' '.join(shlex.quote(c) for c in cmd[:8])}"
            f"...\n{p.stderr[-1200:]}"
        )


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [settings.ffmpeg, "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    ).stderr
    # ffmpeg writes "time=00:00:03.06" on the last progress line.
    times = [t for t in out.split("time=") if ":" in t[:9]]
    if not times:
        return 0.0
    hh, mm, ss = times[-1][:11].split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def _escape_drawtext(s: str) -> str:
    """ffmpeg drawtext needs colons, quotes, backslashes and % escaped."""
    return (
        s.replace("\\", r"\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("%", r"\%")
        .replace(",", r"\,")
    )


def _caption_filter(text: str, w: int, h: int, wrap: int = 22) -> str:
    """Bottom-third caption block with a readable scrim behind it."""
    lines = textwrap.wrap(text.strip(), wrap)[:3]
    if not lines:
        return ""
    size = max(34, int(w * 0.062))
    line_h = int(size * 1.32)
    block_h = line_h * len(lines)
    top = int(h * 0.70) - block_h // 2

    parts = [
        f"drawbox=x=0:y={top - 28}:w={w}:h={block_h + 56}:"
        f"color=black@0.55:t=fill"
    ]
    for i, line in enumerate(lines):
        parts.append(
            f"drawtext=text='{_escape_drawtext(line)}':"
            f"fontcolor=white:fontsize={size}:"
            f"borderw={max(3, size // 14)}:bordercolor=black@0.9:"
            f"x=(w-text_w)/2:y={top + i * line_h}"
        )
    return ",".join(parts)


def _still_to_motion(still: Path, dst: Path, seconds: float, w: int, h: int,
                     zoom_to: float = 1.12) -> Path:
    """Ken Burns fallback when Wan is unavailable or a clip fails.

    zoompan operates per output frame, so the zoom rate is derived from the
    total frame count rather than hard-coded, otherwise clip length changes
    silently change the speed of the move.
    """
    fps = 30
    frames = max(1, int(seconds * fps))
    step = (zoom_to - 1.0) / frames
    vf = (
        f"scale={w * 2}:{h * 2},"
        f"zoompan=z='min(zoom+{step:.6f},{zoom_to})':d={frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps},"
        f"setsar=1"
    )
    _run([
        settings.ffmpeg, "-y", "-loglevel", "error", "-loop", "1",
        "-i", str(still), "-t", f"{seconds}", "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(fps), str(dst),
    ])
    return dst


def _extend_clip(clip: Path, dst: Path, seconds: float, w: int, h: int) -> Path:
    """Stretch a short Wan clip to fill its beat.

    The clip plays once, then the final frame is held with a slow push so the
    motion does not visibly loop. Looping a 3s clip four times reads as a GIF
    and is the fastest way to make a short look cheap.
    """
    have = probe_duration(clip)
    fps = 30
    if have <= 0:
        raise VideoError(f"could not read duration of {clip}")

    base = ASSETS / f"{dst.stem}_base.mp4"
    _run([
        settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(clip),
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,"
               f"crop={w}:{h},fps={fps},setsar=1",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", str(base),
    ])

    if have >= seconds - 0.15:
        _run([
            settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(base),
            "-t", f"{seconds}", "-c", "copy", str(dst),
        ])
        base.unlink(missing_ok=True)
        return dst

    tail = ASSETS / f"{dst.stem}_tail.png"
    _run([
        settings.ffmpeg, "-y", "-loglevel", "error", "-sseof", "-0.2",
        "-i", str(base), "-vframes", "1", str(tail),
    ])
    hold = ASSETS / f"{dst.stem}_hold.mp4"
    _still_to_motion(tail, hold, seconds - have, w, h, zoom_to=1.06)

    lst = ASSETS / f"{dst.stem}_concat.txt"
    lst.write_text(f"file '{base}'\nfile '{hold}'\n")
    _run([
        settings.ffmpeg, "-y", "-loglevel", "error", "-f", "concat",
        "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-pix_fmt", "yuv420p", str(dst),
    ])
    for f in (base, tail, hold, lst):
        f.unlink(missing_ok=True)
    return dst


def render_scene(
    brand: Brand, scene: Scene, slug: str, w: int, h: int, use_wan: bool = True
) -> Path:
    """Still -> motion clip -> captioned beat, at the target resolution."""
    from ..platforms.spec import PlatformSpec

    vertical = PlatformSpec(key="_v", label="_v", max_chars=0,
                            image_size=(_nearest_even(w), _nearest_even(h)))
    scene.still = generate_image(brand, vertical, scene.visual, f"{slug}_still")

    motion = ASSETS / f"{slug}_motion.mp4"
    made = False
    if use_wan and settings.enable_video:
        try:
            handle = comfy.upload_image(scene.still)
            graph = comfy.build_i2v(
                handle,
                f"{scene.motion}. {scene.visual}",
                "static, frozen, jitter, flicker, warped, distorted, text, watermark",
                seed=random.randint(1, 2**31 - 1),
                high_noise=settings.wan_high_noise,
                low_noise=settings.wan_low_noise,
                vae=settings.wan_vae,
                text_encoder=settings.wan_text_encoder,
                length=settings.wan_frames,
                width=settings.wan_width,
                height=settings.wan_height,
                fps=settings.wan_fps,
            )
            with gpu(MEDIA):
                files = comfy.run(graph)
            raw = ASSETS / f"{slug}_wan.webp"
            comfy.fetch(files[0], raw)
            _extend_clip(raw, motion, scene.seconds, w, h)
            raw.unlink(missing_ok=True)
            made = True
        except Exception as e:  # noqa: BLE001
            # A failed clip must not kill the whole video - fall back to the
            # Ken Burns move, which always works from the same still.
            log.warning("wan i2v failed for %s, using still motion: %s", slug, e)

    if not made:
        _still_to_motion(scene.still, motion, scene.seconds, w, h)

    if not scene.caption.strip():
        return motion

    out = ASSETS / f"{slug}_beat.mp4"
    _run([
        settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(motion),
        "-vf", _caption_filter(scene.caption, w, h),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", str(out),
    ])
    motion.unlink(missing_ok=True)
    scene.clip = out
    return out


def _nearest_even(n: int) -> int:
    return n - (n % 2)


def assemble(
    clips: list[Path], dst: Path, w: int, h: int, crossfade: float = 0.35
) -> Path:
    """Concatenate beats with cross-fades and a silent audio track.

    Every platform in scope re-encodes on upload and several treat a missing
    audio stream as a malformed file, so a silent track is always attached.
    """
    if not clips:
        raise VideoError("no clips to assemble")

    if len(clips) == 1:
        merged = clips[0]
    else:
        inputs: list[str] = []
        for c in clips:
            inputs += ["-i", str(c)]
        # xfade offsets are cumulative: each transition steals `crossfade`
        # seconds from the running total, so the offset must be tracked rather
        # than computed from clip index.
        filters, prev, offset = [], "[0:v]", 0.0
        for i in range(1, len(clips)):
            offset += probe_duration(clips[i - 1]) - crossfade
            label = f"[v{i}]"
            filters.append(
                f"{prev}[{i}:v]xfade=transition=fade:duration={crossfade}:"
                f"offset={max(0.1, offset):.3f}{label}"
            )
            prev = label
        merged = ASSETS / f"{dst.stem}_merged.mp4"
        _run([
            settings.ffmpeg, "-y", "-loglevel", "error", *inputs,
            "-filter_complex", ";".join(filters), "-map", prev,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", str(merged),
        ], timeout=1800)

    _run([
        settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(merged),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst),
    ], timeout=1800)
    if merged != clips[0]:
        merged.unlink(missing_ok=True)
    return dst


def render_script(
    brand: Brand, script: Script, slug: str, size: tuple[int, int],
    use_wan: bool = True,
) -> Path:
    w, h = size
    clips = []
    for i, scene in enumerate(script.scenes):
        log.info("scene %d/%d: %s", i + 1, len(script.scenes), scene.caption[:60])
        clips.append(render_scene(brand, scene, f"{slug}_s{i}", w, h, use_wan))
    out = assemble(clips, ASSETS / f"{slug}.mp4", w, h)
    for c in clips:
        c.unlink(missing_ok=True)
    log.info("video %s -> %s (%.1fs)", slug, out.name, probe_duration(out))
    return out
