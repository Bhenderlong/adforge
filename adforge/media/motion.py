"""Motion-graphics renderer: an HTML document becomes a video, frame by frame.

Why this exists rather than more image-to-video generation: professional ads in
this market are not photoreal footage. They are typography, product UI and
clean geometric motion on a brand palette - Stripe, Linear, Vercel, Modal all
look like that, and an audience of ML engineers identifies generated b-roll
instantly. Wan renders drifting photoreal imagery, which is the wrong medium no
matter how many sampling steps it gets.

The contract with a template is deliberately narrow:

    window.__frame(t)   // t in seconds, called before every screenshot

The document must render the exact state of the animation at t and nothing
else. No CSS transitions, no requestAnimationFrame, no Date.now(). A frame
must depend only on t, so the same second always produces the same pixels -
which is what makes a render reproducible and lets a single frame be inspected
in isolation instead of scrubbing a video.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from ..config import ASSETS, ROOT, settings

log = logging.getLogger("adforge.motion")


class MotionError(RuntimeError):
    pass


def render(
    html: str,
    out: Path,
    *,
    seconds: float,
    fps: int = 30,
    width: int = 1080,
    height: int = 1920,
    scale: int = 1,
) -> Path:
    """Render `html` to an h264 mp4. Returns the output path."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover - environment problem, not logic
        raise MotionError(
            "playwright is not installed; motion rendering is unavailable"
        ) from None

    frames = int(round(seconds * fps))
    if frames < 2:
        raise MotionError(f"{seconds}s at {fps}fps is {frames} frame(s)")

    tmp = ASSETS / f".{out.stem}_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    for stale in tmp.glob("*.png"):
        stale.unlink()

    with sync_playwright() as p:
        # channel="chrome" uses the Chrome already on this machine. The
        # alternative is a 500MB per-browser download into ~/.cache, on a disk
        # that has been at 97% all week.
        browser = p.chromium.launch(
            channel="chrome",
            args=["--force-device-scale-factor=1", "--hide-scrollbars",
                  "--disable-lcd-text", "--font-render-hinting=none"],
        )
        page = browser.new_page(
            viewport={"width": width // scale, "height": height // scale},
            device_scale_factor=scale,
        )
        page.set_content(html, wait_until="load")
        # Fonts must be resident before the first screenshot or frame 0 renders
        # in a fallback face and the whole clip flickers on the second frame.
        page.evaluate("document.fonts.ready")
        if not page.evaluate("typeof window.__frame === 'function'"):
            browser.close()
            raise MotionError("template does not define window.__frame(t)")

        for i in range(frames):
            page.evaluate("t => window.__frame(t)", i / fps)
            page.screenshot(path=str(tmp / f"{i:05d}.png"), animations="disabled")
        browser.close()

    got = len(list(tmp.glob("*.png")))
    if got != frames:
        raise MotionError(f"captured {got} frames, expected {frames}")

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        settings.ffmpeg, "-y", "-loglevel", "error",
        "-framerate", str(fps), "-i", str(tmp / "%05d.png"),
        # yuv420p and even dimensions: without both, the file plays here and is
        # rejected or shows as a black rectangle on iOS and in some browsers.
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-preset", "medium", "-movflags", "+faststart",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise MotionError(f"ffmpeg failed: {(r.stderr or '')[:400]}")

    _add_audio(out, seconds)

    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    log.info("motion render %s: %d frames @%dfps -> %.1f MB",
             out.name, frames, fps, out.stat().st_size / 1e6)
    return out


MUSIC_SUFFIXES = (".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg")


def pick_track(seed: str) -> Path | None:
    """A track from settings.music_dir, chosen deterministically by `seed`.

    Deterministic so a re-render of the same post keeps the same bed - a short
    that changes music every time it is regenerated is impossible to review.
    """
    d = Path(settings.music_dir)
    if not d.is_absolute():
        d = ROOT / d
    if not d.is_dir():
        return None
    tracks = sorted(p for p in d.iterdir()
                    if p.suffix.lower() in MUSIC_SUFFIXES and p.stat().st_size > 4096)
    if not tracks:
        return None
    n = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    return tracks[n % len(tracks)]


def _add_audio(video: Path, seconds: float) -> None:
    """Give the file an audio stream: a music bed if available, else silence.

    A video with NO audio stream at all is not merely quiet. Some platform
    encoders and players treat it as malformed, and TikTok in particular has
    rejected uploads over it, so silence is written explicitly rather than
    left absent.
    """
    track = pick_track(video.stem)
    tmp = video.with_suffix(".withaudio.mp4")
    fade = max(0.1, min(settings.music_fade_seconds, seconds / 3))

    if track:
        filt = (f"volume={settings.music_gain_db}dB,"
                f"afade=t=in:st=0:d={fade:.2f},"
                f"afade=t=out:st={seconds - fade:.2f}:d={fade:.2f}")
        cmd = [settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
               # -stream_loop so a track shorter than the short still fills it.
               "-stream_loop", "-1", "-i", str(track),
               "-filter_complex", f"[1:a]{filt}[a]",
               "-map", "0:v", "-map", "[a]", "-t", f"{seconds:.3f}",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart", str(tmp)]
    else:
        cmd = [settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
               "-f", "lavfi", "-i",
               "anullsrc=channel_layout=stereo:sample_rate=48000",
               "-map", "0:v", "-map", "1:a", "-t", f"{seconds:.3f}",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
               "-movflags", "+faststart", str(tmp)]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        # Not fatal: a short with no audio stream is still a usable short, and
        # failing the whole render over the bed would be the wrong trade.
        log.warning("could not attach audio to %s: %s",
                    video.name, (r.stderr or "")[:200])
        tmp.unlink(missing_ok=True)
        return
    tmp.replace(video)
    log.info("audio: %s", track.name if track else "silent stream (no tracks in "
             f"{settings.music_dir})")
