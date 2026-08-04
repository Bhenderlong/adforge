"""
AdForge configuration.

Everything is env-overridable so the same tree runs on the workstation and on
the Dell without edits. Secrets live in .env (gitignored); nothing here has a
usable default credential.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BRANDS_DIR = ROOT / "brands"
ASSETS = DATA / "assets"


def _find_ffmpeg() -> str:
    """ffmpeg is not installed system-wide on this box, but several pixi envs
    ship one and imageio-ffmpeg bundles a static build. Prefer PATH, then the
    pip-installed static binary, then anything already on disk."""
    if p := shutil.which("ffmpeg"):
        return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for cand in (
        Path.home() / ".ce/.pixi/envs/hunyuan3dpart-nodes/bin/ffmpeg",
        Path.home() / ".ce/envs/geometrypack-nodes/.pixi/envs/default/bin/ffmpeg",
    ):
        if cand.exists():
            return str(cand)
    return "ffmpeg"  # will fail loudly at call time rather than silently here


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_prefix="ADFORGE_", extra="ignore"
    )

    # ---- inference backend -------------------------------------------------
    # Both Ollama and vLLM expose an OpenAI-compatible /v1; we speak that dialect
    # to either. llm_backend only changes the default base_url and whether we
    # send Ollama-specific keep_alive hints.
    llm_backend: Literal["ollama", "vllm"] = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    vllm_url: str = "http://127.0.0.1:8000"

    # Big model writes the copy, small fast model does classification/scoring.
    # These must already be pulled; we never auto-download multi-GB weights.
    #
    # Avoid *reasoning* models here (qwen3.x, glm-4.x, deepseek-r1). They spend
    # the token budget on a chain of thought and return an empty `content`,
    # which surfaces as blank posts. Instruct-tuned models only.
    model_writer: str = "llama3.3:70b"
    model_critic: str = "llama3.3:70b"
    model_fast: str = "qwen2.5:14b-instruct"

    # Ordered fallbacks used when the primary will not fit in VRAM. A 70B needs
    # both 5090s, so if anything else has taken memory we still want output
    # rather than a failed scheduled run.
    model_fallbacks: str = "qwen2.5:14b-instruct,hermes3:latest"

    llm_timeout: int = 600

    @property
    def writer_chain(self) -> list[str]:
        return self._chain(self.model_writer)

    @property
    def critic_chain(self) -> list[str]:
        return self._chain(self.model_critic)

    @property
    def fast_chain(self) -> list[str]:
        return self._chain(self.model_fast)

    def _chain(self, primary: str) -> list[str]:
        seen, out = set(), []
        for m in [primary, *self.model_fallbacks.split(",")]:
            m = m.strip()
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    # ---- media -------------------------------------------------------------
    comfy_url: str = "http://127.0.0.1:8189"
    comfy_checkpoint: str = "RealVisXL_V5.0_fp16.safetensors"
    comfy_timeout: int = 1800
    enable_video: bool = True
    ffmpeg_bin: str = ""

    # Wan 2.2 image-to-video, two-expert split. Names must match exactly what
    # the engine reports for UNETLoader/VAELoader/CLIPLoader.
    wan_high_noise: str = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    wan_low_noise: str = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    # MUST be the 2.1 VAE despite the 2.2 model names. wan2.2_vae belongs to
    # the TI2V-5B variant and has a different latent channel count; pairing it
    # with the 14B i2v experts fails inside KSamplerAdvanced with
    # "expected input[...] to have 36 channels, but got 64 channels instead".
    wan_vae: str = "wan_2.1_vae.safetensors"
    wan_text_encoder: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    # Frames per generated clip. Wan wants 4n+1; 49 @ 16fps is ~3s of motion,
    # which is then held/cross-faded to fill a 30-40s short.
    wan_frames: int = 49
    wan_fps: int = 16
    # Native generation resolution for vertical clips. Upscaled to 1080x1920 by
    # ffmpeg - generating at full vertical HD is far slower for no visible gain
    # after the platform re-encodes it anyway.
    wan_width: int = 480
    wan_height: int = 832

    # ---- publishing safety -------------------------------------------------
    # Master switch. While true NOTHING is transmitted to any platform; adapters
    # log the exact payload they would have sent. Flipped per-account in the UI.
    dry_run: bool = True
    # Minutes a generated post waits in REVIEW before auto-publishing. 0 = fully
    # hands-off. Only applies to auto-post accounts.
    review_window_minutes: int = 60
    # Hard ceiling per brand per platform per day, independent of schedule.
    daily_post_cap: int = 6

    # Roughly how long one post takes end to end: angle, up to four write
    # attempts, critic and factcheck on the writer model, plus an image. Used
    # only to warn when a schedule asks for more than the box can produce in a
    # day - measured at ~12 minutes for a 70B on two 5090s, so the default is
    # deliberately not optimistic. Video posts cost several times this.
    minutes_per_post: int = 12

    # ---- radar -------------------------------------------------------------
    radar_enabled: bool = True
    radar_interval_minutes: int = 30
    # Radar replies ALWAYS carry a disclosure line and ALWAYS require approval.
    # These are not configurable by design - see docs/POLICY.md.
    radar_max_replies_per_day: int = 8

    # ---- server ------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8770
    # The UI process runs the scheduler by default. Set false when running a
    # separate `run.py --worker`, otherwise both processes tick and both try to
    # publish the same posts (the claim stops a duplicate send, but the second
    # scheduler is still wasted work and confusing in the logs).
    ui_runs_scheduler: bool = True
    # Extra hostnames the UI will accept state-changing requests on, comma
    # separated. Only needed behind a reverse proxy; localhost is always
    # allowed. Anything not listed is refused, which is what closes DNS
    # rebinding.
    extra_hosts: str = ""
    db_url: str = f"sqlite:///{DATA / 'adforge.db'}"
    timezone: str = "America/New_York"

    @property
    def llm_base(self) -> str:
        base = self.ollama_url if self.llm_backend == "ollama" else self.vllm_url
        return base.rstrip("/") + "/v1"

    @property
    def ffmpeg(self) -> str:
        return self.ffmpeg_bin or _find_ffmpeg()


settings = Settings()

# Owner-only. `data/` holds the credential database and the logs, and the
# default umask on a normal account (022, or 002 here) would otherwise leave
# both group- and world-readable.
for _d in (DATA, ASSETS, DATA / "logs"):
    _d.mkdir(parents=True, exist_ok=True)
    try:
        _d.chmod(0o700)
    except OSError:  # noqa: S110 - a read-only mount is not worth dying over
        pass
