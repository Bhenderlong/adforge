"""
ComfyUI client and workflow builders.

Talks to the AdStudio engine already running on :8189 rather than standing up a
second one - the box has ~86GB free and the Wan 2.2 + SDXL model set is ~41GB,
so a duplicate would not fit.

The graph builders are adapted from the AdStudio client, which is proven
against this exact engine and model set.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import httpx

from ..config import settings

log = logging.getLogger("adforge.comfy")


class ComfyError(RuntimeError):
    pass


def _url(path: str) -> str:
    return f"{settings.comfy_url.rstrip('/')}{path}"


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        _url(path),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(path: str) -> dict:
    with urllib.request.urlopen(_url(path), timeout=60) as r:
        return json.load(r)


def is_up() -> bool:
    try:
        _get("/system_stats")
        return True
    except Exception:  # noqa: BLE001
        return False


def list_options(node_class: str, field: str) -> list[str]:
    """Exact choices the engine accepts for a loader field."""
    try:
        info = _get(f"/object_info/{node_class}")
        opts = info[node_class]["input"]["required"][field][0]
        return list(opts) if isinstance(opts, list) else []
    except Exception as e:  # noqa: BLE001
        log.warning("could not list %s.%s: %s", node_class, field, e)
        return []


# The shared model directory also contains motion modules, detection models and
# LoRAs that are not loadable checkpoints, plus adult-content merges that must
# never be selected for brand imagery. Filter the dropdown the UI shows.
_NOT_A_CHECKPOINT = (
    "animatediff", "admotion", "sdpose", "sam3", "detr", "lora",
    "inpaint-fill", "motion",
)
_BRAND_UNSAFE = ("nsfw", "pony-realism", "hentai", "porn")


def checkpoints(safe_only: bool = True) -> list[str]:
    out = []
    for c in list_options("CheckpointLoaderSimple", "ckpt_name"):
        low = c.lower()
        if any(b in low for b in _NOT_A_CHECKPOINT):
            continue
        if safe_only and any(b in low for b in _BRAND_UNSAFE):
            continue
        out.append(c)
    return out


def unet_models() -> list[str]:
    return list_options("UNETLoader", "unet_name")


def run(graph: dict, timeout: int | None = None) -> list[dict]:
    """Submit a graph, wait for completion, return output file descriptors."""
    timeout = timeout or settings.comfy_timeout
    resp = _post("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    if "prompt_id" not in resp:
        raise ComfyError(f"engine rejected the graph: {str(resp)[:400]}")
    pid = resp["prompt_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = _get(f"/history/{pid}")
        if pid in hist:
            status = hist[pid].get("status", {})
            if status.get("status_str") == "error":
                # Surface the real node error rather than a generic failure -
                # a silent "execution error" here is very hard to debug later.
                for m in status.get("messages", []):
                    if m[0] == "execution_error":
                        info = m[1]
                        msg = (info.get("exception_message") or "").splitlines()
                        raise ComfyError(
                            f"{info.get('node_type')}: {msg[0] if msg else 'unknown'}"
                        )
                raise ComfyError("ComfyUI reported an execution error")
            files = []
            for node in hist[pid]["outputs"].values():
                for key in ("images", "gifs", "videos"):
                    files.extend(node.get(key, []))
            if not files:
                raise ComfyError("graph completed but produced no output files")
            return files
        time.sleep(1.5)
    raise ComfyError(f"job {pid} timed out after {timeout}s")


def fetch(fileinfo: dict, dest: Path) -> Path:
    q = urllib.parse.urlencode(
        {
            "filename": fileinfo["filename"],
            "subfolder": fileinfo.get("subfolder", ""),
            "type": fileinfo.get("type", "output"),
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_url(f"/view?{q}"), timeout=300) as r:
        dest.write_bytes(r.read())
    return dest


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_txt2img(
    checkpoint: str,
    prompt: str,
    negative: str,
    *,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    cfg: float = 4.5,
    steps: int = 30,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["3", 0],
                         "negative": ["4", 0], "latent_image": ["2", 0],
                         "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler,
                         "denoise": 1.0}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "adforge"}},
    }


def upload_image(path: Path, subfolder: str = "adforge") -> str:
    """Upload a local image into the engine's input dir; returns its handle.

    Uses httpx rather than a hand-built multipart body - getting the boundary
    and CRLF framing exactly right by hand is easy to get subtly wrong, and
    ComfyUI answers a malformed body with a bare 500 that says nothing about
    what was wrong with it.
    """
    with httpx.Client(timeout=180) as client:
        r = client.post(
            _url("/upload/image"),
            files={"image": (path.name, path.read_bytes(), "image/png")},
            data={"subfolder": subfolder, "overwrite": "true"},
        )
    if r.status_code >= 400:
        raise ComfyError(f"image upload failed [{r.status_code}]: {r.text[:300]}")
    out = r.json()
    sub = out.get("subfolder", "")
    return f"{sub}/{out['name']}" if sub else out["name"]


def build_i2v(
    image_handle: str,
    prompt: str,
    negative: str,
    *,
    seed: int,
    high_noise: str,
    low_noise: str,
    vae: str,
    text_encoder: str,
    cfg: float = 4.0,
    steps: int = 20,
    sampler: str = "euler",
    scheduler: str = "simple",
    length: int = 49,
    width: int = 480,
    height: int = 832,
    fps: int = 16,
) -> dict:
    """Wan 2.2 image-to-video, two-expert (high/low noise) split sampling.

    The high-noise expert runs the first half of the steps and passes leftover
    noise to the low-noise expert - that hand-off is what the two-model release
    is designed around, and collapsing it to one sampler degrades motion badly.
    """
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": high_noise, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "UNETLoader",
              "inputs": {"unet_name": low_noise, "weight_dtype": "fp8_e4m3fn"}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": text_encoder, "type": "wan"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_handle}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["3", 0]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["3", 0]}},
        "8": {"class_type": "WanImageToVideo",
              "inputs": {"positive": ["6", 0], "negative": ["7", 0],
                         "vae": ["4", 0], "start_image": ["5", 0],
                         "width": width, "height": height,
                         "length": length, "batch_size": 1}},
        "9": {"class_type": "KSamplerAdvanced",
              "inputs": {"model": ["1", 0], "add_noise": "enable",
                         "noise_seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler,
                         "start_at_step": 0, "end_at_step": steps // 2,
                         "return_with_leftover_noise": "enable",
                         "positive": ["8", 0], "negative": ["8", 1],
                         "latent_image": ["8", 2]}},
        "10": {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["2", 0], "add_noise": "disable",
                          "noise_seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": sampler, "scheduler": scheduler,
                          "start_at_step": steps // 2, "end_at_step": 10000,
                          "return_with_leftover_noise": "disable",
                          "positive": ["8", 0], "negative": ["8", 1],
                          "latent_image": ["9", 0]}},
        "11": {"class_type": "VAEDecode",
               "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        # SaveWEBM (VP9), not SaveAnimatedWEBP. Animated WebP needs libwebp's
        # animated demuxer, which the static imageio-ffmpeg build lacks - it
        # rejects the file with "Decode error rate 1 exceeds maximum" and the
        # clip is unusable. VP9-in-WebM decodes everywhere.
        "12": {"class_type": "SaveWEBM",
               "inputs": {"images": ["11", 0], "filename_prefix": "adforge_clip",
                          "codec": "vp9", "fps": float(fps), "crf": 28.0}},
    }
