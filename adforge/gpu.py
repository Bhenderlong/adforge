"""
GPU arbitration between the LLM backend and ComfyUI.

Both want the same two cards. A 70B writer model needs essentially all of both
5090s, and ComfyUI holding a checkpoint is enough to make it fail with
`cudaMalloc failed: out of memory`. Left uncoordinated the scheduler dies
whenever copy generation and image generation overlap.

So the box has one mode at a time:

    with gpu(TEXT):   ComfyUI is told to unload, then the LLM runs
    with gpu(MEDIA):  the LLM is told to unload, then ComfyUI runs

The lock is an OS file lock, so it also holds across separate processes (the
web UI and the scheduler worker are not necessarily the same process).
Switching modes costs a model reload, so callers should batch: generate all the
copy for a run, then all the media.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import threading
import time
from pathlib import Path

import httpx

from .config import DATA, settings

log = logging.getLogger("adforge.gpu")

TEXT = "text"
MEDIA = "media"

_LOCK_FILE = DATA / "gpu.lock"
_MODE_FILE = DATA / "gpu.mode"


def _comfy_unload() -> None:
    """Ask ComfyUI to drop loaded models and free its cache."""
    try:
        with httpx.Client(timeout=30) as c:
            r = c.post(
                f"{settings.comfy_url}/free",
                json={"unload_models": True, "free_memory": True},
            )
            log.info("comfy /free -> %s", r.status_code)
    except Exception as e:  # noqa: BLE001 - comfy being down is fine here
        log.debug("comfy unload skipped: %s", e)


def _ollama_unload() -> None:
    """Unload every resident Ollama model by re-requesting it with keep_alive=0.

    Only Ollama supports this; a vLLM server owns its memory for its lifetime,
    so with vLLM the user must run ComfyUI on a separate box (or accept that
    only the smaller writer models fit).
    """
    if settings.llm_backend != "ollama":
        log.warning(
            "llm_backend=%s cannot release VRAM on demand; media generation may "
            "OOM if the server is holding a large model",
            settings.llm_backend,
        )
        return
    try:
        with httpx.Client(timeout=30) as c:
            running = c.get(f"{settings.ollama_url}/api/ps").json().get("models", [])
            for m in running:
                name = m.get("name") or m.get("model")
                if not name:
                    continue
                c.post(
                    f"{settings.ollama_url}/api/generate",
                    json={"model": name, "keep_alive": 0, "prompt": ""},
                )
                log.info("ollama unloaded %s", name)
    except Exception as e:  # noqa: BLE001
        log.debug("ollama unload skipped: %s", e)


def current_mode() -> str | None:
    try:
        return _MODE_FILE.read_text().strip() or None
    except OSError:
        return None


_local = threading.local()


@contextlib.contextmanager
def gpu(mode: str, settle: float = 3.0):
    """Acquire exclusive GPU use in `mode`, evicting the other consumer.

    Re-entrant within a thread: flock() on a second file descriptor would
    deadlock against the one this thread already holds, and nesting is natural
    here (write_post takes TEXT, and a caller may already hold it for a batch).
    """
    if mode not in (TEXT, MEDIA):
        raise ValueError(f"mode must be {TEXT!r} or {MEDIA!r}, got {mode!r}")

    held = getattr(_local, "mode", None)
    if held is not None:
        if held != mode:
            raise RuntimeError(
                f"cannot acquire GPU in {mode!r} while this thread holds {held!r}; "
                "finish the outer block first (batch text and media separately)"
            )
        yield  # already ours
        return

    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = os.open(_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    # Everything after the open lives in the try. Previously the flock and the
    # bookkeeping between it and the try were unprotected, so an exception
    # there leaked the descriptor - and os.open fds are not closed by GC, so
    # the flock was held for the life of the process. Every later gpu() call,
    # in this process and in a `--worker` one, would block on it forever.
    try:
        t0 = time.time()
        fcntl.flock(fh, fcntl.LOCK_EX)
        waited = time.time() - t0
        if waited > 1:
            log.info("waited %.1fs for the GPU lock", waited)
        _local.mode = mode
        if current_mode() != mode:
            log.info("switching GPU mode %s -> %s", current_mode(), mode)
            if mode == TEXT:
                _comfy_unload()
            else:
                _ollama_unload()
            # Freeing is asynchronous inside both servers; without a pause the
            # next allocation can still see the old resident memory.
            time.sleep(settle)
            _MODE_FILE.write_text(mode)
        yield
    finally:
        _local.mode = None
        fcntl.flock(fh, fcntl.LOCK_UN)
        os.close(fh)


def vram() -> list[dict]:
    """Per-GPU memory, for the UI status panel. Empty if nvidia-smi is absent."""
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, used, total = (p.strip() for p in line.split(","))
        gpus.append(
            {"index": int(idx), "name": name, "used_mb": int(used),
             "total_mb": int(total), "pct": round(100 * int(used) / int(total))}
        )
    return gpus
