"""
OpenAI-compatible chat client.

Ollama (/v1) and vLLM (/v1) both speak this dialect, so one client covers both
backends and the user can switch with a single setting.
"""

from __future__ import annotations

import json
import logging
import re
import time

import httpx

from ..config import settings

log = logging.getLogger("adforge.llm")


class LLMError(RuntimeError):
    pass


_THINK_TAGS = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I
)


class OutOfMemory(LLMError):
    """The backend could not fit the model. Caller may retry with a smaller one."""


def _is_oom(msg: str) -> bool:
    m = msg.lower()
    return any(
        s in m
        for s in ("out of memory", "cudamalloc", "unable to allocate", "oom")
    )


def _content_of(data: dict) -> str:
    """Pull the answer out of a response, tolerating reasoning models.

    Reasoning models (qwen3.x, glm-4.x, deepseek-r1) return their chain of
    thought in a separate `reasoning` field and leave `content` empty when the
    token budget runs out mid-thought. Reading `content` blindly yields an
    empty post that then fails the length gate for no obvious reason, so this
    is handled explicitly rather than left to surface as a mystery.
    """
    msg = data["choices"][0].get("message", {})
    content = (msg.get("content") or "").strip()
    if content:
        return _THINK_TAGS.sub("", content).strip()

    if msg.get("reasoning") or msg.get("reasoning_content"):
        raise LLMError(
            "model returned only reasoning tokens and no answer - it is a "
            "reasoning model that ran out of budget. Raise max_tokens or set "
            "a non-reasoning instruct model for this role."
        )
    return ""


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.85,
    max_tokens: int = 1200,
    top_p: float = 0.92,
    stop: list[str] | None = None,
    retries: int = 2,
) -> str:
    model = model or settings.model_writer
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=settings.llm_timeout) as c:
                r = c.post(f"{settings.llm_base}/chat/completions", json=payload)
                if r.status_code >= 400:
                    body = r.text[:600]
                    if _is_oom(body):
                        # Retrying the same model will not help; the caller
                        # decides whether to fall back to something smaller.
                        raise OutOfMemory(f"{model} does not fit: {body[:300]}")
                    raise LLMError(f"{r.status_code}: {body[:400]}")
                data = r.json()
            text = _content_of(data)
            if text:
                return text
            last = LLMError("empty completion")
            log.warning("empty completion from %s on attempt %d", model, attempt + 1)
        except OutOfMemory:
            raise
        except Exception as e:  # noqa: BLE001 - retry any transport/decode failure
            last = e
            log.warning("llm attempt %d/%d failed: %s", attempt + 1, retries + 1, e)
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"chat failed after {retries + 1} attempts: {last}") from last


def chat_with_fallback(
    messages: list[dict], models: list[str], **kw
) -> tuple[str, str]:
    """Try each model in order, dropping to the next on OOM.

    Returns (text, model_actually_used). Lets a box that normally runs a 70B
    keep producing when something else has taken the VRAM, instead of failing
    the whole scheduled run.
    """
    last: Exception | None = None
    for m in models:
        try:
            return chat(messages, model=m, **kw), m
        except OutOfMemory as e:
            log.warning("%s did not fit, trying next model: %s", m, e)
            last = e
    raise LLMError(f"no model in {models} could run: {last}") from last


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def chat_json(
    messages: list[dict],
    models: list[str] | None = None,
    temperature: float = 0.4,
    max_tokens: int = 1200,
    retries: int = 3,
) -> dict:
    """Chat, then coerce the reply to a dict.

    Local models fence their JSON, prepend prose, or emit a bare object inside
    a sentence roughly a third of the time, so parsing is deliberately
    forgiving rather than strict.
    """
    chain = models or [settings.model_fast]
    last: Exception | None = None
    msgs = list(messages)
    for attempt in range(retries):
        raw, _ = chat_with_fallback(
            msgs, chain, temperature=temperature, max_tokens=max_tokens
        )
        for candidate in _json_candidates(raw):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError as e:
                last = e
        msgs = list(messages) + [
            {"role": "assistant", "content": raw[:600]},
            {
                "role": "user",
                "content": "That was not valid JSON. Reply with ONLY the JSON "
                "object, no prose, no code fence.",
            },
        ]
        temperature = 0.1
    raise LLMError(f"no JSON object in model output after {retries} attempts: {last}")


def _json_candidates(raw: str):
    raw = raw.strip()
    if m := _FENCE.search(raw):
        yield m.group(1)
    yield raw
    # Widest balanced-looking span, for replies wrapped in commentary.
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j > i:
        yield raw[i : j + 1]


def available_models() -> list[str]:
    try:
        with httpx.Client(timeout=10) as c:
            r = c.get(f"{settings.llm_base}/models")
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []))
    except Exception as e:  # noqa: BLE001
        log.warning("model list unavailable: %s", e)
        return []


def health() -> tuple[bool, str]:
    try:
        models = available_models()
        if not models:
            return False, f"{settings.llm_backend} reachable but exposes no models"
        missing = [
            m
            for m in {settings.model_writer, settings.model_critic, settings.model_fast}
            if m not in models
        ]
        if missing:
            return False, f"missing model(s): {', '.join(sorted(missing))}"
        return True, f"{settings.llm_backend}: {len(models)} models"
    except Exception as e:  # noqa: BLE001
        return False, str(e)
