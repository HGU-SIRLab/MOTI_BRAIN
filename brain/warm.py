"""Persona prewarm — pay the 18K prefill while nobody is waiting.

A persona that was never prefilled costs ~18-21s before the first sound (spec §13.9),
and `launcher.py` pushes its greeting turn the moment it connects, so someone standing
in front of the robot waits that long. vLLM's prefix cache lives as long as the vLLM
process, so warming the exact bytes the robot will send makes that greeting instant.

*Exact bytes* is the whole point, and it is why this records what the robot actually
sent instead of rebuilding it. The previous version imported `build_persona_system_
instruction` from the local MOTI-HRI clone, which is frozen on `jetson-moti`; the
robot's `local-brain-integration` persona diverges from it at char 592 (1.7%), so the
warm-up hit almost nothing and an unrecognised-face session waited 21.6s (2026-09-29,
session 464842aa, 20,806 prompt tokens, no cache).

So: every `hello` is saved under `state/hellos/` (system prompt + tool schemas, as sent),
and warming replays the most recent ones. Tools go in too — the Gemma 4 template appends
them to the system turn, so without them the cached prefix stops ~2.4K tokens short.

The files hold the persona including a user's remembered facts. `state/` is gitignored
— this repository is public.

    PYTHONPATH= .venv_tts/bin/python brain/warm.py          # warm now
    PYTHONPATH= .venv_tts/bin/python brain/warm.py --wait   # wait for vLLM first (boot)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("brain.warm")

HELLOS = Path(__file__).resolve().parent.parent / "state" / "hellos"
# KV budget is ~3.7 personas (67,712 tokens / 18,344). Warming more than fits just makes
# them evict each other, and the one we care most about (unknown user) could lose.
KEEP = 3
# Below this a cold prefill is under a second (a browser test persona is ~100 chars), so
# spending a KEEP slot on it would only evict a persona that actually costs 18s.
MIN_CHARS = 5000
VLLM = "http://127.0.0.1:8000"
MODEL = "google/gemma-4-E4B-it"


def record(system: str, tools: list) -> None:
    """Save one robot hello. Same persona again only refreshes its mtime (= recency)."""
    blob = json.dumps({"system": system, "tools": tools}, ensure_ascii=False)
    path = HELLOS / f"{hashlib.sha1(blob.encode()).hexdigest()[:12]}.json"
    try:
        HELLOS.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.touch()
        else:
            path.write_text(blob)
        for old in _newest()[KEEP:]:
            old.unlink(missing_ok=True)
    except OSError as exc:          # never let bookkeeping break a live session
        log.warning("could not record hello: %s", exc)


def _newest() -> list[Path]:
    return sorted(HELLOS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def vllm_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{VLLM}/v1/models", timeout=5):
            return True
    except OSError:
        return False


def _warm_one(hello: dict) -> float:
    # Same shape as pipeline.Session._post, so the template renders the same prefix.
    body = {"model": MODEL, "max_tokens": 1, "temperature": 0,
            "messages": [{"role": "system", "content": hello["system"]},
                         {"role": "user", "content": "."}]}
    if hello.get("tools"):
        body["tools"] = hello["tools"]
        body["tool_choice"] = "auto"
    req = urllib.request.Request(f"{VLLM}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.perf_counter() - t0


def warm(wait: bool = False) -> int:
    """Warm the recorded personas, oldest first so the newest ends up most recent in LRU."""
    if wait:
        while not vllm_ready():
            time.sleep(20)
    files = list(reversed(_newest()[:KEEP]))
    if not files:
        log.info("prewarm: no recorded hello yet — the first robot session will record one")
        return 0
    for f in files:
        try:
            dt = _warm_one(json.loads(f.read_text()))
            log.info("prewarm: %s %.2fs", f.stem, dt)
        except (OSError, ValueError) as exc:
            log.warning("prewarm: %s failed: %s", f.stem, exc)
    return len(files)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warm(wait="--wait" in sys.argv)
