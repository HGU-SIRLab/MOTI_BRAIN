"""EXP-12 A/B: does acknowledging during the pause change what the user hears, and when?

Real latency cannot improve — the reply is generated at the same speed either way. What
changes is the silence: §13.4 measured 1.98s of it, and §4.2 argues that filling it is
where perceived liveness actually comes from. So the number that matters is
**time to first sound**, not time to first reply.

Server must be running. Then:
  PYTHONPATH= .venv_tts/bin/python client/exp12_backchannel.py [runs]
"""
from __future__ import annotations

import asyncio
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client import local_live  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URI = "ws://127.0.0.1:8765"
CHUNK = 1600
TAIL = 3.0
CLIPS = ["a1_tired", "a2_happy", "a3_anxious", "c1_neutral"]
CONFIG = SimpleNamespace(
    system_instruction=("너는 공감 로봇 모티야. 사용자의 말에 따뜻하게 공감하며 대화해. "
                        "답변은 2~3문장으로 짧게 해. 이모지는 쓰지 마."),
    tools=[])


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


async def one(stem: str, backchannel: bool, after: float | None = None) -> tuple[float, float]:
    pcm = pcm16k(stem)
    first_sound = first_reply = None

    local_live.SESSION_ID = local_live.uuid.uuid4().hex
    async with local_live.connect(URI, config=CONFIG, backchannel=backchannel,
                                  backchannel_after=after) as session:
        async def send() -> float:
            for i in range(0, len(pcm), CHUNK):
                await session.send_realtime_input(SimpleNamespace(data=pcm[i:i + CHUNK]))
                await asyncio.sleep(CHUNK / 2 / 16000)
            stopped = time.perf_counter()
            for _ in range(int(TAIL * 16000 * 2 / CHUNK)):
                await session.send_realtime_input(SimpleNamespace(data=b"\x00" * CHUNK))
                await asyncio.sleep(CHUNK / 2 / 16000)
            return stopped

        sender = asyncio.create_task(send())
        while True:
            ended = False
            async for m in session.receive():
                if m.data:
                    now = time.perf_counter()
                    if first_sound is None:
                        first_sound = now
                    if first_reply is None and session.audio_kind == "reply":
                        first_reply = now
                sc = m.server_content
                if sc and (sc.turn_complete or sc.interrupted):
                    ended = True
            if ended:
                break
        stopped_at = await sender

    return first_sound - stopped_at, first_reply - stopped_at


async def main(runs: int) -> None:
    results: dict[bool, list[tuple[float, float]]] = {False: [], True: []}
    for i in range(runs):
        stem = CLIPS[i % len(CLIPS)]
        for on in (False, True):
            snd, rep = await one(stem, on)
            results[on].append((snd, rep))
            print(f"[{i+1}] {stem:<12} 맞장구 {'ON ' if on else 'OFF'}  "
                  f"첫 소리 {snd:5.2f}s  첫 응답 {rep:5.2f}s")

    print(f"\nEXP-12 (n={runs} per arm):")
    for on in (False, True):
        snd = [a for a, _ in results[on]]
        rep = [b for _, b in results[on]]
        print(f"  맞장구 {'ON ' if on else 'OFF'}  첫 소리 mean {statistics.mean(snd):.2f}s"
              f"   첫 응답 mean {statistics.mean(rep):.2f}s")
    off = statistics.mean([a for a, _ in results[False]])
    on_ = statistics.mean([a for a, _ in results[True]])
    print(f"\n  첫 소리까지의 침묵이 {off:.2f}s -> {on_:.2f}s ({off - on_:+.2f}s)")
    print("  첫 '응답'은 바뀌지 않아야 정상이다 — 실제 지연은 그대로이고 침묵만 채운다.")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4))
