"""EXP-8: end-to-end voice-to-voice latency, as a distribution (§12).

Single measurements have ranged 0.24–1.29s, so a single number would be meaningless.
§12 asks for mean and p95. Measured from the client, through the wire, which is what the
user actually experiences — not from inside the pipeline.

Reported in two forms, because they answer different questions:
  perceived   speaking stops -> first audio. What a person in front of the robot waits
              through, including the stop_secs silence the VAD needs (§9.1a).
  brain       turn detected -> first audio. What this code costs, with the wait removed.

An earlier version measured from "all trailing silence sent" and got *negative* numbers:
the brain endpoints at stop_secs and starts replying while the client is still sending
the rest of the silence. That was an artifact of the test, not a result.

Server must be running. Then:
  PYTHONPATH= .venv_tts/bin/python client/exp8_latency.py [runs]
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
TAIL_SILENCE_SEC = 2.5
STOP_SECS = 1.5          # must match TurnDetector's stop_secs (§9.1a)
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


async def one(stem: str) -> tuple[float, float, str]:
    pcm = pcm16k(stem)
    first_audio = None
    said: list[str] = []

    # A fresh session each run: resuming would grow history and confound the timing.
    async with local_live.connect(URI, config=CONFIG) as session:
        local_live.SESSION_ID = local_live.uuid.uuid4().hex

        async def send() -> float:
            for i in range(0, len(pcm), CHUNK):
                await session.send_realtime_input(SimpleNamespace(data=pcm[i:i + CHUNK]))
                await asyncio.sleep(CHUNK / 2 / 16000)
            stopped = time.perf_counter()
            for _ in range(int(TAIL_SILENCE_SEC * 16000 * 2 / CHUNK)):
                await session.send_realtime_input(SimpleNamespace(data=b"\x00" * CHUNK))
                await asyncio.sleep(CHUNK / 2 / 16000)
            return stopped

        sender = asyncio.create_task(send())
        while True:
            ended = False
            async for m in session.receive():
                if m.data and first_audio is None:
                    first_audio = time.perf_counter()
                sc = m.server_content
                if sc and sc.output_transcription and sc.output_transcription.text:
                    said.append(sc.output_transcription.text)
                if sc and (sc.turn_complete or sc.interrupted):
                    ended = True
            if ended:
                break
        stopped_at = await sender

    detected_at = stopped_at + STOP_SECS      # when the VAD can first know the turn ended
    return (first_audio - detected_at, first_audio - stopped_at, "".join(said).strip())


def report(name: str, xs: list[float]) -> None:
    o = sorted(xs)
    p95 = o[min(int(len(o) * 0.95), len(o) - 1)]
    print(f"  {name:<12} mean {statistics.mean(xs):5.2f}s  p95 {p95:5.2f}s  "
          f"min {o[0]:5.2f}s  max {o[-1]:5.2f}s")


async def main(runs: int) -> None:
    brain, perceived = [], []
    for i in range(runs):
        stem = CLIPS[i % len(CLIPS)]
        b, p, said = await one(stem)
        brain.append(b)
        perceived.append(p)
        print(f"[{i+1:2d}] {stem:<12} brain {b:5.2f}s  perceived {p:5.2f}s  | {said[:40]}")

    print(f"\nEXP-8 (n={runs}):")
    report("brain", brain)
    report("perceived", perceived)
    share = STOP_SECS / statistics.mean(perceived) * 100
    print(f"\n  체감 지연의 {share:.0f}%가 stop_secs({STOP_SECS}s) 대기다 — 뇌가 아니라 "
          f"턴 감지가 병목이라는 뜻이고, 그게 Q19(smart-turn)의 가치다.")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
