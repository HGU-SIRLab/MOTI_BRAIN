"""A stand-in for the robot, so the whole loop can be exercised without hardware.

Does what `launcher.py` does against Gemini Live: opens a session, streams mic PCM
continuously (no client-side VAD — the brain decides turns), executes tool calls and
returns results, and collects the reply audio. Writes the reply to a WAV so the result
can be listened to, not just asserted on.

Run the server first, then:
  PYTHONPATH= .venv_tts/bin/python brain/fake_robot.py [clip_stem]
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
URI = "ws://127.0.0.1:8765"
CHUNK = 1600                    # 100ms of PCM16 @16kHz, about one mic callback
TAIL_SILENCE_SEC = 2.5          # the pause that ends a turn (brain's stop_secs is 1.5)

SYSTEM = ("너는 공감 로봇 모티야. 사용자의 말에 따뜻하게 공감하며 대화해. "
          "답변은 2~3문장으로 짧게 해. 이모지는 쓰지 마. "
          "사용자의 감정이 느껴지면 set_emotion을 호출해.")
TOOLS = [{"type": "function", "function": {
    "name": "set_emotion", "description": "로봇 표정을 바꾼다",
    "parameters": {"type": "object", "properties": {"emotion": {
        "type": "string", "enum": ["neutral", "happy", "sad", "tender", "excited"]}},
        "required": ["emotion"]}}}]


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


async def main(stem: str = "a1_tired") -> None:
    pcm = pcm16k(stem)
    chunks, rate, done = [], None, asyncio.Event()
    t_sent_end = 0.0
    first_audio_at = None

    async with websockets.connect(URI, max_size=None) as ws:
        await ws.send(json.dumps({"t": "hello", "system": SYSTEM, "tools": TOOLS}))

        async def receive() -> None:
            nonlocal rate, first_audio_at
            pending_rate = None
            async for msg in ws:
                if isinstance(msg, bytes):
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter()
                    chunks.append(msg)
                    rate = pending_rate
                    continue
                m = json.loads(msg)
                kind = m.get("t")
                if kind == "audio":
                    pending_rate = m["rate"]
                elif kind == "transcript":
                    print(f"  [{m['role']}] {m['text']}")
                elif kind == "tool_call":
                    # The robot is where tools actually run (§1). Pretend, and report back.
                    results = []
                    for c in m["calls"]:
                        print(f"  tool   {c['name']}({c['arguments']}) -> ok")
                        results.append({"id": c.get("id", ""), "name": c["name"],
                                        "result": "ok"})
                    await ws.send(json.dumps({"t": "tool_result", "results": results}))
                elif kind == "turn_complete":
                    done.set()
                elif kind == "error":
                    print(f"  ERROR  {m['detail']}")
                    done.set()

        receiver = asyncio.create_task(receive())

        # Stream in real time. Faster would not test what the brain does with a live mic.
        print(f"마이크 스트리밍: {stem} ({len(pcm) / 2 / 16000:.1f}s)")
        for i in range(0, len(pcm), CHUNK):
            await ws.send(pcm[i:i + CHUNK])
            await asyncio.sleep(CHUNK / 2 / 16000)
        silence = b"\x00" * CHUNK
        for _ in range(int(TAIL_SILENCE_SEC * 16000 * 2 / CHUNK)):
            await ws.send(silence)
            await asyncio.sleep(CHUNK / 2 / 16000)
        t_sent_end = time.perf_counter()
        print("발화 끝, 침묵 전송 완료 — 뇌의 응답 대기")

        await asyncio.wait_for(done.wait(), timeout=120)
        receiver.cancel()

    if not chunks:
        print("\n오디오를 받지 못했다")
        sys.exit(1)

    out = ROOT / "tts_eval" / f"roundtrip_{stem}.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate or 22050)
        w.writeframes(b"".join(chunks))
    secs = sum(len(c) for c in chunks) / 2 / (rate or 22050)
    print(f"\n응답 음성 {secs:.1f}s @ {rate}Hz -> {out}")
    if first_audio_at:
        print(f"침묵 종료 시점부터 첫 오디오까지: {first_audio_at - t_sent_end:.2f}s")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "a1_tired"))
