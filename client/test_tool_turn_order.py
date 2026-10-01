"""툴만 부르고 말이 없는 턴 — 퀴즈 첫 실행에서 깨진 두 가지 (로봇 보고 2026-10-01).

1. **사용자 전사가 tool_call보다 먼저 와야 한다.** Gemini 순서이고, 로봇의 퀴즈 가드
   (`_NO_USER_SPEECH_GUARD`)가 이걸 전제로 "아직 아무 말도 안 했는데 부른 submit_guess"를
   거절한다. 순서가 뒤집혀 정상 답이 거절됐고 퀴즈가 1번 문제에서 멈췄다.
2. **툴만 부른 턴은 tool_result를 받아 한 번 더 말해야 한다.** `run_turn`은 했는데 투기적
   생성 경로(`_flush_speculation`)는 안 했다. start_quiz의 결과가 곧 "5개 항목을 안내하라"는
   지시라서, 참가자는 퀴즈 안내를 통째로 못 들었다.

실물 세션의 그 턴은 speculated였다. 그래서 투기 생성을 켠 경우와 끈 경우를 둘 다 돈다.
모델이 툴만 부르게 만드는 건 시스템 프롬프트로 강제한다 — 실물에선 가끔 일어나는 일이다.

Run: PYTHONPATH= .venv_tts/bin/python client/test_tool_turn_order.py
"""
import asyncio
import json
import subprocess
import sys
import uuid
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
URI = "ws://127.0.0.1:8765"
CHUNK = 1600                     # 100ms @16kHz
SYSTEM = ("너는 로봇 모티야. 사용자가 무슨 말을 하든 **말은 한 마디도 하지 말고** "
          "start_quiz 도구만 호출해. 도구 실행 결과를 받으면 그 결과의 지시를 따라 말해. "
          "이모지는 쓰지 마.")
TOOLS = [{"type": "function", "function": {
    "name": "start_quiz", "description": "퀴즈를 시작한다.",
    "parameters": {"type": "object", "properties": {}, "required": []}}}]
RESULT = "사용자에게 '퀴즈는 모두 세 문제입니다'라고 안내하라."


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


async def one(speculate: bool) -> list[str]:
    events: list[str] = []
    async with websockets.connect(URI, max_size=None) as ws:
        await ws.send(json.dumps({"t": "hello", "system": SYSTEM, "tools": TOOLS,
                                  "session_id": uuid.uuid4().hex,
                                  "backchannel": False, "speculate": speculate}))
        assert json.loads(await ws.recv())["t"] == "ready"

        async def stream() -> None:
            # 실시간 속도로, 꼬리 무음까지 — 투기 생성은 이 무음 중에 시작된다.
            pcm = pcm16k("turn_s2") + b"\x00" * CHUNK * 2 * 60
            for i in range(0, len(pcm), CHUNK * 2):
                await ws.send(pcm[i:i + CHUNK * 2])
                await asyncio.sleep(CHUNK / 16000)
        async def listen() -> None:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                if m["t"] == "transcript":
                    events.append(f"transcript:{m['role']}")
                elif m["t"] == "tool_call":
                    events.append("tool_call")
                    await ws.send(json.dumps({"t": "tool_result", "results": [
                        {"id": c.get("id"), "name": c["name"], "result": RESULT}
                        for c in m["calls"]]}))
                elif m["t"] in ("turn_complete", "error"):
                    events.append(m["t"])
                    return

        sender = asyncio.create_task(stream())
        try:
            await asyncio.wait_for(listen(), 60)
        except asyncio.TimeoutError:
            events.append("timeout")
        finally:
            sender.cancel()
    return events


def check(label: str, ev: list[str]) -> list[str]:
    print(f"  {label}: {' → '.join(ev)}")
    bad = []
    if "tool_call" not in ev:
        return [f"{label}: 모델이 툴을 안 불렀다 — 이 테스트의 전제가 안 선다(재실행)"]
    if "transcript:user" not in ev or ev.index("transcript:user") > ev.index("tool_call"):
        bad.append(f"{label}: 사용자 전사가 tool_call보다 늦다 (퀴즈 가드가 정상 답을 거절한다)")
    after = ev[ev.index("tool_call") + 1:]
    if "transcript:model" not in after:
        bad.append(f"{label}: tool_result 뒤 후속 발화가 없다 (퀴즈 안내가 통째로 빠진다)")
    if ev[-1] != "turn_complete":
        bad.append(f"{label}: turn_complete로 안 끝났다 ({ev[-1]})")
    return bad


async def main() -> None:
    failures: list[str] = []
    for spec in (True, False):
        label = "투기 생성 켬" if spec else "투기 생성 끔"
        failures += check(label, await one(spec))
    if failures:
        print("\n실패:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\n툴만 부른 턴: 전사 → 툴 → 후속 발화 → turn_complete 순서 정상 (두 경로 모두)")


if __name__ == "__main__":
    asyncio.run(main())
