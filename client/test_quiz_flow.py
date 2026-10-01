"""퀴즈 여러 문항 — 툴 호출이 히스토리에 남아야 다음 문항에서도 판정 툴을 부른다.

로봇 보고(2026-10-01 저녁, 세션 fa0f5948): 1번 문제는 정상 채점됐는데 2번부터 사용자 답에
모델이 **1토큰 빈 응답**(툴도 말도 없음)을 냈다. 히스토리가 툴 호출을 버려서 1번 답이
"사용자 답 → 모티 무응답"으로 남았고, 모델이 그 패턴을 따라 했다. 로봇의 실제 퀴즈
페르소나로 오프라인 재현: 옛 히스토리 submit_guess 3/15, tool 메시지로 남기면 15/15.

이 테스트는 로봇이 하는 그대로 돈다: 문제 주입 → 사용자 답 → submit_guess →
"[OK] 침묵. 대기." → 정답 공개 주입 → 다음 문제. 확인하는 것:
  - 세 문항 모두 submit_guess를 부른다
  - 침묵 지시를 받은 턴에서는 아무 말도 하지 않는다 (8/10 낭독 사고의 반대편)

Run: PYTHONPATH= .venv_tts/bin/python client/test_quiz_flow.py
     ... client/test_quiz_flow.py state/hellos/<id>.json   # 로봇이 보낸 실제 페르소나·툴로

간이 페르소나로는 "2번부터 툴을 안 부름" 증상 자체가 안 나온다(실제 퀴즈 페르소나에서만
재현됐다). 그래서 기록된 hello를 인자로 받는다 — state/는 공개 저장소에 안 올라가므로
기본값으로 쓸 수는 없다.
"""
import asyncio
import json
import sys
import uuid

import subprocess
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent

URI = "ws://127.0.0.1:8765"
SYSTEM = ("너는 로봇 모티야. 지금은 사진 퀴즈 중이다. 사용자가 사진에 대해 답을 말하면 "
          "**반드시 submit_guess 도구를 호출**하고 정답 여부를 직접 말하지 마라. "
          "도구 결과가 침묵하라고 하면 아무 말도 하지 마라. "
          "진행자 지시가 오면 그 지시대로 짧게 말해라. 이모지는 쓰지 마.")
TOOLS = [{"type": "function", "function": {
    "name": "submit_guess", "description": "사용자의 답을 채점한다.",
    "parameters": {"type": "object",
                   "properties": {"guess_text": {"type": "string",
                                                 "description": "사용자가 말한 답"}},
                   "required": ["guess_text"]}}}]
HOLD = "[OK] 침묵. 대기."
ITEMS = [("브로콜리", "음 나무처럼 보이는데"), ("빨래집게", "어 빨래 집게"),
         ("우산", "이거 우산 아니야?")]


def result_for(c: dict) -> str:
    # What the robot actually returns. "ok" for everything else was too kind: the
    # regression of 2026-10-01 night (quiz went silent after start_quiz) only shows up
    # once set_emotion is answered the way the robot answers it.
    if c["name"] == "submit_guess":
        return HOLD
    if c["name"] == "set_emotion":
        return f"emotion set to {(c.get('args') or {}).get('emotion')}"
    return "ok"


def speech(k: int) -> bytes | None:
    """A recorded answer for question k (`testdata/quiz_answer{k}.*`), as 16kHz mic audio
    plus 3s of silence to end the turn — or None, and the answer goes in as text.

    Text is not what the robot sends, and it shows: the brain suffixes injected text
    with "방금 한 말에 반응해줘." and the model sometimes answers *that* instead of
    grading. Answers synthesized with Piper were tried and are worse than useless — it is
    Moti's own voice, and the model replied "저는 로봇 모티야." (2026-10-01). It needs a
    person's voice: record the three answers in ITEMS.
    """
    src = next((ROOT / "testdata").glob(f"quiz_answer{k}.*"), None)
    if src is None:
        return None
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
                          "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
                         check=True, capture_output=True).stdout
    return pcm + b"\0" * 32000 * 3


async def turn(ws, text: str, pcm: bytes | None = None) -> tuple[list, str]:
    """One turn — injected text, or mic audio streamed in real time; answer tool calls;
    return (calls, what Moti said)."""
    if pcm is None:
        await ws.send(json.dumps({"t": "text", "text": text}))
    else:
        async def stream() -> None:
            for i in range(0, len(pcm), 3200):
                await ws.send(pcm[i:i + 3200])
                await asyncio.sleep(0.1)
        sender = asyncio.create_task(stream())
    calls, said = [], ""
    async for raw in ws:
        if isinstance(raw, bytes):
            continue
        m = json.loads(raw)
        if m["t"] == "transcript" and m["role"] == "model":
            said += m["text"]
        elif m["t"] == "transcript" and m["role"] == "user":
            print(f"     (들은 말: {m['text']!r})")
        elif m["t"] == "tool_call":
            calls += m["calls"]
            await ws.send(json.dumps({"t": "tool_result", "results": [
                {"id": c["id"], "name": c["name"], "result": result_for(c)}
                for c in m["calls"]]}))
        elif m["t"] in ("turn_complete", "error"):
            break
    if pcm is not None:
        sender.cancel()
    return calls, said.strip()


async def main() -> None:
    global SYSTEM, TOOLS
    if len(sys.argv) > 1:
        hello = json.load(open(sys.argv[1]))
        SYSTEM, TOOLS = hello["system"], hello["tools"]
        print(f"  실제 hello: {sys.argv[1]} ({len(SYSTEM):,}자, 툴 {len(TOOLS)}개)")
    failures: list[str] = []
    async with websockets.connect(URI, max_size=None) as ws:
        await ws.send(json.dumps({"t": "hello", "system": SYSTEM, "tools": TOOLS,
                                  "session_id": uuid.uuid4().hex,
                                  "backchannel": False}))
        assert json.loads(await ws.recv())["t"] == "ready"
        for k, (answer, guess) in enumerate(ITEMS, 1):
            ordinal = "첫 문제" if k == 1 else "다음 문제"
            _, asked = await asyncio.wait_for(turn(ws, f'(진행자 지시: 화면에 {ordinal}({k}/3)가 떴습니다. '
                                                       '"이 물건은 무엇일까요?"라고 물어보세요.)'), 60)
            print(f"  {k}번 문제 제시: {asked!r}")
            if not asked:
                failures.append(f"{k}번: 문제를 묻는 턴에서 아무 말도 안 했다")
            calls, said = await asyncio.wait_for(turn(ws, guess, speech(k)), 60)
            graded = any(c["name"] == "submit_guess" for c in calls)
            print(f"  {k}번 '{guess}': submit_guess={'O' if graded else 'X'}  말한 것={said!r}")
            if not graded:
                failures.append(f"{k}번 문제에서 submit_guess를 안 불렀다")
            if graded and said:
                failures.append(f"{k}번: 침묵 지시를 받고도 말했다 — {said!r}")
            _, revealed = await asyncio.wait_for(turn(ws, f"(진행자 지시: 정답은 {answer}입니다. 정답을 "
                                                          "알려주고 짧게 반응하세요. 다음 문제를 미리 묻지 마세요.)"), 60)
            print(f"  {k}번 정답 공개: {revealed!r}")
            if not revealed:
                failures.append(f"{k}번: 정답 공개 턴에서 아무 말도 안 했다")
    if failures:
        print("\n실패:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\n퀴즈 3문항: 묻고, 채점 툴을 부르고, 판정 직후엔 침묵하고, 정답을 말한다")


if __name__ == "__main__":
    asyncio.run(main())
