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

import websockets

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


async def turn(ws, text: str, on_call=None) -> tuple[list, str]:
    """Send one text turn; answer tool calls; return (calls, what Moti said)."""
    await ws.send(json.dumps({"t": "text", "text": text}))
    calls, said = [], ""
    async for raw in ws:
        if isinstance(raw, bytes):
            continue
        m = json.loads(raw)
        if m["t"] == "transcript" and m["role"] == "model":
            said += m["text"]
        elif m["t"] == "tool_call":
            calls += m["calls"]
            await ws.send(json.dumps({"t": "tool_result", "results": [
                {"id": c["id"], "name": c["name"],
                 "result": HOLD if c["name"] == "submit_guess" else "ok"}
                for c in m["calls"]]}))
        elif m["t"] in ("turn_complete", "error"):
            return calls, said.strip()
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
                                  "backchannel": False, "speculate": False}))
        assert json.loads(await ws.recv())["t"] == "ready"
        for k, (answer, guess) in enumerate(ITEMS, 1):
            ordinal = "첫 문제" if k == 1 else "다음 문제"
            await asyncio.wait_for(turn(ws, f'(진행자 지시: 화면에 {ordinal}({k}/3)가 떴습니다. '
                                            '"이 물건은 무엇일까요?"라고 물어보세요.)'), 60)
            calls, said = await asyncio.wait_for(turn(ws, guess), 60)
            graded = any(c["name"] == "submit_guess" for c in calls)
            print(f"  {k}번 '{guess}': submit_guess={'O' if graded else 'X'}  말한 것={said!r}")
            if not graded:
                failures.append(f"{k}번 문제에서 submit_guess를 안 불렀다")
            if graded and said:
                failures.append(f"{k}번: 침묵 지시를 받고도 말했다 — {said!r}")
            await asyncio.wait_for(turn(ws, f"(진행자 지시: 정답은 {answer}입니다. 정답을 알려주고 "
                                            "짧게 반응하세요. 다음 문제를 미리 묻지 마세요.)"), 60)
    if failures:
        print("\n실패:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\n퀴즈 3문항: 매번 채점 툴 호출, 침묵 지시 준수")


if __name__ == "__main__":
    asyncio.run(main())
