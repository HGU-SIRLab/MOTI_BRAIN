"""로봇이 주입한 텍스트가 사용자 발화로 되돌아오지 않는가.

2026-09-22 실물 보고: 프라이버시 안내를 주입한 턴에서 로봇이 보낸 문장이
`input_transcription`으로 되돌아왔고, `launcher.py`가 그걸 사용자 발화로 로그에 찍어
`user_result/*/대화.txt`에 남겼다 — **연구 데이터 오염**이다.

원인은 음향 에코가 아니었다(로봇 쪽이 AEC 잔향을 실측해 배제했다). `launcher.py:307
inject_turn()`은 로봇의 대사를 `role="user"`로 보내는데 — Live API에서 모티가 먼저 말하게
하는 유일한 방법이다 — 뇌가 그 텍스트 파트를 `transcribe()`에 넣었고, 전사 프롬프트가
"이 오디오의 발화 내용만 그대로 받아적어"라서 모델이 고분고분 그대로 돌려줬다.
전사 경로가 **오디오만 온다고 가정**하고 쓰인 것이다.

뇌 서버가 떠 있어야 한다.
Run: PYTHONPATH= .venv_tts/bin/python client/test_injected_turn.py
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from client import local_live  # noqa: E402

URI = "ws://127.0.0.1:8765"
CFG = SimpleNamespace(
    system_instruction="너는 공감 로봇 모티야. 2~3문장으로 따뜻하게 답해. 이모지는 쓰지 마.",
    tools=[])
# `core/trust_notice.py`의 실제 문구 모양 그대로.
NOTICE = ("(진행자 지시: 아래 문장을 그대로 말해줘) 안내사항이 있습니다! "
          "저와 조형민님의 대화는 유출되지 않고, 오직 모티만 기억하고 있겠습니다.")


async def main() -> None:
    user_lines, model_lines = [], []

    async with local_live.connect(URI, config=CFG, backchannel=False) as s:
        local_live.SESSION_ID = local_live.uuid.uuid4().hex
        await s.send_client_content(
            turns=SimpleNamespace(parts=[SimpleNamespace(text=NOTICE)]),
            turn_complete=True)
        try:
            async def drain():
                async for m in s.receive():
                    sc = m.server_content
                    if not sc:
                        continue
                    if sc.input_transcription and sc.input_transcription.text:
                        user_lines.append(sc.input_transcription.text)
                    if sc.output_transcription and sc.output_transcription.text:
                        model_lines.append(sc.output_transcription.text)
            await asyncio.wait_for(drain(), timeout=90)
        except asyncio.TimeoutError:
            pass

    said = "".join(model_lines).strip()
    heard = "".join(user_lines).strip()
    print(f"  모티가 말한 것: {said[:72]}…")
    print(f"  input_transcription: {heard!r}")

    failures = []
    if heard:
        failures.append(f"주입 턴에서 사용자 전사가 나왔다 — 대화록이 오염된다: {heard[:60]!r}")
    # 뇌가 책임지는 불변식은 "주입 턴이 도달해서 모티가 말을 했다"까지다. 받은 문장을
    # **그대로** 따라 읽게 하는 건 로봇 페르소나의 문구이고(로봇 소유), 이 테스트의 시험용
    # 시스템 프롬프트에는 그 지시가 없다. 여기서 그걸 단언하면 남의 책임을 재는 셈이다.
    if not said:
        failures.append("주입 턴에 모티가 아무 말도 안 했다 — 주입 경로가 깨졌다")

    print()
    if failures:
        print("실패:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("주입 턴 정상 — 모티는 말했고, 사용자 전사는 나가지 않았다")


if __name__ == "__main__":
    asyncio.run(main())
