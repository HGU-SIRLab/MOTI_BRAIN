"""Drive the shim the way `launcher.py` drives Gemini, without the robot.

This mirrors launcher's recv_loop field-for-field — `message.data`,
`sc.input_transcription.text`, `sc.output_transcription.text`, `message.tool_call
.function_calls` (name/args/id), `sc.turn_complete`, `sc.interrupted`, plus the
`go_away` / `session_resumption_update` checks it makes on every message. If the shim
is right, this loop needs no changes to run against the real robot's code.

Server must be running. Then:
  PYTHONPATH= .venv_tts/bin/python client/test_local_live.py
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client import local_live  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URI = "ws://127.0.0.1:8765"
CHUNK = 1600

called: list[tuple[str, dict]] = []


# Plain callables with hints, exactly how launcher.py supplies tools.
def set_emotion(emotion: Literal["neutral", "happy", "sad", "tender"]) -> str:
    """로봇 표정을 바꾼다."""
    called.append(("set_emotion", {"emotion": emotion}))
    return "ok"


def remember_fact(field: str, value: str, confidence: str = "certain") -> str:
    """대화 중 알게 된 사용자 정보를 저장한다."""
    called.append(("remember_fact", {"field": field, "value": value}))
    return "saved"


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


def test_tool_schemas() -> None:
    schemas = local_live.tool_schemas([set_emotion, remember_fact])
    by_name = {s["function"]["name"]: s["function"] for s in schemas}
    assert set(by_name) == {"set_emotion", "remember_fact"}, by_name
    assert by_name["set_emotion"]["parameters"]["properties"]["emotion"]["enum"] == [
        "neutral", "happy", "sad", "tender"], by_name["set_emotion"]
    # confidence has a default, so it must not be required
    assert by_name["remember_fact"]["parameters"]["required"] == ["field", "value"]
    assert by_name["set_emotion"]["description"].startswith("로봇 표정")
    print("tool_schemas OK — 평범한 callable에서 스키마 추출")


async def main() -> None:
    test_tool_schemas()

    config = SimpleNamespace(
        system_instruction=("너는 공감 로봇 모티야. 따뜻하게 공감하며 2~3문장으로 답해. "
                            "이모지 금지. 감정이 느껴지면 set_emotion을 호출해."),
        tools=[set_emotion, remember_fact])

    audio_chunks: list[bytes] = []
    turn_user, turn_moti = [], []
    saw_turn_complete = False

    async with local_live.connect(URI, config=config) as session:
        pcm = pcm16k("a1_tired")
        print(f"마이크 스트리밍: {len(pcm) / 2 / 16000:.1f}s")

        async def stream() -> None:
            for i in range(0, len(pcm), CHUNK):
                await session.send_realtime_input(
                    SimpleNamespace(data=pcm[i:i + CHUNK]))
                await asyncio.sleep(CHUNK / 2 / 16000)
            for _ in range(int(2.5 * 16000 * 2 / CHUNK)):
                await session.send_realtime_input(
                    SimpleNamespace(data=b"\x00" * CHUNK))
                await asyncio.sleep(CHUNK / 2 / 16000)

        sender = asyncio.create_task(stream())

        # ---- this block is launcher.py's recv_loop, structurally unchanged ----
        while not saw_turn_complete:
            async for message in session.receive():
                if (message.session_resumption_update
                        and message.session_resumption_update.resumable):
                    pass                                   # brain never sends these
                if message.go_away:
                    pass

                sc = message.server_content
                if sc and sc.interrupted:
                    print("  interrupted")

                if message.data:
                    audio_chunks.append(message.data)

                if sc and sc.input_transcription and sc.input_transcription.text:
                    turn_user.append(sc.input_transcription.text)
                if sc and sc.output_transcription and sc.output_transcription.text:
                    turn_moti.append(sc.output_transcription.text)

                if message.tool_call:
                    responses = []
                    for fc in message.tool_call.function_calls:
                        fn = {"set_emotion": set_emotion,
                              "remember_fact": remember_fact}.get(fc.name)
                        result = fn(**(fc.args or {})) if fn else "unknown"
                        print(f"  🔧 {fc.name}({fc.args}) -> {result}")
                        responses.append(SimpleNamespace(
                            id=fc.id, name=fc.name, response={"result": result}))
                    await session.send_tool_response(function_responses=responses)

                if sc and sc.turn_complete:
                    saw_turn_complete = True
                    break
        # ---- end of launcher-shaped block ----

        sender.cancel()
        rate = session.output_rate

    print(f"\n  [나]   {''.join(turn_user)}")
    print(f"  [모티] {''.join(turn_moti)}")

    assert turn_user, "사용자 전사가 오지 않았다 — 대화 기록이 비게 된다"
    assert turn_moti, "모티 전사가 오지 않았다"
    assert audio_chunks, "오디오가 오지 않았다"
    assert saw_turn_complete, "turn_complete를 못 받아 launcher는 영원히 대기한다"
    assert called, "툴이 실행되지 않았다"
    for name, args in called:
        assert isinstance(args, dict), (name, args)      # fn(**args) must work

    out = ROOT / "tts_eval" / "shim_reply.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate or 22050)
        w.writeframes(b"".join(audio_chunks))
    secs = sum(len(c) for c in audio_chunks) / 2 / (rate or 22050)
    print(f"  실행된 툴: {called}")
    print(f"\n응답 음성 {secs:.1f}s @ {rate}Hz -> {out}")
    print("launcher.py 형태의 루프가 수정 없이 동작함")


if __name__ == "__main__":
    asyncio.run(main())
