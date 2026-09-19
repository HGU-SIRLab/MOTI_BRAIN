"""§11.5 [MANDATORY]: a dropped link must not wipe the conversation.

Talks, disconnects, reconnects with the same session id, then asks the brain about what
was said before the drop. Remembering it is the whole requirement — over Tailscale the
link will drop, and launcher.py's reconnect loop is built to survive that.

Server must be running. Then:
  PYTHONPATH= .venv_tts/bin/python client/test_reconnect.py
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client import local_live  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URI = "ws://127.0.0.1:8765"
CHUNK = 1600
CONFIG = SimpleNamespace(
    system_instruction="너는 공감 로봇 모티야. 2~3문장으로 짧게 답해. 이모지 금지.",
    tools=[])


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


async def drain(session) -> str:
    """Consume one turn, returning what Moti said."""
    said = []
    while True:
        got_end = False
        async for m in session.receive():
            sc = m.server_content
            if sc and sc.output_transcription and sc.output_transcription.text:
                said.append(sc.output_transcription.text)
            if sc and (sc.turn_complete or sc.interrupted):
                got_end = True
        if got_end:
            return "".join(said).strip()


async def speak(session, pcm: bytes) -> None:
    for i in range(0, len(pcm), CHUNK):
        await session.send_realtime_input(SimpleNamespace(data=pcm[i:i + CHUNK]))
        await asyncio.sleep(CHUNK / 2 / 16000)
    for _ in range(int(2.5 * 16000 * 2 / CHUNK)):
        await session.send_realtime_input(SimpleNamespace(data=b"\x00" * CHUNK))
        await asyncio.sleep(CHUNK / 2 / 16000)


async def main() -> None:
    print(f"session_id: {local_live.SESSION_ID[:8]}")

    async with local_live.connect(URI, config=CONFIG) as session:
        assert not session.resumed, "첫 연결인데 resumed로 왔다"
        print("1차 연결 — 발화 전송")
        asyncio.create_task(speak(session, pcm16k("a1_tired")))
        reply = await asyncio.wait_for(drain(session), timeout=120)
        print(f"  [모티] {reply[:70]}")

    print("연결 끊김 (링크 다운 시뮬레이션)")
    await asyncio.sleep(1.0)

    async with local_live.connect(URI, config=CONFIG) as session:
        await asyncio.sleep(0.5)          # let `ready` arrive
        print(f"2차 연결 — resumed={session.resumed}, "
              f"이어받은 턴={session.turns_carried}")
        assert session.resumed, "재연결인데 새 세션으로 처리됐다 — 대화가 날아간다"
        assert session.turns_carried >= 1, session.turns_carried

        await session.send_client_content(turns=SimpleNamespace(
            parts=[SimpleNamespace(text="내가 방금 전에 뭐 때문에 힘들다고 했는지 기억해?")]))
        reply = await asyncio.wait_for(drain(session), timeout=120)
        print(f"  [모티] {reply[:110]}")

    # The prior turn was about being exhausted from consecutive all-nighters.
    if not any(k in reply for k in ("밤", "피곤", "잠", "지치", "수면", "새우")):
        print("\n재연결 후 직전 대화를 기억하지 못했다 — 세션 복구 실패")
        sys.exit(1)
    print("\n재연결 후에도 대화가 이어짐 — §11.5 충족")


if __name__ == "__main__":
    asyncio.run(main())
