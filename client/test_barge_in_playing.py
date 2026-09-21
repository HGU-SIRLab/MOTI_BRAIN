"""끼어들기가 '로봇이 아직 재생 중일 때'도 되는가.

기존 barge-in 테스트는 뇌가 **생성하는 중에** 끼어들었고 그건 잘 됐다(0.07초, §13.3).
실물에서 안 된 건 다른 창이다 — 뇌는 14초짜리 답변을 2초에 다 보내고 턴을 닫는다. 그 뒤
12초 동안 로봇은 계속 말하는데 뇌에는 취소할 `self.current`가 없어서, 그때 끼어든 발화가
barge-in이 아니라 새 턴이 됐다(2026-09-21 실물 세션: 세션 전체에서 취소 0건).

`fake_robot.py`가 오디오를 실시간으로 재생하지 않았기 때문에 이 창은 어떤 테스트에도
존재한 적이 없다. 이 테스트는 **받은 오디오만큼 실제로 기다렸다가** 끼어든다.

Run: PYTHONPATH= .venv_tts/bin/python client/test_barge_in_playing.py
"""
import asyncio
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from client import local_live  # noqa: E402

URI = "ws://127.0.0.1:8765"
CHUNK = 1600                     # 100ms @16kHz
CONFIG = SimpleNamespace(
    system_instruction=("너는 공감 로봇 모티야. 사용자의 말에 따뜻하게 공감하며 대화해. "
                        "답변은 2~3문장으로 해. 이모지는 쓰지 마."),
    tools=[])


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


async def stream(session, pcm: bytes) -> None:
    """실시간 속도로 흘려보낸다 — 로봇 마이크가 하는 그대로."""
    for i in range(0, len(pcm), CHUNK):
        await session.send_realtime_input(SimpleNamespace(data=pcm[i:i + CHUNK]))
        await asyncio.sleep(CHUNK / 2 / 16000)


async def main() -> None:
    first, second = pcm16k("turn_s2"), pcm16k("turn_s6")
    failures: list[str] = []

    async with local_live.connect(URI, config=CONFIG, backchannel=False) as s:
        local_live.SESSION_ID = local_live.uuid.uuid4().hex

        # 1) 말을 걸고 답변을 끝까지 받는다. 받은 오디오 길이를 재둔다 —
        #    그게 로봇이 앞으로 재생해야 할 시간이다.
        await stream(s, first)
        sender = asyncio.create_task(stream(s, b"\x00" * CHUNK * 60))   # 6초 무음
        audio_secs = 0.0
        async for m in s.receive():
            if m.data:
                audio_secs += len(m.data) / 2 / (s.output_rate or 24000)
        sender.cancel()
        print(f"받은 답변 오디오: {audio_secs:.1f}초 (로봇은 이만큼 재생 중)")
        if audio_secs < 3.0:
            failures.append(f"답변이 {audio_secs:.1f}초뿐 — 이 테스트의 전제가 안 선다")

        # 2) 재생이 한창일 때 끼어든다. 실물에서 사용자가 하는 그대로.
        await asyncio.sleep(1.0)
        print("재생 1.0초 지점에서 끼어든다...")
        t0 = time.perf_counter()
        barge = asyncio.create_task(stream(s, second))

        interrupted = False
        try:
            async def watch():
                nonlocal interrupted
                async for m in s.receive():
                    if m.server_content and m.server_content.interrupted:
                        interrupted = True
                        return
            await asyncio.wait_for(watch(), timeout=6.0)
        except asyncio.TimeoutError:
            pass
        dt = time.perf_counter() - t0
        barge.cancel()

        if interrupted:
            print(f"interrupted 수신 — {dt:.2f}초")
            if dt > 2.0:
                failures.append(f"끼어들기 반응이 {dt:.2f}초로 느리다")
        else:
            failures.append("interrupted가 오지 않았다 — 재생 중 끼어들기가 안 먹는다")

    print()
    if failures:
        print("실패:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("재생 중 끼어들기 정상")


if __name__ == "__main__":
    asyncio.run(main())
