"""Guard the failure path the robot hangs on.

`local_live.py` handled {"t":"error"} with `continue` from its very first commit
(76dfc89, six hours before it bit). The brain reported the failure correctly; the shim
dropped it and waited for a `turn_complete` that was never coming. launcher.py would
hang identically — its recv_loop also exits only on turn_complete — so on the real
robot this is a wedged conversation with no error printed anywhere.

Nothing caught it because nothing ever made the brain fail a turn. These two checks are
cheap enough to run without vLLM, a brain server or the robot.

Run: PYTHONPATH= .venv_tts/bin/python client/test_error_path.py
"""
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from client.local_live import Session  # noqa: E402


class FakeWs:
    """Just enough websocket: an async-iterable of frames the brain would have sent."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent: list = []

    def __aiter__(self):
        async def gen():
            for f in self._frames:
                yield f
        return gen()

    async def send(self, data):
        self.sent.append(data)


async def turn_ends_on_error() -> list[str]:
    """A brain error must end the turn, not strand the client."""
    failures = []
    ws = FakeWs([json.dumps({"t": "error", "detail": "turn failed"})])
    session = Session(ws)

    msgs = []
    try:
        # The bug was an infinite wait, so the assertion is the timeout itself.
        async def drain():
            async for m in session.receive():
                msgs.append(m)
        await asyncio.wait_for(drain(), timeout=5.0)
    except asyncio.TimeoutError:
        return ["receive()가 끝나지 않았다 — 로봇이 여기서 영원히 멈춘다"]

    if not msgs:
        failures.append("에러가 아무 메시지도 만들지 않았다 — launcher는 턴 종료를 못 본다")
    elif not msgs[-1].server_content.turn_complete:
        failures.append("에러 메시지에 turn_complete가 없다")
    if session.last_error != "turn failed":
        failures.append(f"last_error가 비었다: {session.last_error!r}")
    await session.close()
    return failures


def every_wire_message_is_handled() -> list[str]:
    """The general form of the bug: the brain gains a message the client ignores.

    A kind the shim never matches is silently dropped, and if it was the one meant to
    end the turn, the robot hangs exactly as above.
    """
    brain = " ".join((ROOT / "brain" / f).read_text()
                     for f in ("server.py", "pipeline.py"))
    shim = (ROOT / "client" / "local_live.py").read_text()

    sends = set(re.findall(r'send\(t="([a-z_]+)"', brain))
    handled = set(re.findall(r'kind == "([a-z_]+)"', shim))
    missing = sorted(sends - handled)

    print(f"  뇌가 보내는 종류 {len(sends)}개, shim이 처리 {len(handled)}개")
    if missing:
        return [f"shim이 처리하지 않는 메시지: {', '.join(missing)}"]
    return []


async def main() -> None:
    print("=" * 62)
    print("뇌가 턴에 실패했을 때 로봇이 빠져나오는가")
    print("=" * 62)
    failures = await turn_ends_on_error()
    print("  턴 종료" if not failures else "  멈춤")

    print()
    print("=" * 62)
    print("뇌가 보내는 모든 메시지를 shim이 처리하는가")
    print("=" * 62)
    failures += every_wire_message_is_handled()

    print()
    if failures:
        print("실패:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("에러 경로 정상 — 실패한 턴이 턴을 끝낸다")


if __name__ == "__main__":
    asyncio.run(main())
