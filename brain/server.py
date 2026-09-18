"""The brain's transport: one WebSocket per robot (§11.5).

Wire, exactly as the existing robot client already consumes it:

  robot -> brain   binary          raw PCM16 @16kHz, streamed continuously
                   {"t":"hello"}   system prompt + tool schemas for this session
                   {"t":"text"}    an injected turn (the greeting trigger)
                   {"t":"tool_result"}
  brain -> robot   binary          reply PCM (rate announced in the preceding audio frame)
                   {"t":"transcript"} {"t":"tool_call"} {"t":"interrupted"}
                   {"t":"turn_complete"}

Turn boundaries are decided here, not by the robot: `launcher.py` streams the mic
unconditionally because Gemini did server-side VAD, and we keep that contract (§4).

Run: PYTHONPATH= .venv_tts/bin/python brain/server.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.pipeline import Session, Tts, Turn  # noqa: E402
from brain.vad import TurnDetector  # noqa: E402

HOST, PORT = "0.0.0.0", 8765
TOOL_RESULT_TIMEOUT = 10.0

log = logging.getLogger("brain")


class Connection:
    """One robot. Owns its turn detector and conversation state."""

    def __init__(self, ws, tts: Tts):
        self.ws = ws
        self.tts = tts
        self.detector = TurnDetector()
        self.session: Session | None = None
        self.turns: asyncio.Queue = asyncio.Queue()
        self.current: Turn | None = None
        self.tool_results: asyncio.Future | None = None

    # ---- outbound -------------------------------------------------------
    async def send(self, **payload) -> None:
        await self.ws.send(json.dumps(payload, ensure_ascii=False))

    async def emit(self, kind: str, payload: dict) -> None:
        if kind == "audio":
            # Rate travels in a text frame ahead of the bytes: Piper is 22,050Hz while
            # the robot's playback path was built for Gemini's 24kHz (§8.3), so the
            # client must be told, not assume.
            await self.send(t="audio", rate=payload["rate"], bytes=len(payload["pcm"]))
            await self.ws.send(payload["pcm"])
        elif kind == "transcript":
            await self.send(t="transcript", role=payload["role"], text=payload["text"])
        elif kind == "tool_call":
            await self.send(t="tool_call", calls=payload["calls"])
        elif kind == "error":
            await self.send(t="error", detail=payload["detail"])

    # ---- turn processing ------------------------------------------------
    async def run_turn(self, pcm: bytes | None = None, text: str | None = None) -> None:
        assert self.session is not None
        turn = Turn(self.session, self.tts, pcm or b"")
        if text is not None:
            # An injected turn (the robot asking Moti to greet first) carries no audio.
            turn.parts = [{"type": "text", "text": text}]
        self.current = turn

        calls: list = []

        async def emit(kind: str, payload: dict) -> None:
            if kind == "tool_call":
                calls.extend(payload["calls"])
            await self.emit(kind, payload)

        try:
            await turn.run(emit)
            if calls and not turn.cancelled:
                await self.collect_tool_results(turn, calls)
        finally:
            self.current = None
            if not turn.cancelled:
                await self.send(t="turn_complete")

    async def collect_tool_results(self, turn: Turn, calls: list) -> None:
        """Wait for the robot to run the tools, then let the model speak again only if
        it has not spoken yet.

        Gemini always round-trips tool results. Doing that unconditionally would cost a
        second generation on every tool turn, and MOTI's tools are mostly fire-and-forget
        (set_emotion, play_gesture, remember_fact) — the model already says its piece in
        the same response. The case that actually breaks is when the model returns *only*
        a tool call: then the user hears nothing at all. That is the one we generate for.
        """
        self.tool_results = asyncio.get_running_loop().create_future()
        try:
            results = await asyncio.wait_for(self.tool_results, TOOL_RESULT_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning("tool results timed out after %.0fs", TOOL_RESULT_TIMEOUT)
            return
        finally:
            self.tool_results = None

        if turn.spoke or turn.cancelled:
            return
        summary = ", ".join(f"{r.get('name')}={r.get('result')}" for r in results)
        follow = Turn(self.session, self.tts, b"")
        follow.parts = [{"type": "text",
                         "text": f"(도구 실행 결과: {summary}) 이어서 사용자에게 한두 문장으로 말해줘."}]
        self.current = follow
        await follow.run(self.emit)

    # ---- inbound --------------------------------------------------------
    async def on_binary(self, pcm: bytes) -> None:
        if self.session is None:
            return
        for audio in self.detector.feed(pcm):
            await self.turns.put(audio)

    async def on_text(self, raw: str) -> None:
        msg = json.loads(raw)
        kind = msg.get("t")
        if kind == "hello":
            self.session = Session(system=msg["system"], tools=msg.get("tools") or [])
            await self.send(t="ready")
            log.info("session opened, %d tools", len(self.session.tools))
        elif kind == "text":
            await self.turns.put(msg["text"])
        elif kind == "tool_result":
            if self.tool_results and not self.tool_results.done():
                self.tool_results.set_result(msg.get("results") or [])
        else:
            log.warning("unknown message %r", kind)

    async def worker(self) -> None:
        while True:
            item = await self.turns.get()
            try:
                if isinstance(item, bytes):
                    await self.run_turn(pcm=item)
                else:
                    await self.run_turn(text=item)
            except Exception:                        # noqa: BLE001 — one bad turn must
                log.exception("turn failed")         # not kill the session
                await self.send(t="error", detail="turn failed")


async def handle(ws, tts: Tts) -> None:
    conn = Connection(ws, tts)
    worker = asyncio.create_task(conn.worker())
    log.info("robot connected: %s", ws.remote_address)
    try:
        async for message in ws:
            if isinstance(message, bytes):
                await conn.on_binary(message)
            else:
                await conn.on_text(message)
    except websockets.ConnectionClosed:
        pass
    finally:
        worker.cancel()
        log.info("robot disconnected")


async def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    tts = Tts()                     # load once; per-turn loading would dominate latency
    log.info("Piper ready (%dHz). listening on ws://%s:%d", tts.rate, HOST, PORT)
    async with websockets.serve(lambda ws: handle(ws, tts), HOST, PORT,
                                max_size=None):      # audio frames are large
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
