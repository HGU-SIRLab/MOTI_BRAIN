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
from brain.backchannel import Backchannel  # noqa: E402
from brain.pipeline import Session, Tts, Turn  # noqa: E402
from brain.vad import TurnDetector  # noqa: E402

HOST, PORT = "0.0.0.0", 8765
TOOL_RESULT_TIMEOUT = 10.0
# Consecutive 32ms speech windows before we treat incoming audio as an interruption.
# One window lets a cough cut Moti off mid-sentence; waiting for a whole turn is far too
# late. Three is ~96ms, inside the ~200ms reaction §4.2 calls the cheapest source of
# perceived liveness.
BARGE_WINDOWS = 3
# EXP-12 (§4.2): how far into a pause to drop a short "응".
# Measured against the real pause distribution (§12.7): at 0.6s it fires *inside* six
# utterances across five clips — the robot talking over someone mid-sentence. Overlap
# only disappears at 1.4s, because the longest intra-utterance pause we have is 1.38s.
BACKCHANNEL_AFTER = 1.4
# Speculative generation (§9.1d): start replying into a buffer this far into a pause.
# It never makes the robot speak sooner — it only has the answer ready when the turn is
# confirmed, taking the brain's ~0.5s off what the user waits through. If the speaker
# resumes, the work is thrown away; cancellation already costs 70ms (§13.3).
# 0.2s, not 0.5s: the fast path closes the turn at 0.6s, so starting at 0.5s left only
# 0.1s of head start and 5 of 6 turns adopted "0 ready" — nothing had been produced yet.
SPECULATE_AFTER = 0.2
# Closing a turn is driven by incoming audio, so a client that stops sending leaves it
# open forever. launcher.py can do exactly that — its RMS gate suppresses silent audio
# while the robot is "asleep" to avoid paying Gemini for it. Wall-clock backstop.
STALL_TIMEOUT = 6.0

log = logging.getLogger("brain")

# §11.5 [MANDATORY]: the brain holds conversation state, so a dropped link — likely over
# Tailscale — does not wipe the conversation. Keyed by an id the client keeps across its
# own reconnects. Bounded because an always-on server would otherwise accumulate sessions
# forever; a lab robot needs a handful, and evicting the oldest loses the least.
SESSIONS: dict[str, Session] = {}
MAX_SESSIONS = 8


class Connection:
    """One robot. Owns its turn detector and conversation state."""

    def __init__(self, ws, tts: Tts, backchannel: Backchannel):
        self.ws = ws
        self.tts = tts
        self.backchannel = backchannel
        self._acked = False          # one backchannel per pause, not per audio chunk
        # Per-session switch rather than a server flag: EXP-12 is an A/B, and restarting
        # this server costs ~33 minutes (§9.1a note on model loading).
        self.backchannel_on = True
        self.backchannel_after = BACKCHANNEL_AFTER
        self.speculate = True
        self._spec: Turn | None = None      # reply being prepared during the pause
        self._spec_task: asyncio.Task | None = None
        self._spec_audio: bytes = b""       # the audio it was generated from
        self._spec_out: list[tuple[str, dict]] = []
        self._spec_live = False             # confirmed: stop buffering, stream instead
        self._last_audio = 0.0
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
            # client must be told, not assume. `kind` separates a real reply from an
            # EXP-12 acknowledgement — the robot should not log "응" as something Moti
            # said, and may want to duck it differently.
            await self.send(t="audio", rate=payload["rate"], bytes=len(payload["pcm"]),
                            kind=payload.get("kind", "reply"))
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
        self._last_audio = asyncio.get_running_loop().time()
        turns = self.detector.feed(pcm)

        if self.detector.speech_run:
            self._acked = False      # they are talking again; arm for the next pause
            self._drop_speculation()  # whatever we were preparing answers the wrong thing

        # Barge-in (§9.2, §11.0-1/-4): the user started talking while Moti was replying.
        # Cancel generation *and* synthesis, and tell the robot — it stops playback and
        # clears its queue, which is the half of the contract we cannot do from here.
        #
        # This assumes AEC on the robot actually works. Without it Moti's own voice comes
        # back through the mic and interrupts her mid-sentence; the robot repo logged
        # exactly that symptom before AEC was in place (§18).
        turn = self.current
        if turn is not None and not turn.cancelled \
                and self.detector.speech_run >= BARGE_WINDOWS:
            turn.cancel()
            await self.send(t="interrupted")
            log.info("barge-in: turn cancelled")

        # EXP-12: the user has stopped but the VAD is not yet sure. Rather than 1.5s of
        # dead air, acknowledge. This does not make the reply arrive sooner — it stops the
        # robot seeming deaf while it waits (§4.2, §13.4).
        if (self.backchannel_on and self.current is None and not self._acked
                and turn is None and self.detector.in_speech
                and self.detector.silence_secs >= self.backchannel_after):
            self._acked = True
            clip = self.backchannel.pick()
            await self.emit("audio", {"pcm": clip, "rate": self.backchannel.rate,
                                      "kind": "backchannel"})
            log.info("backchannel (%.0fms of silence)",
                     self.detector.silence_secs * 1000)

        # Start preparing a reply while the VAD is still deciding.
        if (self.speculate and self.session is not None and self.current is None
                and self._spec is None and not turns
                and self.detector.silence_secs >= SPECULATE_AFTER):
            provisional = self.detector.provisional()
            if provisional:
                self._begin_speculation(provisional)

        for audio in turns:
            await self.turns.put(audio)

    async def on_text(self, raw: str) -> None:
        msg = json.loads(raw)
        kind = msg.get("t")
        if kind == "hello":
            sid = msg.get("session_id") or ""
            self.backchannel_on = bool(msg.get("backchannel", True))
            self.speculate = bool(msg.get("speculate", True))
            self.backchannel_after = float(msg.get("backchannel_after",
                                                   BACKCHANNEL_AFTER))
            prior = SESSIONS.get(sid)
            if prior is not None:
                # Resume. Take the *new* persona: launcher rebuilds it on reconnect and
                # it may now contain the user's name, learned via remember_fact mid-session.
                prior.system = msg["system"]
                prior.tools = msg.get("tools") or []
                self.session = prior
            else:
                self.session = Session(system=msg["system"],
                                       tools=msg.get("tools") or [])
                if sid:
                    if len(SESSIONS) >= MAX_SESSIONS:
                        SESSIONS.pop(next(iter(SESSIONS)))
                    SESSIONS[sid] = self.session
            await self.send(t="ready", resumed=prior is not None,
                            turns=len(self.session.history) // 2)
            log.info("session %s: %s, %d tools, %d turns carried over",
                     sid[:8] or "(anon)", "resumed" if prior else "new",
                     len(self.session.tools), len(self.session.history) // 2)
        elif kind == "text":
            await self.turns.put(msg["text"])
        elif kind == "tool_result":
            if self.tool_results and not self.tool_results.done():
                self.tool_results.set_result(msg.get("results") or [])
        else:
            log.warning("unknown message %r", kind)

    # ---- speculative generation ----------------------------------------
    def _begin_speculation(self, audio: bytes) -> None:
        assert self.session is not None
        turn = Turn(self.session, self.tts, audio)
        # The buffer must be a captured local, not `self._spec_out`. Closing over the
        # attribute meant that reassigning it in _flush_speculation left the collector
        # appending into a fresh list, and the prepared reply arrived as 0 events.
        buffer: list[tuple[str, dict]] = []
        self._spec, self._spec_audio, self._spec_out = turn, audio, buffer
        self._spec_live = False

        async def collect(kind: str, payload: dict) -> None:
            # Buffer until the turn is confirmed, then switch to sending live. Waiting
            # for the whole turn before releasing anything made first-audio *worse* than
            # no speculation at all: the first sentence is normally out in ~1.4s, but
            # holding everything until the last sentence finished pushed it past 3.5s.
            if self._spec_live:
                await self.emit(kind, payload)
            else:
                buffer.append((kind, payload))

        self._spec_task = asyncio.create_task(turn.run(collect))
        log.info("speculating on %.1fs of audio", len(audio) / 2 / 16000)

    def _drop_speculation(self) -> None:
        if self._spec is not None:
            self._spec.cancel()
        if self._spec_task is not None:
            self._spec_task.cancel()
        self._spec = self._spec_task = None
        self._spec_audio = b""
        self._spec_out = []

    async def _flush_speculation(self) -> bool:
        """Adopt the in-flight reply: send what is ready, then let the rest stream."""
        task, turn, out = self._spec_task, self._spec, self._spec_out
        if task is None or turn is None:
            return False
        head = len(out)
        for kind, payload in out:
            await self.emit(kind, payload)
        self._spec_live = True               # collector now sends directly
        self.current = turn                  # so barge-in can still cancel it
        try:
            await task
        except asyncio.CancelledError:
            return False
        finally:
            self.current = None
            self._spec = self._spec_task = None
            self._spec_out = []
            self._spec_live = False
        if turn.cancelled:
            return False
        await self.send(t="turn_complete")
        log.info("speculation adopted (%d ready, rest streamed)", head)
        return True

    async def watchdog(self) -> None:
        """Close an open turn if the client goes quiet (see STALL_TIMEOUT)."""
        while True:
            await asyncio.sleep(1.0)
            if self.session is None or self.current is not None:
                continue
            idle = asyncio.get_running_loop().time() - self._last_audio
            if self._last_audio and idle >= STALL_TIMEOUT:
                pending = self.detector.flush()
                self._last_audio = 0.0
                if pending:
                    log.warning("client silent %.1fs — closing turn anyway", idle)
                    await self.turns.put(pending)

    async def worker(self) -> None:
        while True:
            item = await self.turns.get()
            try:
                if isinstance(item, bytes):
                    # Identical audio means the prepared reply is still the right one.
                    if item == self._spec_audio and await self._flush_speculation():
                        continue
                    self._drop_speculation()
                    await self.run_turn(pcm=item)
                else:
                    self._drop_speculation()
                    await self.run_turn(text=item)
            except Exception:                        # noqa: BLE001 — one bad turn must
                log.exception("turn failed")         # not kill the session
                await self.send(t="error", detail="turn failed")


async def handle(ws, tts: Tts, backchannel: Backchannel) -> None:
    conn = Connection(ws, tts, backchannel)
    worker = asyncio.create_task(conn.worker())
    watchdog = asyncio.create_task(conn.watchdog())
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
        watchdog.cancel()
        log.info("robot disconnected")


async def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    tts = Tts()                     # load once; per-turn loading would dominate latency
    backchannel = Backchannel(tts)  # pre-synthesized; a live call would defeat the point
    log.info("Piper ready (%dHz), %d backchannel clips. listening on ws://%s:%d",
             tts.rate, len(backchannel.clips), HOST, PORT)
    async with websockets.serve(lambda ws: handle(ws, tts, backchannel), HOST, PORT,
                                max_size=None):      # audio frames are large
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
