"""Moti brain — the cascade itself: audio in, speech out. No transport here.

Implements the §11 contract's inner half:
  audio (<=30s clips, §11.0-5) -> E4B via vLLM -> sentence chunks (§11.0-2) -> Piper -> PCM

What this module deliberately does NOT do yet, so it does not get forgotten:
  §11.0-1 playback buffer + interrupt flush   -> robot side, plus `Turn.cancel()` here
  §11.0-3 VAD-lag compensation ring buffer    -> belongs with VAD, not yet written
  VAD / smart-turn-v3 turn detection          -> next slice; for now the caller marks turn ends
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "google/gemma-4-E4B-it"
IN_RATE = 16000          # robot mic -> brain
CLIP_LIMIT_SEC = 30      # §11.0-5, hard ceiling; exceeding it truncates silently (§12.3)
CLIP_LIMIT_BYTES = CLIP_LIMIT_SEC * IN_RATE * 2

VOICE_PATH = Path(__file__).resolve().parent.parent / "tts_eval" / "ko_KR-kss-medium.onnx"

# Korean needs sentence-final endings as well as punctuation (§11.0-2): splitting on
# ".?!" alone leaves whole paragraphs intact, because Korean often runs clauses together
# without them. Endings are matched only when followed by space or end-of-text so that
# "그래요?" or a mid-word 다 does not split.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[다까요죠])(?=\s)")


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()]


def take_sentences(buf: str) -> tuple[list[str], str]:
    """Peel off completed sentences, returning them and the untouched remainder.

    The remainder must not be rebuilt by re-joining split pieces: doing that strips the
    trailing space, so the next streamed delta fuses onto the previous word and TTS
    pronounces "정도로지치셨다니" as one word. Slice, never rejoin.
    """
    done, last = [], 0
    for m in _SENTENCE_END.finditer(buf):
        piece = buf[last:m.end()].strip()
        if piece:
            done.append(piece)
        last = m.end()
    return done, buf[last:]


def pcm_to_wav(pcm: bytes, rate: int = IN_RATE) -> bytes:
    """§11.1: the wire carries raw PCM; WAV wrapping happens only here, at the
    vLLM boundary, because `input_audio` requires a container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def audio_parts(pcm: bytes) -> list[dict]:
    """Split into <=30s clips. §11.0-5: a single longer clip is silently truncated."""
    parts = []
    for i in range(0, len(pcm), CLIP_LIMIT_BYTES):
        wav = pcm_to_wav(pcm[i:i + CLIP_LIMIT_BYTES])
        parts.append({"type": "input_audio",
                      "input_audio": {"data": base64.b64encode(wav).decode(),
                                      "format": "wav"}})
    return parts


class Tts:
    """Piper. Loaded once — model load is seconds, per-utterance load would dominate."""

    def __init__(self, voice_path: Path = VOICE_PATH):
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(voice_path))
        self.rate = self.voice.config.sample_rate

    def synth(self, text: str) -> bytes:
        return b"".join(c.audio_int16_bytes for c in self.voice.synthesize(text))


@dataclass
class Session:
    """Conversation state. Lives on the brain so a dropped link does not lose it
    (§11.5 reconnection)."""
    system: str
    tools: list = field(default_factory=list)
    history: list = field(default_factory=list)
    # The persona alone is 18,344 tokens of a 32,768 window (§13.1), leaving ~14k for
    # history plus the current turn's audio. Unbounded history overflows it after roughly
    # 118 turns and the request simply fails — and this server is always-on and resumes
    # sessions across reconnects, so it accumulates. Trimming old turns is safe here
    # precisely because long-term memory is not in this list: it lives in the user's
    # `facts`, which the robot re-injects through the persona.
    max_history: int = 40           # 20 exchanges

    def remember(self, user: str, assistant: str) -> None:
        self.history += [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]
        if len(self.history) > self.max_history:
            del self.history[:len(self.history) - self.max_history]

    def _messages(self, parts: list) -> list:
        return [{"role": "system", "content": self.system}, *self.history,
                {"role": "user", "content": parts}]

    def _post(self, parts: list, *, stream: bool, max_tokens: int,
              with_tools: bool) -> urllib.request.Request:
        body = {"model": MODEL, "messages": self._messages(parts),
                "temperature": 0.7, "max_tokens": max_tokens, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        if with_tools and self.tools:
            body["tools"] = self.tools
            body["tool_choice"] = "auto"
        return urllib.request.Request(
            VLLM_URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})

    def transcribe(self, parts: list) -> str:
        """User-side transcript (§12.6 gap 1). Audio parts come first so this call
        shares the [system][audio] prefix with the reply call and does not pay the
        audio prefill twice."""
        req = self._post(parts + [{"type": "text",
                                   "text": "이 오디오의 발화 내용만 그대로 받아적어. 설명하지 마."}],
                         stream=False, max_tokens=256, with_tools=False)
        with urllib.request.urlopen(req) as r:
            return (json.load(r)["choices"][0]["message"].get("content") or "").strip()

    def reply_stream(self, parts: list):
        """Yields ('text', str) deltas and ('tool_call', list) as they arrive."""
        req = self._post(parts + [{"type": "text", "text": "방금 한 말에 반응해줘."}],
                         stream=True, max_tokens=256, with_tools=True)
        pending: dict[int, dict] = {}
        with urllib.request.urlopen(req) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                choices = json.loads(payload).get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    yield "text", delta["content"]
                for tc in delta.get("tool_calls") or []:
                    slot = pending.setdefault(tc["index"],
                                              {"id": "", "name": "", "arguments": ""})
                    slot["id"] += tc.get("id") or ""
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
        if pending:
            yield "tool_call", [pending[k] for k in sorted(pending)]


class Turn:
    """One user turn. Cancellable as a unit (§11.0-4): cancelling must stop both the
    LLM stream and any TTS work, and leave nothing that affects the next turn."""

    def __init__(self, session: Session, tts: Tts, pcm: bytes):
        self.session, self.tts, self.parts = session, tts, audio_parts(pcm)
        self.cancelled = False
        self.marks: dict[str, float] = {}   # §20 rule 5: instrument from the first commit

    def cancel(self) -> None:
        self.cancelled = True

    async def run(self, emit) -> None:
        """emit(kind, payload) -> awaitable. kinds: transcript, audio, tool_call, done."""
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        # The user transcript is for the conversation log, not for the reply — blocking on
        # it before generating cost 2.4s of pure silence in the first measurement. Start it
        # alongside the reply and collect it before `done`, which is when launcher.py needs
        # it (it builds turn_user, then reads it at turn_complete).
        transcript_task = loop.run_in_executor(None, self.session.transcribe, self.parts)

        queue: asyncio.Queue = asyncio.Queue()

        def pump() -> None:
            try:
                for kind, value in self.session.reply_stream(self.parts):
                    if self.cancelled:
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, (kind, value))
            except Exception as exc:                      # noqa: BLE001 — surface, don't die
                loop.call_soon_threadsafe(queue.put_nowait, ("error", repr(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, (None, None))

        loop.run_in_executor(None, pump)

        buf, spoken, calls = "", [], []
        while True:
            kind, value = await queue.get()
            if kind is None or self.cancelled:
                break
            if kind == "error":
                await emit("error", {"detail": value})
                break
            if kind == "tool_call":
                calls = value
                continue
            buf += value
            if "first_token" not in self.marks:
                self.marks["first_token"] = time.perf_counter() - t0
            # Synthesize each completed sentence immediately (§11.0-2) — waiting for the
            # whole reply is what collapses first-audio latency.
            done, buf = take_sentences(buf)
            for sentence in done:
                await self._speak(sentence, emit, t0)
                spoken.append(sentence)

        if buf.strip() and not self.cancelled:
            tail = buf.strip()
            await self._speak(tail, emit, t0)
            spoken.append(tail)
            buf = ""

        full = " ".join(spoken).strip()
        if calls and not self.cancelled:
            await emit("tool_call", {"calls": calls})

        user_text = await transcript_task
        self.marks["transcript"] = time.perf_counter() - t0
        if self.cancelled:
            return
        await emit("transcript", {"role": "user", "text": user_text})
        self.session.remember(user_text, full)
        await emit("done", {"text": full, "marks": self.marks})

    async def _speak(self, sentence: str, emit, t0: float) -> None:
        if self.cancelled or not sentence:
            return
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(None, self.tts.synth, sentence)
        if self.cancelled:
            return
        if "first_audio" not in self.marks:
            self.marks["first_audio"] = time.perf_counter() - t0
        await emit("transcript", {"role": "model", "text": sentence})
        await emit("audio", {"pcm": pcm, "rate": self.tts.rate})
