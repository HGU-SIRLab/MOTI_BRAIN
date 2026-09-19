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
import uuid
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
# The third alternative matters: the model routinely omits the space after a period
# ("마음이 아프네요.푹 쉬시면서"). Without it nothing splits at all, the whole reply becomes
# one TTS chunk, and §11.0-2's whole point — first audio before the reply finishes — is lost.
# Digits are excluded so "3.5초" stays intact.
_SENTENCE_END = re.compile(
    r"(?<=[.!?])\s+|(?<=[다까요죠])(?=\s)|(?<=[.!?])(?=[^\s\d])")


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


# vLLM 0.19.0's `gemma4` tool-call parser does not handle streaming: `delta.tool_calls`
# stays empty and the raw markup arrives inside `delta.content` instead, like
#   <|tool_call>call:set_emotion{emotion:<|"|>sad<|"|>}<tool_call|>
# Left alone it reaches TTS and the robot says "tool call set emotion sad" out loud.
# Non-streaming parses correctly but would cost the whole reply's latency (§13.2), so we
# strip and parse it ourselves. Stripping is required regardless; parsing rides along free.
_TOOL_OPEN, _TOOL_CLOSE = "<|tool_call>", "<tool_call|>"
_TOOL_REGION = re.compile(re.escape(_TOOL_OPEN) + r"(.*?)" + re.escape(_TOOL_CLOSE), re.S)
_BARE_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")

# §9.3 proactive audio: the persona tells the model to answer with exactly this when the
# user was not talking to it. The brain must drop it rather than speak it — the token is
# a control signal, and TTS would happily pronounce it. The wording that makes the model
# emit it reliably lives on the robot side (its persona); see brain/test_gates.py for the
# measurement that settled it.
SILENT_TOKEN = "<SILENT>"


def _parse_call(body: str) -> dict | None:
    """`call:set_emotion{emotion:<|"|>sad<|"|>}` -> {id, name, args}.

    `args` is a dict, not a JSON string, and the field is named `args` — both because
    §11.5 says so and because `launcher.py` splats it: `fn(**(fc.args or {}))`.
    """
    body = body.strip()
    if not body.startswith("call:"):
        return None
    head, brace, rest = body[5:].partition("{")
    call = {"id": uuid.uuid4().hex[:8], "name": head.strip(), "args": {}}
    if not brace:
        return call
    raw = "{" + rest.rsplit("}", 1)[0] + "}"
    raw = raw.replace('<|"|>', '"')
    raw = _BARE_KEY.sub(r'\1"\2":', raw)            # {emotion:"sad"} is not valid JSON
    try:
        call["args"] = json.loads(raw)
    except json.JSONDecodeError:
        pass                                         # never crash the turn over markup
    return call


def _find_bare_call(buf: str, names: tuple[str, ...]) -> tuple[int, int, dict] | None:
    """Locate an unwrapped `set_emotion{...}` call. Only declared tool names count —
    matching any `word{...}` would swallow ordinary text."""
    for name in names:
        i = buf.find(name + "{")
        if i == -1:
            continue
        j = buf.find("}", i)
        if j == -1:
            return (i, -1, {})                       # opened, not yet closed: hold it
        call = _parse_call("call:" + buf[i:j + 1])
        if call:
            return (i, j + 1, call)
    return None


def _holdback(text: str, names: tuple[str, ...]) -> int:
    """Index from which `text` might still be growing into a tool call, or -1."""
    cut = text.find(_TOOL_OPEN)
    if cut != -1:
        return cut
    for marker in (_TOOL_OPEN, *(n + "{" for n in names)):
        for i in range(1, len(marker)):
            if text.endswith(marker[:i]):
                return len(text) - i
    return -1


def strip_tool_calls(buf: str, names: tuple[str, ...] = ()) -> tuple[str, list[dict], str]:
    """Split a streamed buffer into (speakable text, calls, tail to hold back).

    The model emits tool calls in two shapes, both as plain `content` (vLLM 0.19.0 never
    populates `delta.tool_calls`). Measured over repeated runs:
        set_emotion{emotion:<|"|>sad<|"|>} 며칠 내내...        <- usual
        <|tool_call>call:set_emotion{...}<tool_call|>...       <- also seen
    Handling only the wrapped one let the bare form reach TTS, and the robot read the
    markup aloud. The tail is whatever might still be growing into either shape.
    """
    calls: list[dict] = []

    def take(m: re.Match) -> str:
        call = _parse_call(m.group(1))
        if call:
            calls.append(call)
        return ""

    text = _TOOL_REGION.sub(take, buf)

    while names:
        hit = _find_bare_call(text, names)
        if hit is None:
            break
        i, j, call = hit
        if j == -1:                                  # unterminated — hold from here
            return text[:i], calls, text[i:]
        calls.append(call)
        text = text[:i] + text[j:]

    cut = _holdback(text, names)
    if cut != -1:
        return text[:cut], calls, text[cut:]
    return text, calls, ""


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

    def _messages(self, parts: list, system: str | None = None) -> list:
        # A transcription call passes its own system prompt and skips history: it must
        # not inherit persona instructions, and prior turns are irrelevant to it.
        if system is not None:
            return [{"role": "system", "content": system},
                    {"role": "user", "content": parts}]
        return [{"role": "system", "content": self.system}, *self.history,
                {"role": "user", "content": parts}]

    def _post(self, parts: list, *, stream: bool, max_tokens: int,
              with_tools: bool, system: str | None = None) -> urllib.request.Request:
        body = {"model": MODEL, "messages": self._messages(parts, system),
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
        """User-side transcript (§12.6 gap 1).

        Uses a bare transcription system prompt, NOT the persona. With the persona the
        model obeys instructions like "always call set_emotion first" even here, and the
        transcript comes back as "set_emotion(tired)\n며칠 내내..." — polluting the
        conversation log with tool syntax the user never said.
        """
        req = self._post(parts + [{"type": "text",
                                   "text": "이 오디오의 발화 내용만 그대로 받아적어. 설명하지 마."}],
                         stream=False, max_tokens=256, with_tools=False,
                         system="오디오를 듣고 발화 내용을 그대로 받아적는 전사기다. 다른 말은 하지 않는다.")
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
        self.tool_names = tuple(
            t.get("function", {}).get("name", "") for t in session.tools) or ()
        self.cancelled = False
        self.spoke = False                  # did the user actually hear anything?
        self.silent = False                 # model chose not to answer (§9.3)
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

        raw, buf, spoken, calls = "", "", [], []
        while True:
            kind, value = await queue.get()
            if kind is None or self.cancelled:
                break
            if kind == "error":
                await emit("error", {"detail": value})
                break
            if kind == "tool_call":
                # Never fires on vLLM 0.19.0 (delta.tool_calls is always empty) but must
                # accumulate, not replace: a future vLLM that streams tool calls properly
                # would otherwise silently discard whatever strip_tool_calls already found.
                calls += value
                continue
            raw += value
            # Pull tool markup out before anything can reach TTS; `raw` keeps only the
            # part that might still turn out to be a tool call.
            clean, found, raw = strip_tool_calls(raw, self.tool_names)
            calls += found
            if SILENT_TOKEN in clean:
                clean = clean.replace(SILENT_TOKEN, "")
                self.silent = True
            if not clean.strip():
                continue
            buf += clean
            if "first_token" not in self.marks:
                self.marks["first_token"] = time.perf_counter() - t0
            # Synthesize each completed sentence immediately (§11.0-2) — waiting for the
            # whole reply is what collapses first-audio latency.
            done, buf = take_sentences(buf)
            for sentence in done:
                await self._speak(sentence, emit, t0)
                spoken.append(sentence)

        # Anything still held back was suspected of being a tool call. An unterminated
        # region is correctly dropped, but a false-positive prefix match is real speech —
        # release it rather than silently losing the end of a sentence.
        if raw and not raw.startswith(_TOOL_OPEN) and not any(
                raw.startswith(n) for n in self.tool_names):
            buf += raw

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
        self.spoke = True
        # Trailing space matters: launcher.py concatenates output_transcription chunks
        # with "".join(), so without it the saved conversation log — and the 마음처방전
        # built from it — reads "힘드셨겠어요.눈이 감기실". Gemini's partial chunks carry
        # their own spacing; ours are whole sentences.
        await emit("transcript", {"role": "model", "text": sentence + " "})
        await emit("audio", {"pcm": pcm, "rate": self.tts.rate})
