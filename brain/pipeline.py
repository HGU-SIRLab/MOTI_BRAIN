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
import logging
import re
import time
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("brain.pipeline")

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
#  `{emotion: "sad"}` and `(emotion="tender")` both reach JSON as `{"emotion": ...}`.
_BARE_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*[:=]")

# §9.3 proactive audio: the persona tells the model to answer with exactly this when the
# user was not talking to it. The brain must drop it rather than speak it — the token is
# a control signal, and TTS would happily pronounce it. The wording that makes the model
# emit it reliably lives on the robot side (its persona); see brain/test_gates.py for the
# measurement that settled it.
SILENT_TOKEN = "<SILENT>"

# Control tokens the robot's persona tells the model to emit — `[대화종료]` ends the
# session, and there may be others we never hear about. They have to reach the robot
# (`launcher.py` scans `output_transcription` for them, and that is its *only* channel),
# but they must not be spoken: measured 2026-09-22, Piper renders `[대화종료]` as 1.23s of
# perfectly clear speech, so Moti announced "대화종료" out loud before hanging up.
#
# Matched by shape rather than by a list, so the brain does not have to know the robot's
# vocabulary: a bracketed run with no spaces. Ordinary empathetic Korean does not contain
# `[한단어]`, and the cost of being wrong is one unspoken bracketed word.
_CONTROL_TOKEN = re.compile(r"[\[<][^\[\]<>\s]+[\]>]")


def _parse_call(body: str, first: dict | None = None) -> dict | None:
    """`call:set_emotion{emotion:<|"|>sad<|"|>}` -> {id, name, args}.

    `args` is a dict, not a JSON string, and the field is named `args` — both because
    §11.5 says so and because `launcher.py` splats it: `fn(**(fc.args or {}))`.
    """
    body = body.strip()
    if not body.startswith("call:"):
        return None
    body = body[5:]
    opens = [body.find(o) for o in _OPENERS if body.find(o) != -1]
    cut = min(opens) if opens else -1
    call = {"id": uuid.uuid4().hex[:8],
            "name": (body[:cut] if cut != -1 else body).strip(), "args": {}}
    if cut == -1 or not call["name"]:
        # A nameless call is markup noise, not an instruction — `<|tool_call>call:<tool_call|>`
        # shows up empty when the payload was already consumed as a bare call. Emitting it
        # would have the robot look up a tool named "".
        return call if call["name"] else None
    raw = "{" + body[cut + 1:].rsplit(_OPENERS[body[cut]], 1)[0] + "}"
    raw = raw.replace('<|"|>', '"')
    raw = _BARE_KEY.sub(r'\1"\2":', raw)            # {emotion:"sad"} is not valid JSON
    try:
        call["args"] = json.loads(raw)
    except json.JSONDecodeError:
        # Positional form: `set_emotion(tender)` has a value but no key. The declaration
        # supplies the key; without it the argument is lost silently.
        inner = raw[1:-1].strip().strip('"\'')
        key = (first or {}).get(call["name"])
        if key and inner and not any(c in inner for c in ':=,'):
            call["args"] = {key: inner}
    return call


_OPENERS = {"{": "}", "(": ")"}                      # see _find_bare_call


def _find_bare_call(buf: str, names: tuple[str, ...],
                    first: dict | None = None) -> tuple[int, int, dict] | None:
    """Locate an unwrapped `set_emotion{...}` or `set_emotion(...)` call.

    Only declared tool names count — matching any `word{...}` would swallow ordinary
    text. The parenthesised shape was added 2026-09-19: under the real 18K persona the
    model produces `set_emotion(emotion="tender")` as often as the brace form, and with
    only braces handled it went straight to TTS and the robot read it aloud. That is the
    third distinct spelling of this bug; hence the table rather than another branch.
    """
    best = None
    for name in names:
        for opener, closer in _OPENERS.items():
            i = buf.find(name + opener)
            if i == -1 or (best is not None and i >= best[0]):
                continue
            j = buf.find(closer, i)
            if j == -1:
                best = (i, -1, {})                   # opened, not yet closed: hold it
                continue
            call = _parse_call("call:" + buf[i:j + 1], first)
            if call:
                best = (i, j + 1, call)
    return best


def _holdback(text: str, names: tuple[str, ...]) -> int:
    """Index from which `text` might still be growing into a tool call, or -1.

    Longest prefix wins. Checking short-to-first-match (as this did until 2026-09-19)
    holds back one character of `remember_fact` because the text happens to end in "r",
    releases `remembe` to TTS, and then the held "r" never grows into the marker — so the
    whole call leaks out a character at a time. Only reachable when a delta ends exactly
    on a short prefix, which is why it survived every test with the toy persona.
    """
    cut = text.find(_TOOL_OPEN)
    if cut != -1:
        return cut
    best = -1
    markers = (_TOOL_OPEN, *(n + o for n in names for o in _OPENERS))
    for marker in markers:
        for i in range(min(len(marker) - 1, len(text)), 0, -1):
            if text.endswith(marker[:i]):
                here = len(text) - i
                best = here if best == -1 else min(best, here)
                break
    return best


def strip_tool_calls(buf: str, names: tuple[str, ...] = (),
                     first: dict | None = None) -> tuple[str, list[dict], str]:
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
        call = _parse_call(m.group(1), first)
        if call:
            calls.append(call)
        return ""

    text = _TOOL_REGION.sub(take, buf)

    while names:
        hit = _find_bare_call(text, names, first)
        if hit is None:
            break
        i, j, call = hit
        if j == -1:
            # Unterminated — hold from here, but the prefix may itself be the opening of
            # a wrapped call. Returning `text[:i]` directly (as this did until
            # 2026-09-19) emitted a live `<|tool_call>call:` as speakable text, and the
            # robot read the markup aloud. Only showed up under the real persona, where
            # the model wraps *and* uses the bare form in the same reply.
            cut = _holdback(text[:i], names)
            cut = i if cut == -1 else cut
            return text[:cut], calls, text[cut:]
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
    """Piper, resampled to what the robot's player actually opens.

    `MOTI-HRI/media/audio_manager.py` hardcodes `OUTPUT_RATE = 24000` and opens the
    sounddevice stream at it; the AEC reference path is derived from the same figure.
    Piper's Korean voice is 22,050Hz. Converting here keeps the robot untouched
    (§20 rule 17) — sending 22,050 and hoping would play everything ~9% fast and low.

    160/147 is exact (gcd(24000, 22050) = 150), so polyphase resampling is clean rather
    than an approximation, and costs ~3ms per 3s of audio.
    """

    OUT_RATE = 24000

    def __init__(self, voice_path: Path = VOICE_PATH):
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(voice_path))
        self.native_rate = self.voice.config.sample_rate
        self.rate = self.OUT_RATE

    def synth(self, text: str) -> bytes:
        raw = b"".join(c.audio_int16_bytes for c in self.voice.synthesize(text))
        return self._to_out_rate(raw)

    def _to_out_rate(self, pcm: bytes) -> bytes:
        if self.native_rate == self.OUT_RATE or not pcm:
            return pcm
        import numpy as np
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(self.OUT_RATE, self.native_rate)
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        y = resample_poly(x, self.OUT_RATE // g, self.native_rate // g)
        # resample_poly can overshoot at transients; clip before narrowing to int16.
        return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


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
              with_tools: bool, system: str | None = None,
              temperature: float = 0.7) -> urllib.request.Request:
        body = {"model": MODEL, "messages": self._messages(parts, system),
                "temperature": temperature, "max_tokens": max_tokens, "stream": stream}
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
                         # Greedy. 0.7 was applied to every call including this one, which
                         # is sampling noise added to a task that has one right answer —
                         # and EXP-13 measured "character-for-character exact" at
                         # temperature 0 (§12.4), so production was never running what was
                         # validated. Found after the robot transcribed 조형민 as 조효형민
                         # and stored the wrong name via remember_fact (2026-09-21).
                         temperature=0.0,
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
                blob = json.loads(payload)
                if usage := blob.get("usage"):
                    # `include_usage`로 이미 받고 있었는데 버리고 있었다. 모니터가
                    # 토큰/초와 프리필 크기를 보여주려면 이게 필요하다.
                    yield "usage", usage
                choices = blob.get("choices") or []
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
        # The model also writes calls positionally — `set_emotion(tender)` — and there is
        # no key to parse out of that. The declaration has one, so map the lone value onto
        # the first required parameter. Without this the call arrives with `args: {}`,
        # `launcher.py` runs `set_emotion()`, its try/except swallows the TypeError, and
        # the robot's face simply never changes. Silent, and only visible on video.
        # A call that arrives without its required arguments is not actionable: the robot
        # would run `set_emotion()` and its try/except would swallow the TypeError, so the
        # face silently stays put. Measured 2026-09-22 on the real robot — the model does
        # this intermittently (1 turn in 3) even with a correct schema and enum. Dropping
        # it here turns an invisible robot-side failure into a visible brain-side log line.
        self.tool_required = {
            t["function"]["name"]: list(t["function"].get("parameters", {})
                                        .get("required") or [])
            for t in session.tools if t.get("function", {}).get("name")}
        self.tool_first_param = {
            t["function"]["name"]: (t["function"].get("parameters", {})
                                    .get("required") or [None])[0]
            for t in session.tools if t.get("function", {}).get("name")}
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

        # 🔴 An injected turn has no user speech in it, so there is nothing to transcribe.
        # `launcher.py:307 inject_turn()` sends the robot's own text as `role="user"` —
        # that is the only way to make Moti speak first against the Live API — and this
        # code fed those text parts to `transcribe()`, whose prompt is "이 오디오의 발화
        # 내용만 그대로 받아적어". Handed text instead of audio, the model politely echoed
        # it, and the brain sent the robot's own privacy notice back as
        # `input_transcription`. `launcher.py` logged it as the user speaking and wrote it
        # into `user_result/*/대화.txt` — research data, corrupted (robot report,
        # 2026-09-22; reproduced here exactly, down to the "(진행자 지시: …)" prefix being
        # dropped because the model read it as meta).
        #
        # The transcription leg was written assuming audio always arrives. It does not.
        injected = next((p["text"] for p in self.parts if p.get("type") == "text"
                         and not any(q.get("type") == "input_audio" for q in self.parts)),
                        None)
        # The user transcript is for the conversation log, not for the reply — blocking on
        # it before generating cost 2.4s of pure silence in the first measurement. Start it
        # alongside the reply and collect it before `done`, which is when launcher.py needs
        # it (it builds turn_user, then reads it at turn_complete).
        transcript_task = (None if injected is not None else
                           loop.run_in_executor(None, self.session.transcribe, self.parts))

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
            if kind == "usage":
                self.marks["prompt_tokens"] = value.get("prompt_tokens", 0)
                self.marks["completion_tokens"] = value.get("completion_tokens", 0)
                continue
            if kind == "tool_call":
                # Never fires on vLLM 0.19.0 (delta.tool_calls is always empty) but must
                # accumulate, not replace: a future vLLM that streams tool calls properly
                # would otherwise silently discard whatever strip_tool_calls already found.
                calls += value
                continue
            raw += value
            # Pull tool markup out before anything can reach TTS; `raw` keeps only the
            # part that might still turn out to be a tool call.
            clean, found, raw = strip_tool_calls(raw, self.tool_names, self.tool_first_param)
            for c in found:
                missing = [a for a in self.tool_required.get(c["name"], [])
                           if a not in (c["args"] or {})]
                if missing:
                    log.warning("dropped %s — model gave no %s",
                                c["name"], ", ".join(missing))
                    continue
                calls.append(c)
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

        if transcript_task is None:
            # Injected turn: history still needs the prompt Moti was answering, but the
            # robot must not be told that the user said it.
            user_text = injected
        else:
            user_text = await transcript_task
            self.marks["transcript"] = time.perf_counter() - t0
        if self.cancelled:
            return
        if transcript_task is not None:
            await emit("transcript", {"role": "user", "text": user_text})
        self.session.remember(user_text, full)
        await emit("done", {"text": full, "marks": self.marks})

    async def _speak(self, sentence: str, emit, t0: float) -> None:
        if self.cancelled or not sentence:
            return
        # The transcript carries the sentence as written; only the audio drops control
        # tokens (see `_CONTROL_TOKEN`). The robot needs to *read* `[대화종료]` and must
        # not *hear* it.
        say = _CONTROL_TOKEN.sub("", sentence).strip()
        if not say:
            # Nothing left to voice — still tell the robot what was said, or the tag is
            # lost and the session never ends.
            await emit("transcript", {"role": "model", "text": sentence + " "})
            return
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(None, self.tts.synth, say)
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
