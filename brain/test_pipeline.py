"""End-to-end check for the cascade: a real recording in, speech out.

Writes the synthesized reply so it can actually be listened to — a pipeline that
returns plausible-looking bytes but unlistenable audio would pass any assertion.
Run: .venv_tts/bin/python brain/test_pipeline.py
"""
import asyncio
import json
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.pipeline import (Session, Tts, Turn, split_sentences,  # noqa: E402
                            strip_tool_calls, take_sentences)

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "testdata" / "a1_tired.wav"
OUT = ROOT / "tts_eval" / "pipeline_reply.wav"

SYSTEM = ("너는 공감 로봇 모티야. 사용자의 말에 따뜻하게 공감하며 대화해. "
          "답변은 2~3문장으로 짧게 해. 이모지는 쓰지 마.")
TOOLS = [{"type": "function", "function": {
    "name": "set_emotion", "description": "로봇 표정을 바꾼다",
    "parameters": {"type": "object", "properties": {"emotion": {
        "type": "string", "enum": ["neutral", "happy", "sad", "tender", "excited"]}},
        "required": ["emotion"]}}}]


def test_split_sentences() -> None:
    # Korean endings must split even without punctuation (§11.0-2).
    assert split_sentences("정말 피곤하시겠어요 푹 쉬세요") == ["정말 피곤하시겠어요", "푹 쉬세요"]
    assert split_sentences("안녕하세요! 반가워요.") == ["안녕하세요!", "반가워요."]
    assert split_sentences("") == []
    print("split_sentences OK")


def test_take_sentences() -> None:
    # Rejoining split pieces used to eat the trailing space, fusing the next delta onto
    # the previous word so TTS said "정도로지치셨다니" as one word.
    buf, out = "", []
    for delta in ["눈이 감길 정도로 ", "지치셨다니 걱정돼요. ", "푹 쉬세요."]:
        buf += delta
        done, buf = take_sentences(buf)
        out += done
    out.append(buf.strip())
    joined = " ".join(out)
    assert "정도로 지치셨다니" in joined, joined
    assert joined.count("푹 쉬세요") == 1, joined      # and no duplication
    print("take_sentences OK")


def test_strip_tool_calls() -> None:
    # vLLM 0.19.0 streams gemma4 tool calls as raw markup inside content. Unstripped it
    # reaches TTS and the robot says "tool call set emotion sad" aloud.
    names = ("set_emotion", "remember_fact")

    # The usual shape has no wrapper at all — handling only the wrapped one let this
    # reach TTS and the robot read "set emotion sad" aloud during the first round trip.
    bare = 'set_emotion{emotion:<|"|>sad<|"|>} 며칠 내내 밤을 새우셨군요.'
    text, calls, tail = strip_tool_calls(bare, names)
    assert "set_emotion" not in text and not tail, (text, tail)
    assert json.loads(calls[0]["arguments"]) == {"emotion": "sad"}, calls

    # Ordinary braces must survive: only declared tool names count as a call.
    text, calls, _ = strip_tool_calls("수식은 f{x}처럼 씁니다.", names)
    assert text == "수식은 f{x}처럼 씁니다." and not calls, (text, calls)

    real = ('<|tool_call>call:set_emotion{emotion:<|"|>sad<|"|>}<tool_call|>'
            "아이고, 피곤하시겠어요.")
    text, calls, tail = strip_tool_calls(real, names)
    assert text == "아이고, 피곤하시겠어요." and not tail, (text, tail)
    assert calls[0]["name"] == "set_emotion", calls
    assert json.loads(calls[0]["arguments"]) == {"emotion": "sad"}, calls

    # ...and it must survive arriving in fragments, which is how streaming delivers it.
    raw, spoken, found = "", "", []
    for i in range(0, len(bare), 7):
        raw += bare[i:i + 7]
        clean, calls, raw = strip_tool_calls(raw, names)
        found += calls
        spoken += clean
    spoken += raw
    assert "<|tool" not in spoken and "set_emotion" not in spoken, spoken
    assert len(found) == 1, found
    print("strip_tool_calls OK")


def load_pcm16k(path: Path) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


async def main() -> None:
    test_split_sentences()
    test_take_sentences()
    test_strip_tool_calls()
    pcm = load_pcm16k(CLIP)
    print(f"입력 {CLIP.name}: {len(pcm) / 2 / 16000:.1f}s")

    tts = Tts()
    print(f"Piper 로드 완료 (출력 {tts.rate}Hz)")

    events, chunks = [], []

    async def emit(kind, payload):
        events.append(kind)
        if kind == "audio":
            chunks.append(payload["pcm"])
            print(f"  audio  +{len(payload['pcm']) / 2 / payload['rate']:.2f}s")
        elif kind == "transcript":
            print(f"  [{payload['role']}] {payload['text']}")
        elif kind == "tool_call":
            for c in payload["calls"]:
                print(f"  tool   {c['name']}({c['arguments']})")
        elif kind == "error":
            print(f"  ERROR  {payload['detail']}")

    turn = Turn(Session(system=SYSTEM, tools=TOOLS), tts, pcm)
    await turn.run(emit)

    assert "transcript" in events, "전사가 나오지 않음"
    assert chunks, "오디오가 생성되지 않음"

    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(tts.rate)
        w.writeframes(b"".join(chunks))

    secs = sum(len(c) for c in chunks) / 2 / tts.rate
    m = turn.marks
    print(f"\n합성 음성 {secs:.1f}s -> {OUT}")
    print(f"계측: 전사 {m.get('transcript', 0):.2f}s | "
          f"첫 토큰 {m.get('first_token', 0):.2f}s | 첫 오디오 {m.get('first_audio', 0):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
