#!/usr/bin/env python3
"""EXP-13 plumbing probe: does E4B accept audio via vLLM, and what does it cost?

Uses generated tones, not speech — this measures the audio *path* and token
accounting (§12.2 kill criteria 2 and 3). Comprehension quality needs real
Korean recordings and is a separate run.
"""
import base64
import io
import json
import math
import struct
import urllib.error
import urllib.request
import wave

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "google/gemma-4-E4B-it"
RATE = 16000


def tone_wav(seconds: float, hz: float = 220.0) -> bytes:
    """16kHz mono PCM16 WAV of a sine tone."""
    n = int(RATE * seconds)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * hz * i / RATE)))
        for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(frames)
    return buf.getvalue()


def ask(parts: list, max_tokens: int = 24) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:400], "status": e.code}


def audio_part(seconds: float) -> dict:
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(tone_wav(seconds)).decode(),
            "format": "wav",
        },
    }


TEXT = {"type": "text", "text": "이 소리를 한 문장으로 설명해."}


def main() -> None:
    base = ask([TEXT])
    if "error" in base:
        print("텍스트 기준선 실패:", base["error"])
        return
    baseline = base["usage"]["prompt_tokens"]
    print(f"오디오 없는 기준선 prompt_tokens = {baseline}\n")

    print(f"{'길이':>6} {'prompt_tok':>11} {'오디오분':>9} {'tok/s':>7}  비고")
    prev = None
    for sec in (5, 10, 30, 45):
        r = ask([TEXT, audio_part(sec)])
        if "error" in r:
            print(f"{sec:5d}s {'—':>11} {'—':>9} {'—':>7}  HTTP {r['status']}: "
                  f"{r['error'][:150]}")
            continue
        tok = r["usage"]["prompt_tokens"]
        audio_tok = tok - baseline
        print(f"{sec:5d}s {tok:11d} {audio_tok:9d} {audio_tok/sec:7.1f}  "
              f"{r['choices'][0]['message']['content'].strip()[:40]!r}")
        prev = audio_tok

    # 기준 3: 30초 상한을 쪼개서 우회할 수 있는가 — 오디오 파트 2개를 한 번에.
    print("\n멀티클립(20s x 2, §12.2 기준3 우회책):")
    r = ask([TEXT, audio_part(20), audio_part(20)])
    if "error" in r:
        print(f"  실패 HTTP {r['status']}: {r['error'][:200]}")
    else:
        tok = r["usage"]["prompt_tokens"]
        print(f"  prompt_tokens {tok}  오디오분 {tok - baseline}  → 파트 2개 수용됨")


if __name__ == "__main__":
    main()
