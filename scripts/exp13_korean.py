#!/usr/bin/env python3
"""EXP-13 criterion 1: does E4B understand Korean speech, and does it hear tone?

Reads testdata/ (see docs/exp13_recordings.md), converts whatever format was
recorded to 16kHz mono PCM16, and runs three comparisons:

  A. audio-direct vs. the reference transcript as text  — is audio comprehension
     as good as a perfect STT would give us?
  B. one >30s clip vs. the same audio split into <=30s parts — does the silent
     truncation found in §12.3 actually lose the tail of a real utterance?
  C. three recordings of an identical sentence, differing only in prosody — if the
     replies differ, the tone channel is real and §7's separate SER is redundant.

temperature=0 throughout. Sampling noise would otherwise be indistinguishable
from a genuine reaction to tone, which would invalidate comparison C entirely.
"""
import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "google/gemma-4-E4B-it"
DATA = Path(__file__).resolve().parent.parent / "testdata"
CLIP_LIMIT_SEC = 30  # §12.3 — hard ceiling, exceeded silently

SYSTEM = (
    "너는 공감 로봇 모티야. 사용자의 말에 따뜻하게 공감하며 대화해. "
    "답변은 2~3문장으로 짧게 해. 이모지는 쓰지 마."
)

TRANSCRIPTS = {
    "a1_tired": "아, 며칠 내내 밤을 새웠더니 진짜 너무 피곤해. 눈이 막 감기네. 너도 피곤하지?",
    "a2_happy": "드디어! 며칠 동안 고생했던 프로젝트가 방금 성공적으로 끝났어. 기분 최고야!",
    "a3_anxious": "내일 중요한 발표인데 아직 준비를 다 못해서 너무 불안해. 중간에 실수하면 어떡하지?",
}
TONE_SET = ["c1_neutral", "c2_suppressed", "c3_bright"]
TONE_SENTENCE = "나 진짜 괜찮아. 너무 신경 안 써도 돼."


def find(stem: str) -> Path | None:
    hits = sorted(DATA.glob(f"{stem}.*"))
    return hits[0] if hits else None


def to_pcm16(src: Path) -> bytes:
    """Any input format -> 16kHz mono PCM16 WAV bytes."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
        check=True, capture_output=True).stdout
    return out


def duration(src: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "format=duration", "-of", "csv=p=0", str(src)], check=True, capture_output=True).stdout
    return float(out.decode().strip())


def slice_pcm16(src: Path, start: float, length: float) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(length),
         "-i", str(src), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
        check=True, capture_output=True).stdout


def audio_part(wav: bytes) -> dict:
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}


def ask(parts: list) -> tuple[str, int]:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": parts}],
        "temperature": 0.0,
        "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            d = json.load(r)
        return (d["choices"][0]["message"]["content"].strip(),
                d["usage"]["prompt_tokens"])
    except urllib.error.HTTPError as e:
        return f"[HTTP {e.code}] {e.read().decode()[:200]}", 0


def probe(src: Path) -> dict:
    meta = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,bit_rate", "-of", "json", str(src)],
        check=True, capture_output=True).stdout
    st = (json.loads(meta).get("streams") or [{}])[0]
    stats = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src), "-map", "0:a:0",
         "-af", "astats=metadata=1:reset=0", "-f", "null", "-"],
        capture_output=True).stderr.decode(errors="replace")
    peak = rms = None
    for line in stats.splitlines():
        if "Peak level dB" in line and peak is None:
            peak = line.split(":")[-1].strip()
        elif "RMS level dB" in line and rms is None:
            rms = line.split(":")[-1].strip()
    return {"codec": st.get("codec_name"), "rate": st.get("sample_rate"),
            "ch": st.get("channels"), "bitrate": st.get("bit_rate"),
            "peak": peak, "rms": rms}


def part_0() -> None:
    """Recording sanity check.

    Phones commonly apply AGC, noise suppression and dynamic-range compression,
    and AAC discards low-energy spectral detail — which is exactly where the
    breathiness and micro-tremor of c2_suppressed live. If that processing
    flattened the prosody, comparison C would return a false negative: the tone
    channel could be working while we conclude it is not. Near-identical RMS
    across the c-set is the warning sign to look for."""
    print("=" * 72)
    print("0. 녹음 상태 점검 — 톤 검증이 유효한 조건인지")
    print("=" * 72)
    print(f"{'파일':<18} {'코덱':<8} {'rate':>6} {'ch':>3} {'peak dB':>9} {'RMS dB':>9}")
    c_rms = []
    for stem in list(TRANSCRIPTS) + ["b1_long"] + TONE_SET:
        src = find(stem)
        if src is None:
            print(f"{stem:<18} (없음)")
            continue
        p = probe(src)
        print(f"{src.name:<18} {str(p['codec']):<8} {str(p['rate']):>6} "
              f"{str(p['ch']):>3} {str(p['peak']):>9} {str(p['rms']):>9}")
        if stem in TONE_SET and p["rms"]:
            try:
                c_rms.append(float(p["rms"]))
            except ValueError:
                pass
    if len(c_rms) == len(TONE_SET):
        spread = max(c_rms) - min(c_rms)
        print(f"\nc세트 RMS 편차 {spread:.1f} dB", end="  ")
        if spread < 1.5:
            print("⚠️ 거의 동일 — 휴대폰 AGC/압축이 톤 dynamics를 평탄화했을 가능성. "
                  "C 결과가 '차이 없음'으로 나오면 모델 한계가 아니라 녹음 문제일 수 있다.")
        else:
            print("→ 톤별 dynamics가 살아있음. C 검증 조건 양호.")


def part_a() -> None:
    print("=" * 72)
    print("A. 오디오 직접 입력 vs 완벽한 전사 텍스트 (기준 1)")
    print("=" * 72)
    for stem, text in TRANSCRIPTS.items():
        src = find(stem)
        if src is None:
            print(f"\n[{stem}] 파일 없음 — 건너뜀")
            continue
        print(f"\n[{stem}] {src.name}  {duration(src):.1f}s")
        a_reply, a_tok = ask([{"type": "text", "text": "방금 한 말에 공감해줘."},
                              audio_part(to_pcm16(src))])
        t_reply, t_tok = ask([{"type": "text", "text": text}])
        print(f"  오디오({a_tok} tok): {a_reply}")
        print(f"  텍스트({t_tok} tok): {t_reply}")


def part_b() -> None:
    print("\n" + "=" * 72)
    print("B. 30초 초과 발화 — 단일 클립 vs 분할 (§12.3 무음 절단 검증)")
    print("=" * 72)
    src = find("b1_long")
    if src is None:
        print("b1_long 파일 없음 — 건너뜀")
        return
    total = duration(src)
    print(f"\n{src.name}  길이 {total:.1f}s")

    whole, tok_whole = ask([{"type": "text", "text": "방금 한 말을 요약하고 공감해줘."},
                            audio_part(to_pcm16(src))])
    print(f"\n  단일 클립 전체 전송 ({tok_whole} tok):\n    {whole}")

    parts = [{"type": "text", "text": "방금 한 말을 요약하고 공감해줘."}]
    n, start = 0, 0.0
    while start < total:
        parts.append(audio_part(slice_pcm16(src, start, CLIP_LIMIT_SEC)))
        start += CLIP_LIMIT_SEC
        n += 1
    split, tok_split = ask(parts)
    print(f"\n  {CLIP_LIMIT_SEC}초씩 {n}개로 분할 ({tok_split} tok):\n    {split}")
    print(f"\n  → 토큰 차이 {tok_split - tok_whole}. 분할 쪽이 크게 많으면 "
          f"단일 전송에서 뒷부분이 실제로 유실된 것.")
    print("  → 응답 내용을 비교해 발화 후반부(주말 반납, 속이 시원하다)가 "
          "단일 클립 응답에 빠졌는지 확인할 것.")


def part_c() -> None:
    print("\n" + "=" * 72)
    print("C. 동일 문장, 톤만 다름 (SER 대체 가능성 — 핵심 검증)")
    print("=" * 72)
    print(f"공통 문장: {TONE_SENTENCE}")
    print("temperature=0이므로 응답 차이는 샘플링 노이즈가 아니라 음향 특징에서만 나온다.\n")

    text_reply, _ = ask([{"type": "text", "text": TONE_SENTENCE}])
    print(f"[텍스트만 — Whisper 경로가 보게 될 것]\n    {text_reply}\n")

    replies = {}
    for stem in TONE_SET:
        src = find(stem)
        if src is None:
            print(f"[{stem}] 파일 없음 — 건너뜀")
            continue
        reply, _ = ask([{"type": "text", "text": "방금 한 말에 공감해줘."},
                        audio_part(to_pcm16(src))])
        replies[stem] = reply
        print(f"[{stem}]\n    {reply}\n")

    uniq = len(set(replies.values()))
    if uniq <= 1 and replies:
        print(f"판정: 응답 {len(replies)}개가 모두 동일 → 톤 채널이 작동하지 않음. "
              f"§7의 별도 SER 모델이 여전히 필요하다.")
    elif replies:
        print(f"판정: 서로 다른 응답 {uniq}종 → 톤 채널이 실제로 작동. "
              f"§7의 emotion2vec 없이 affective dialog가 가능하다는 직접 증거.")


def main() -> None:
    if not DATA.is_dir() or not any(DATA.iterdir()):
        print(f"{DATA}에 녹음 파일이 없습니다. docs/exp13_recordings.md 참고.")
        sys.exit(1)
    part_0()
    part_a()
    part_b()
    part_c()


if __name__ == "__main__":
    main()
