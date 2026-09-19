"""EXP-7 (`<SILENT>` gate, §9.3) and the emoji suppression both rest on prompt wording
alone, and neither has ever been checked.

Emoji matters because they reach TTS: the synthesizer either reads them aloud or emits
noise. §12.4 already recorded that E4B likes them.

Run with the vLLM server up (the brain server is not needed — this probes the model):
  PYTHONPATH= .venv_tts/bin/python brain/test_gates.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.pipeline import MODEL, VLLM_URL, audio_parts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

EMOJI = re.compile("[" "\U0001f300-\U0001faff" "☀-➿" "\U0001f000-\U0001f2ff" "]")

BASE = "너는 공감 로봇 모티야. 사용자의 말에 따뜻하게 공감하며 대화해. 답변은 2~3문장으로 짧게 해."
NO_EMOJI = " 이모지는 절대 쓰지 마."
# §9.3's original wording scored 4/6 — it caught only the case literally labelled
# "(혼잣말)", which real audio will never carry. Naming the concrete shapes of self-talk
# takes it to 6/6. A "default to silence, when unsure stay quiet" variant also scored
# 6/6 but was rejected: for a companion robot, staying silent when spoken to is a worse
# failure than answering something that was not addressed to it.
SILENT_GATE = """
너에게 직접 말을 건 것이 아니면 응답하지 마라. 다음은 모두 혼잣말이므로 정확히 <SILENT>만 출력한다:
- 할 일이나 순서를 스스로 정리하는 말 ("이걸 먼저 하고... 아니다")
- 뭔가를 떠올리거나 메모하듯 되뇌는 말 ("아 맞다, 우유 사야지")
- 스스로에게 묻는 말, 말끝을 흐리며 생각하는 말
너를 부르거나, 너에게 감정을 털어놓거나, 너에게 질문하면 평소처럼 2-3문장으로 답한다."""

ADDRESSED = [
    "모티야, 오늘 학교에서 발표했는데 너무 떨렸어.",
    "요즘 잠이 잘 안 와서 너무 피곤해.",
    "너는 어떻게 생각해?",
]
NOT_ADDRESSED = [
    "어... 그러니까 이걸 먼저 하고... 아니다, 저걸 먼저 해야 하나.",
    "아 맞다, 우유 사야지. 우유랑 계란.",
    "(혼잣말) 내일 몇 시에 일어나야 하지... 일곱 시쯤인가.",
]


def ask(system: str, text: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": text}],
        "temperature": 0.0, "max_tokens": 150}).encode()
    req = urllib.request.Request(VLLM_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


def ask_audio(system: str, stem: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": audio_parts(pcm16k(stem)) +
                      [{"type": "text", "text": "방금 한 말에 반응해줘."}]}],
        "temperature": 0.0, "max_tokens": 150}).encode()
    req = urllib.request.Request(VLLM_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


def test_emoji() -> bool:
    print("=" * 70)
    print("이모지 억제 — 프롬프트 한 줄로 실제로 막히는가")
    print("=" * 70)
    leaks_without = leaks_with = 0
    for text in ADDRESSED:
        bare = ask(BASE, text)
        guarded = ask(BASE + NO_EMOJI, text)
        leaks_without += bool(EMOJI.search(bare))
        leaks_with += bool(EMOJI.search(guarded))
        print(f"  억제 없음: {'이모지 O' if EMOJI.search(bare) else '이모지 X'}  "
              f"| 억제 지시: {'이모지 O ✗' if EMOJI.search(guarded) else '이모지 X ✓'}")
    # Audio input too — that is the production path.
    a = ask_audio(BASE + NO_EMOJI, "a1_tired")
    audio_leak = bool(EMOJI.search(a))
    print(f"  오디오 입력 경로: {'이모지 O ✗' if audio_leak else '이모지 X ✓'}")
    print(f"\n  억제 없을 때 {leaks_without}/{len(ADDRESSED)} 누출, "
          f"억제 지시 시 {leaks_with}/{len(ADDRESSED)} 누출")
    return leaks_with == 0 and not audio_leak


def test_silent_gate() -> bool:
    print()
    print("=" * 70)
    print("EXP-7: <SILENT> 게이트 (§9.3 문구 그대로)")
    print("=" * 70)
    system = BASE + NO_EMOJI + SILENT_GATE
    hits = misses = 0
    print("  [말을 건 경우 — 응답해야 함]")
    for text in ADDRESSED:
        out = ask(system, text)
        silent = "<SILENT>" in out
        hits += not silent
        print(f"    {'✗ 침묵' if silent else '✓ 응답'}  {text[:32]}")
    print("  [혼잣말 — 침묵해야 함]")
    for text in NOT_ADDRESSED:
        out = ask(system, text)
        silent = "<SILENT>" in out
        misses += silent
        print(f"    {'✓ 침묵' if silent else '✗ 응답: ' + out[:28]}  {text[:28]}")
    total = len(ADDRESSED) + len(NOT_ADDRESSED)
    print(f"\n  정확도 {hits + misses}/{total}")
    return (hits + misses) == total


if __name__ == "__main__":
    ok_emoji = test_emoji()
    ok_gate = test_silent_gate()
    print()
    print(f"이모지 억제: {'통과' if ok_emoji else '실패'} | "
          f"<SILENT> 게이트: {'통과' if ok_gate else '실패'}")
    sys.exit(0 if (ok_emoji and ok_gate) else 1)
