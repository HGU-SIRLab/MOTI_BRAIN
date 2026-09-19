"""Check turn detection against the real recordings.

The recordings are single utterances, so correct behaviour is: exactly one turn each,
covering roughly the file's speech, and the first syllable must survive (§11.0-3).
Run: PYTHONPATH= .venv_tts/bin/python brain/test_vad.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.vad import RATE, TurnDetector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Spontaneous recordings are the representative set: the scripted ones have unnaturally
# short pauses (1.38s vs 2.98s) and smart-turn behaves differently on read speech (§9.1b).
# They are still checked, but a split there is informational, not a failure.
CLIPS = ["s1_recall", "s2_project", "s3_weekend", "s4_undecided", "s5_explain", "s6_explain"]
SCRIPTED = ["a1_tired", "a2_happy", "a3_anxious", "c2_suppressed", "b1_long"]
CHUNK = 1600  # 100ms of PCM16, roughly what a mic callback delivers


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


def main() -> None:
    print(f"{'파일':<16} {'입력':>7} {'턴수':>5} {'검출':>8} {'비율':>6}")
    # Each recording is one continuous monologue, so every extra turn is the brain
    # interrupting. Perfection is not the bar — the measured configuration costs 3
    # (§9.1e), against 19 for the plain 1.5s timer. The budget catches a regression
    # without failing on the trade we chose.
    SPLIT_BUDGET = 4
    failures = []
    splits = 0
    for stem in CLIPS:
        pcm = pcm16k(stem)
        det = TurnDetector()
        turns = []
        for i in range(0, len(pcm), CHUNK):
            turns += det.feed(pcm[i:i + CHUNK])
        # Real speech does not end the moment the file does; flush what is still open.
        tail = det.flush()
        if tail:
            turns.append(tail)

        secs_in = len(pcm) / 2 / RATE
        secs_out = sum(len(t) for t in turns) / 2 / RATE
        ratio = secs_out / secs_in if secs_in else 0
        print(f"{stem:<16} {secs_in:6.1f}s {len(turns):5d} {secs_out:7.1f}s {ratio:5.0%}")

        splits += len(turns) - 1
        if ratio < 0.5:
            failures.append(f"{stem}: 음성의 {ratio:.0%}만 잡힘 — 너무 많이 버림")

    print("\n대본 클립 (참고 — 대표성 없음):")
    for stem in SCRIPTED:
        pcm = pcm16k(stem)
        det = TurnDetector()
        turns = []
        for i in range(0, len(pcm), CHUNK):
            turns += det.feed(pcm[i:i + CHUNK])
        if det.flush():
            turns.append(b"")
        print(f"  {stem:<16} {len(turns)}턴")

    print(f"\n잘못된 분할 {splits}개 (예산 {SPLIT_BUDGET}, 타이머 단독이면 19)")
    if splits > SPLIT_BUDGET:
        failures.append(f"분할 {splits}개 > 예산 {SPLIT_BUDGET} — 턴 감지가 퇴행했다")

    print()
    if failures:
        print("실패:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("턴 감지 정상 — 분할이 예산 이내")


if __name__ == "__main__":
    main()
