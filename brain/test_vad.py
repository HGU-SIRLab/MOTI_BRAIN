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
CLIPS = ["a1_tired", "a2_happy", "a3_anxious", "c2_suppressed", "b1_long"]
CHUNK = 1600  # 100ms of PCM16, roughly what a mic callback delivers


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


def main() -> None:
    print(f"{'파일':<16} {'입력':>7} {'턴수':>5} {'검출':>8} {'비율':>6}")
    failures = []
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

        if len(turns) != 1:
            failures.append(f"{stem}: 턴이 {len(turns)}개 (1개여야 함)")
        elif ratio < 0.5:
            failures.append(f"{stem}: 음성의 {ratio:.0%}만 잡힘 — 너무 많이 버림")

    print()
    if failures:
        print("실패:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("전부 단일 턴으로 정상 검출")


if __name__ == "__main__":
    main()
