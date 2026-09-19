"""Gate for smart-turn, which is now wired in (§9.1e) and load-bearing.

The previous version of this file asserted that speech must score higher than silence.
That premise was wrong — 8s of silence *is* a finished turn, and the assertion outlived
its usefulness by failing for the wrong reason. What matters now is the discrimination
the endpointing actually depends on: a pause the speaker talks through must score low,
and the end of an utterance must score high.

Labels come free from the spontaneous recordings: every internal pause is ground truth
"still going" (they continued), the end of each file is "finished".

Run: PYTHONPATH= .venv_tts/bin/python brain/test_smart_turn.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.smart_turn import FRAMES, N_MELS, SmartTurn, log_mel  # noqa: E402
from brain.vad import CONTEXT, RATE, WINDOW  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ["s1_recall", "s2_project", "s3_weekend", "s4_undecided", "s5_explain",
         "s6_explain"]
PAUSE = b"\x00" * (int(0.3 * RATE) * 2)     # the tail the model is given (§9.1e)


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


def speech_mask(pcm: bytes) -> np.ndarray:
    vad = ort.InferenceSession(str(ROOT / "models" / "silero_vad.onnx"),
                               providers=["CPUExecutionProvider"])
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    state = np.zeros((2, 1, 128), dtype=np.float32)
    ctx = np.zeros(CONTEXT, dtype=np.float32)
    out = []
    for i in range(0, len(x) - WINDOW, WINDOW):
        chunk = x[i:i + WINDOW]
        p, state = vad.run(None, {"input": np.concatenate([ctx, chunk])[None, :],
                                  "state": state,
                                  "sr": np.array(RATE, dtype=np.int64)})
        out.append(float(p[0][0]) >= 0.5)
        ctx = chunk[-CONTEXT:]
    return np.array(out)


def main() -> None:
    m = log_mel(np.zeros(8 * RATE, dtype=np.float32))
    assert m.shape == (N_MELS, FRAMES), m.shape

    st = SmartTurn()
    going, ended = [], []
    for stem in CLIPS:
        pcm = pcm16k(stem)
        mask = speech_mask(pcm)
        run, started = 0, False
        for i, voiced in enumerate(mask):
            if voiced:
                if started and run * WINDOW / RATE >= 1.0:
                    cut = (i - run) * WINDOW * 2
                    going.append(st.probability(pcm[:cut] + PAUSE))
                started, run = True, 0
            elif started:
                run += 1
        last = np.where(mask)[0][-1]
        ended.append(st.probability(pcm[:(last + 1) * WINDOW * 2] + PAUSE))

    going, ended = np.array(going), np.array(ended)
    print(f"계속 중 (내부 침묵) {len(going):2d}개: 중앙 {np.median(going):.3f}")
    print(f"끝남   (발화 종료) {len(ended):2d}개: 중앙 {np.median(ended):.3f}")

    # The fast path releases a turn at >=0.95; the veto holds it below 0.7 (§9.1e).
    fast = (ended >= 0.95).sum()
    premature = (going >= 0.95).sum()
    print(f"빠른경로 0.95: 즉시응답 {fast}/{len(ended)} | 섣부른 {premature}/{len(going)}")

    if np.median(going) >= np.median(ended):
        print("\n실패: 계속 중인 침묵이 발화 종료만큼 높게 나온다 — 분리가 안 된다.")
        sys.exit(1)
    if premature > len(going) * 0.2:
        print(f"\n실패: 섣부른 응답이 {premature}/{len(going)}로 너무 많다.")
        sys.exit(1)
    print("\n통과 — 턴 감지가 의존하는 분리가 유지된다")


if __name__ == "__main__":
    main()
