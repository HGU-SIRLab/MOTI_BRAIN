"""Gate for smart-turn. Currently FAILS by design — the mel is wrong (see smart_turn.py).

The first version of this test only compared a full clip against a truncated one and
passed on differences of 0.001. That was too weak to notice the model had no opinion.
The discriminating check is against extremes: if silence and white noise score at or
above genuinely finished speech, the preprocessing is not producing Whisper features.

Run: PYTHONPATH= .venv_tts/bin/python brain/test_smart_turn.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.smart_turn import FRAMES, N_MELS, SmartTurn, log_mel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def pcm16k(stem: str) -> bytes:
    src = next((ROOT / "testdata").glob(f"{stem}.*"))
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True).stdout


def main() -> None:
    st = SmartTurn()
    speech = [st.probability(pcm16k(s))
              for s in ("a1_tired", "a2_happy", "a3_anxious", "c1_neutral")]
    silence = st.probability(np.zeros(16000 * 4, dtype=np.int16).tobytes())
    noise = st.probability((np.random.randn(16000 * 4) * 3000).astype(np.int16).tobytes())

    m = log_mel(np.frombuffer(pcm16k("a1_tired"), dtype=np.int16).astype(np.float32) / 32768.0)
    assert m.shape == (N_MELS, FRAMES), m.shape

    print(f"완결 발화 P(끝남): {[round(p, 3) for p in speech]}  (최저 {min(speech):.3f})")
    print(f"무음 {silence:.3f}   백색소음 {noise:.3f}")

    # Speech that clearly ended must beat structureless audio. It does not, today.
    if min(speech) <= max(silence, noise):
        print("\n실패: 무음/소음이 완결 발화 이상으로 채점됨 → mel 전처리가 틀렸다.")
        print("smart-turn을 파이프라인에 연결하지 말 것. vad.py의 stop_secs 폴백 사용 중.")
        sys.exit(1)
    print("\n통과")


if __name__ == "__main__":
    main()
