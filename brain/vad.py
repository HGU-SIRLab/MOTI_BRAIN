"""Turn detection for the brain (§9, §11.0-3).

Silero VAD over ONNX Runtime — no torch. That is deliberate: this machine exports a
global PYTHONPATH pointing at HARU's ROS venv, so any pip torch lands next to a Jetson
torch and breaks (see the troubleshooting memory). The ONNX path avoids the question.

smart-turn-v3 was attempted and is NOT used — see smart_turn.py for the evidence that the
hand-rolled Whisper mel is wrong (silence and white noise both score *higher* than finished
speech). Endpointing therefore rests on a silence threshold alone, and `stop_secs` is set from
measurement, not taste: intra-utterance pauses in our own recordings reach 1.38s (the emotionally suppressed take), so anything
shorter splits a single utterance into several turns. The cost is that this 1.5s lands on every
turn's latency — which is exactly what a working smart-turn would buy back.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

RATE = 16000
WINDOW = 512                     # Silero v5: 512 new samples per step @16kHz
CONTEXT = 64                     # ...prepended with the previous step's last 64 samples.
BYTES_PER_WINDOW = WINDOW * 2
# Omitting CONTEXT does not raise — the model just returns ~0.001 for everything,
# including obvious speech. Verified: with context max prob 1.000, without it 0.003.
MODEL = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"


class TurnDetector:
    """Feed PCM16 bytes, get turn boundaries out.

    speech_prob > `threshold` starts a turn; `stop_secs` of continuous non-speech ends it.
    The returned audio starts `pad_secs` *before* speech was detected (§11.0-3) — without
    that the first syllable is gone, because VAD only fires once speech is already underway.
    """

    def __init__(self, threshold: float = 0.5, stop_secs: float = 1.5,
                 pad_secs: float = 0.3, min_speech_secs: float = 0.3):
        self.session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.stop_windows = max(1, int(stop_secs * RATE / WINDOW))
        self.pad_windows = max(1, int(pad_secs * RATE / WINDOW))
        self.min_speech_windows = max(1, int(min_speech_secs * RATE / WINDOW))
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._tail = b""                # leftover bytes shorter than one window
        self._pre: list[bytes] = []      # ring buffer of pre-speech audio (§11.0-3)
        self._turn: list[bytes] = []
        self._speaking = False
        self._silence = 0
        self._speech_windows = 0
        self._run = 0                    # consecutive speech windows right now

    @property
    def in_speech(self) -> bool:
        return self._speaking

    @property
    def silence_secs(self) -> float:
        """How long the current pause has lasted, while still inside a turn.

        Backchanneling (EXP-12) reads this: it fires part-way through the pause, well
        before `stop_secs` decides the turn is over.
        """
        return self._silence * WINDOW / RATE

    @property
    def speech_run(self) -> int:
        """Consecutive speech windows in the current run (32ms each).

        Barge-in reads this rather than `in_speech`: reacting to a single window makes
        a cough cut the robot off mid-sentence, while waiting for a full turn is far too
        late — §4.2 puts useful interruption reaction near 200ms.
        """
        return self._run

    def _prob(self, window: bytes) -> float:
        samples = (np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0)
        inp = np.concatenate([self._context, samples])
        out, self._state = self.session.run(
            None, {"input": inp[None, :], "state": self._state,
                   "sr": np.array(RATE, dtype=np.int64)})
        self._context = samples[-CONTEXT:]
        return float(out[0][0])

    def feed(self, pcm: bytes) -> list[bytes]:
        """Returns completed turns (raw PCM16), usually empty."""
        turns: list[bytes] = []
        data = self._tail + pcm
        n = len(data) // BYTES_PER_WINDOW
        self._tail = data[n * BYTES_PER_WINDOW:]

        for i in range(n):
            window = data[i * BYTES_PER_WINDOW:(i + 1) * BYTES_PER_WINDOW]
            speech = self._prob(window) >= self.threshold
            self._run = self._run + 1 if speech else 0

            if not self._speaking:
                self._pre.append(window)
                if len(self._pre) > self.pad_windows:
                    self._pre.pop(0)
                if speech:
                    self._speaking = True
                    self._turn = [*self._pre, window]
                    self._pre = []
                    self._silence = 0
                    self._speech_windows = 1
                continue

            self._turn.append(window)
            if speech:
                self._silence = 0
                self._speech_windows += 1
            else:
                self._silence += 1
                if self._silence >= self.stop_windows:
                    # Drop the trailing silence that ended the turn; keep the rest.
                    audio = b"".join(self._turn[:-self._silence])
                    long_enough = self._speech_windows >= self.min_speech_windows
                    self._speaking = False
                    self._turn = []
                    self._silence = 0
                    self._speech_windows = 0
                    if long_enough:
                        turns.append(audio)
        return turns

    def flush(self) -> bytes | None:
        """End an in-progress turn, e.g. the stream closed. Returns audio or None."""
        if self._speaking and self._speech_windows >= self.min_speech_windows:
            audio = b"".join(self._turn)
            self.reset()
            return audio
        self.reset()
        return None
