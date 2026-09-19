"""Turn detection for the brain (§9, §11.0-3).

Silero VAD over ONNX Runtime — no torch. That is deliberate: this machine exports a
global PYTHONPATH pointing at HARU's ROS venv, so any pip torch lands next to a Jetson
torch and breaks (see the troubleshooting memory). The ONNX path avoids the question.

Endpointing is a hybrid (§9.1c): smart-turn ends the turn early when it is confident the
speaker finished; otherwise the silence timer is the backstop. Neither alone is good enough.

The timer alone is genuinely bad on spontaneous speech. Natural thinking pauses in our
recordings reach **2.98s**, so `stop_secs=1.5` cuts the user off at 19 of 22 pauses. Raising
it to clear 2.98s would put a 3-second wait on every single turn.

smart-turn is wired as a **veto on the timer, not an accelerator**. When the timer is about to
end the turn, smart-turn is asked whether the speaker actually finished; if it says no, the wait
is extended up to `max_wait`. Measured on spontaneous speech, internal pauses score a median of
0.019 — it is confident, and honouring that is what removes false interruptions.

The first attempt did the opposite (end *early* when confident) and measured **worse than no
smart-turn at all**: it can only add splits, never prevent them, because the timer still fires
at 1.5s regardless. Direction matters more than threshold here.
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
                 pad_secs: float = 0.3, min_speech_secs: float = 0.3,
                 endpoint_confidence: float = 0.7, max_wait: float = 4.0,
                 fast_after: float = 0.6, fast_confidence: float = 0.95):
        self.session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        self.threshold = threshold
        # When the timer expires, smart-turn gets a veto: below `endpoint_confidence` the
        # speaker is judged still going and the wait extends, to at most `max_wait`. The
        # cap matters — a wrong veto must not hang the conversation.
        self.endpoint_confidence = endpoint_confidence
        self.max_wait_windows = max(1, int(max_wait * RATE / WINDOW))
        # Fast path (§9.1e). Real-time conversation is the goal and waiting out the timer
        # on every turn is not it, so smart-turn gets one early question and may release
        # the turn at 0.6s.
        #
        # 0.95 is a deliberate trade, not a tuned optimum: end-of-turn wait drops
        # 1.22s -> 0.77s, at the cost of splitting `c2_suppressed` — the emotionally
        # suppressed take, which is precisely the user §9.1a says must not be cut off.
        # A wrong fire is recovered by barge-in in 70ms (§13.3), so it is brief rather
        # than harmless. **Set this to 0.99 to effectively disable the fast path** and get
        # back the safe-but-slower behaviour; that is the one-line revert.
        self.fast_windows = max(1, int(fast_after * RATE / WINDOW))
        self.fast_confidence = fast_confidence
        # How much trailing silence to show smart-turn. It matters a lot and in both
        # directions: strip it entirely and the model has no pause to judge; hand it the
        # full accumulated silence and everything reads as "finished" (the veto fired on
        # 1 of 22 pauses with 0.3s, 6 of 22 with the lot). The reference calls the model
        # right as VAD detects silence, i.e. with a short fixed tail.
        self.pause_windows = max(1, int(0.3 * RATE / WINDOW))
        self._smart = None
        try:
            from brain.smart_turn import SmartTurn
            self._smart = SmartTurn()
        except Exception:                        # noqa: BLE001 — degrade to timer only
            pass
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
        self._asked_at = -1.0            # smart-turn consulted once per pause (veto)
        self._asked_fast = False         # ...and once early (fast path)
        self._vetoed = False             # ...and it said the speaker is not done

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
                self._asked_at = -1.0
                self._asked_fast = False
                self._vetoed = False
                self._speech_windows += 1
            else:
                self._silence += 1
                pause = self._silence * WINDOW / RATE
                early = False
                done = self._silence >= self.stop_windows

                # Fast path: one question, early in the pause.
                if (not done and self._smart is not None and not self._asked_fast
                        and self._silence >= self.fast_windows
                        and self._speech_windows >= self.min_speech_windows):
                    self._asked_fast = True
                    if self._smart.probability(self._with_pause()) >= self.fast_confidence:
                        done = True

                if done and self._smart is not None and self._asked_at < 0:
                    # Ask once per pause. Re-asking every window would turn a modest
                    # per-check error rate into a near-certainty over a long pause.
                    self._asked_at = pause
                    if self._smart.probability(self._with_pause()) < self.endpoint_confidence:
                        self._vetoed = True
                if self._vetoed and self._silence < self.max_wait_windows:
                    done = False
                if done:
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

    def _with_pause(self) -> bytes:
        """Turn audio plus a fixed short tail of silence — the shape smart-turn expects."""
        keep = min(self._silence, self.pause_windows)
        drop = self._silence - keep
        return b"".join(self._turn[:len(self._turn) - drop])

    def provisional(self) -> bytes | None:
        """The turn as it stands mid-pause, for speculative generation.

        Same bytes the confirmed turn will contain if the speaker does not resume, so a
        reply generated from this is valid the moment the turn closes.
        """
        if (not self._speaking or self._silence == 0
                or self._speech_windows < self.min_speech_windows):
            return None
        return b"".join(self._turn[:-self._silence])

    def flush(self) -> bytes | None:
        """End an in-progress turn, e.g. the stream closed. Returns audio or None."""
        if self._speaking and self._speech_windows >= self.min_speech_windows:
            audio = b"".join(self._turn)
            self.reset()
            return audio
        self.reset()
        return None
