"""smart-turn-v3 — wired in and load-bearing (§9.1e).

🔄 *Corrected 2026-09-21.* This docstring said "PROVEN CORRECT, but still NOT WIRED IN"
for two days after it stopped being true. Q19b resolved it on 2026-09-19 and the model has
been in the turn-detection path ever since; the header was written when it was not, and
nothing failed when it went stale. Exactly what §20 rule 19 is about.

**How it is used now**: asked twice per pause, for opposite purposes.
  fast path  — 0.6s into the pause, >= 0.95 ends the turn immediately
  veto       — when the 1.5s timer expires, < 0.7 extends the wait to 4.0s max
Together: 3 wrong splits over six spontaneous recordings against 19 for the plain timer,
at the same 0.77s wait. `fast_confidence = 0.99` disables the fast path in one line.

**Two bugs were fixed to get here**, both found by diffing against the reference
(pipecat-ai/smart-turn `inference.py`):
  1. missing `do_normalize` — zero-mean/unit-variance on the waveform before the mel
  2. a sigmoid applied to an output that is *already* a probability despite being named
     `logits`. That squashed everything into 0.50-0.73 = sigmoid(0)-sigmoid(1); the model
     had been answering 0.0 and 1.0 the whole time.
`log_mel()` matches `WhisperFeatureExtractor(chunk_length=8, do_normalize=True)`
**exactly** — max absolute error 0.0000 across all 80x800 values, filterbank identical.

**The methodology lesson outlived the bugs.** It scored 2/5 on scripted recordings and
that was read as "the model is broken". It was the test material: read speech does not
carry turn-final prosody. On spontaneous audio the separation is 0.019 vs 0.880.

Original intent below. ------------------------------------------------------------

smart-turn-v3: has the speaker actually finished, or just paused? (§9.1)

Why this exists: measured on our own recordings, intra-utterance pauses reach 1.06s.
Pure silence thresholding therefore needs stop_secs > 1.2s to avoid cutting people
off mid-sentence, and that 1.2s lands on every single turn's latency. Semantic
turn-end detection lets the silence threshold stay short.

The model is a Whisper Tiny encoder + linear head, so it wants Whisper's log-mel:
80 mels x 800 frames (8s @ 10ms hop). numpy only — no torch on this machine (see vad.py).

A wrong mel does not raise; it produces confident nonsense. `test_smart_turn.py`
checks behaviour instead of internals: a finished question must score higher than the
same audio cut off mid-word.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
FRAMES = 800                     # model input width = 8s
MODEL = Path(__file__).resolve().parent.parent / "models" / "smart_turn_v32.onnx"


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney mel: linear below 1kHz, log above. librosa's default, which Whisper uses."""
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (hz - f_min) / f_sp
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    high = hz >= min_log_hz
    mels[high] = min_log_mel + np.log(hz[high] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    high = mels >= min_log_mel
    freqs[high] = min_log_hz * np.exp(logstep * (mels[high] - min_log_mel))
    return freqs


def _mel_filters() -> np.ndarray:
    """librosa.filters.mel(sr=16000, n_fft=400, n_mels=80) with slaney normalization."""
    fft_freqs = np.fft.rfftfreq(N_FFT, 1.0 / RATE)
    mel_pts = np.linspace(_hz_to_mel(np.array([0.0]))[0],
                          _hz_to_mel(np.array([RATE / 2.0]))[0], N_MELS + 2)
    hz_pts = _mel_to_hz(mel_pts)
    diff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    weights = np.zeros((N_MELS, len(fft_freqs)), dtype=np.float32)
    for i in range(N_MELS):
        lower = -ramps[i] / diff[i]
        upper = ramps[i + 2] / diff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    # Slaney area normalization — without it the band energies scale with bandwidth.
    weights *= (2.0 / (hz_pts[2:N_MELS + 2] - hz_pts[:N_MELS]))[:, None]
    return weights


_FILTERS = _mel_filters()
_WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic, matches torch.hann_window


def log_mel(audio: np.ndarray) -> np.ndarray:
    """float32 mono @16kHz -> (80, FRAMES), Whisper features.

    Mirrors the reference (pipecat-ai/smart-turn `inference.py`): keep the last 8s,
    left-pad if shorter, then `WhisperFeatureExtractor(..., do_normalize=True)`.
    """
    need = 8 * RATE
    audio = audio[-need:] if len(audio) >= need else np.pad(audio, (need - len(audio), 0))

    # do_normalize: zero mean, unit variance on the *waveform*. Omitting this was the
    # first of the two bugs — the mel came out on a scale the model never saw.
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)

    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=(n_frames, N_FFT),
        strides=(padded.strides[0] * HOP, padded.strides[0])) * _WINDOW
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=-1)) ** 2
    mel = _FILTERS @ power[:-1].T                       # Whisper drops the last frame

    spec = np.log10(np.maximum(mel, 1e-10))
    spec = np.maximum(spec, spec.max() - 8.0)
    spec = (spec + 4.0) / 4.0
    return spec[:, :FRAMES].astype(np.float32)


class SmartTurn:
    def __init__(self, threshold: float = 0.5):
        self.session = ort.InferenceSession(str(MODEL),
                                            providers=["CPUExecutionProvider"])
        self.threshold = threshold

    def probability(self, pcm: bytes) -> float:
        """P(the speaker finished) for the trailing audio of a turn.

        The ONNX output is **already a probability** despite being named `logits`; the
        reference reads it straight. Applying a sigmoid on top was the second bug and it
        squashed everything into 0.50–0.73, which is exactly sigmoid(0)–sigmoid(1) — the
        model had been answering 0.0 and 1.0 all along.
        """
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        out = self.session.run(None, {"input_features": log_mel(audio)[None, ...]})[0]
        return float(out[0][0])

    def is_complete(self, pcm: bytes) -> bool:
        return self.probability(pcm) >= self.threshold
