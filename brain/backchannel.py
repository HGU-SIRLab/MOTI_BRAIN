"""EXP-12: fill the dead air while the VAD is still deciding (§4.2, §12).

§13.4 measured 1.98s of perceived latency, of which 1.5s is the `stop_secs` silence.
Until smart-turn removes that wait (Q19b), the user hears nothing at all for a second
and a half after finishing a sentence. A short "응" at 0.6s does not make the reply
arrive sooner — it makes the robot stop looking deaf, which §4.2 argues is where most of
the perceived liveness actually comes from.

Pre-synthesized once at startup: a full LLM round trip would defeat the purpose, and
Piper at RTF 0.22 would still add hundreds of milliseconds per token if called live.
"""
from __future__ import annotations

import random

import numpy as np

# Deliberately minimal and non-committal. Anything with content ("그렇구나") is dropped:
# the user has not finished the sentence, so agreeing with it is a guess.
PHRASES = ["응", "응...", "어", "음"]

_SILENCE = 300          # int16 amplitude below which we call it silence


def _trim(pcm: bytes) -> bytes:
    """Strip Piper's leading/trailing padding.

    Raw synthesis of "응" runs 0.81s, nearly all of it silence. A backchannel that long
    overlaps the real reply and stops reading as a quick acknowledgement.
    """
    x = np.frombuffer(pcm, dtype=np.int16)
    loud = np.where(np.abs(x) > _SILENCE)[0]
    if len(loud) == 0:
        return pcm
    return x[loud[0]:loud[-1] + 1].tobytes()


class Backchannel:
    def __init__(self, tts, phrases: list[str] | None = None):
        self.rate = tts.rate
        self.clips = [_trim(tts.synth(p)) for p in (phrases or PHRASES)]
        self._last = -1

    def pick(self) -> bytes:
        """Never the same one twice running — repetition is what makes a canned
        acknowledgement sound canned."""
        choices = [i for i in range(len(self.clips)) if i != self._last]
        self._last = random.choice(choices)
        return self.clips[self._last]
