"""
Multi-gate confidence filter for streaming gesture recognition.

The live BSL pipeline emits a prediction every frame, but a single confident
frame is not enough to speak a word out loud: webcam glitches, half-formed
signs, and brief look-alikes between gestures all produce confident but wrong
predictions. ``ConfidenceGate`` sits between the LSTM and the TTS layer and
only lets a label through when several independent checks agree.

The gates, in order of evaluation:

1. **Softmax margin.** Top-1 probability minus top-2 probability. A model that
   is "85% A vs 12% B" is more trustworthy than "85% A vs 80% B"; the second
   case usually means the gesture is mid-transition.
2. **Predictive entropy.** Shannon entropy of the full softmax. A flat
   distribution across many classes is rejected even if the top-1 score
   happens to clear the threshold.
3. **Temporal majority vote.** The same label must dominate a sliding window
   of recent confident predictions. Catches single-frame spikes.
4. **Stability counter.** The same majority label must repeat for N
   consecutive votes before we commit. Catches oscillation between two
   labels that each briefly win the majority.
5. **Announcement cooldown.** After we speak a label, suppress further
   announcements for a short window so the speaker is not interrupted.

The gate is fully reset whenever the hands leave the frame; a sign sequence
cannot span a moment of "no input" without an explicit re-trigger.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Sequence


@dataclass(frozen=True)
class GateDecision:
    """Outcome of a single ``ConfidenceGate.evaluate`` call.

    Attributes:
        announce: True only when every gate passed and the label is fresh.
        label:    The label that the gate currently considers most likely,
                  even when ``announce`` is False. Useful for on-screen
                  candidate display.
        reason:   Short string describing which gate the prediction failed,
                  or ``"accepted"`` when ``announce`` is True. Drives
                  diagnostics without forcing the caller to re-derive state.
    """

    announce: bool
    label: Optional[str]
    reason: str


@dataclass
class ConfidenceGate:
    """Streaming filter that decides when a gesture is stable enough to speak.

    All thresholds are constructor arguments so tests and tuning scripts can
    sweep them without subclassing. Defaults match the values used in the
    live demo on the 45-class BSL set.

    Args:
        min_top1: Lower bound on the top-1 softmax probability.
        min_margin: Required gap between top-1 and top-2 probabilities.
        max_entropy: Upper bound on Shannon entropy (in nats) of the
            softmax distribution. ``None`` disables the entropy gate, which
            is useful when the model only has a handful of classes and a
            uniform distribution still has low entropy.
        window_size: Length of the sliding majority-vote window.
        min_majority: Minimum count of the dominant label inside the window.
        stability: Number of consecutive winning votes required before the
            label is allowed to fire.
        cooldown_sec: Suppress repeat announcements for this many seconds.
        clock: Optional callable returning the current monotonic time. Pull
            it out so tests do not have to sleep.
    """

    min_top1: float = 0.70
    min_margin: float = 0.15
    max_entropy: Optional[float] = 2.0
    window_size: int = 7
    min_majority: int = 4
    stability: int = 3
    cooldown_sec: float = 1.5
    clock: callable = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # validate once at construction; cheap and saves debugging time later
        if not (0.0 <= self.min_top1 <= 1.0):
            raise ValueError("min_top1 must be in [0, 1]")
        if not (0.0 <= self.min_margin <= 1.0):
            raise ValueError("min_margin must be in [0, 1]")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not (1 <= self.min_majority <= self.window_size):
            raise ValueError("min_majority must be in [1, window_size]")
        if self.stability < 1:
            raise ValueError("stability must be >= 1")
        if self.cooldown_sec < 0.0:
            raise ValueError("cooldown_sec must be non-negative")

        # default to monotonic so cooldown is robust to wall-clock jumps
        if self.clock is None:
            import time

            self.clock = time.monotonic

        self._history: Deque[str] = deque(maxlen=self.window_size)
        self._candidate: Optional[str] = None
        self._candidate_streak: int = 0
        self._last_spoken: Optional[str] = None
        self._suppressed_until: float = 0.0

    # ------------------------------------------------------------------ api
    def on_hands_lost(self) -> None:
        """Reset all temporal state when the signer leaves the frame.

        Calling this on every "no hands" frame is cheap (idempotent) and
        guarantees that a new gesture sequence starts from a clean slate.
        Without this reset, a stale candidate from before a long pause could
        survive into the next sign and fire incorrectly.
        """
        self._history.clear()
        self._candidate = None
        self._candidate_streak = 0
        self._last_spoken = None
        self._suppressed_until = 0.0

    def evaluate(
        self,
        probs: Sequence[float],
        labels: Sequence[str],
    ) -> GateDecision:
        """Run all gates against one frame of model output.

        ``probs`` is the post-softmax distribution and ``labels`` is the
        index-aligned label list. Returns a :class:`GateDecision`; the
        caller only speaks when ``announce`` is True.
        """
        if len(probs) != len(labels):
            raise ValueError("probs and labels must align in length")
        if not probs:
            return GateDecision(announce=False, label=None, reason="empty_probs")

        now = self.clock()

        # gate 1: top-1 must clear the absolute confidence floor
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_idx = ranked[0]
        top_prob = float(probs[top_idx])
        top_label = labels[top_idx]
        if top_prob < self.min_top1:
            return GateDecision(False, top_label, "below_min_top1")

        # gate 2: top-1 must dominate the runner-up by at least the margin.
        # this is the single most useful gate against mid-transition frames
        # where two adjacent gestures are both plausible
        second_prob = float(probs[ranked[1]]) if len(ranked) > 1 else 0.0
        if (top_prob - second_prob) < self.min_margin:
            return GateDecision(False, top_label, "below_margin")

        # gate 3: shannon entropy. low entropy = peaked distribution
        if self.max_entropy is not None:
            entropy = _shannon_entropy_nats(probs)
            if entropy > self.max_entropy:
                return GateDecision(False, top_label, "above_max_entropy")

        # gates 1-3 passed: this frame is admissible, push to history
        self._history.append(top_label)

        # gate 4: dominant label inside the rolling window
        majority_label, majority_count = Counter(self._history).most_common(1)[0]
        if majority_count < self.min_majority:
            self._candidate = None
            self._candidate_streak = 0
            return GateDecision(False, top_label, "below_majority")

        # gate 5: the same majority label must repeat for N consecutive votes
        if majority_label == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = majority_label
            self._candidate_streak = 1

        if self._candidate_streak < self.stability:
            return GateDecision(False, majority_label, "stabilising")

        # cooldown: do not re-fire the same label too quickly
        if now < self._suppressed_until and majority_label == self._last_spoken:
            return GateDecision(False, majority_label, "cooldown")

        self._last_spoken = majority_label
        self._suppressed_until = now + self.cooldown_sec
        # consume the streak so the very next frame does not immediately re-fire
        self._candidate_streak = 0
        return GateDecision(True, majority_label, "accepted")


def _shannon_entropy_nats(probs: Sequence[float]) -> float:
    """Numerically stable Shannon entropy in nats.

    Using ``math.log`` (base e) keeps the threshold easy to reason about:
    a uniform distribution over k classes has entropy ``ln(k)``, so a
    threshold of 2.0 nats roughly says "the model is concentrated on fewer
    than e^2 ~= 7 classes". The 1e-12 guard avoids ``log(0)`` for sparse
    softmax outputs without distorting non-zero terms.
    """
    total = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        total -= p * math.log(p + 1e-12)
    return total
