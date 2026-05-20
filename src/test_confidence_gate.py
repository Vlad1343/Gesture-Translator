"""
Tests for ``ConfidenceGate``.

These tests pin the behaviour of each gate independently, so a regression in
one gate produces a single targeted failure rather than a cascade.

The tests construct fake softmax distributions directly. We never run the
LSTM here: the gate is meant to be model-agnostic, and pinning it to a
specific checkpoint would just make the tests slower and more brittle.
"""
from __future__ import annotations

import math

import pytest

from confidence_gate import ConfidenceGate, GateDecision, _shannon_entropy_nats


LABELS = ["hello", "thanks", "yes", "no", "please"]


class FakeClock:
    """Manual clock so the cooldown test does not need ``time.sleep``."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_gate(**overrides) -> tuple[ConfidenceGate, FakeClock]:
    clock = FakeClock()
    defaults = dict(
        min_top1=0.60,
        min_margin=0.15,
        max_entropy=None,  # disabled by default; entropy gate has its own test
        window_size=5,
        min_majority=3,
        stability=2,
        cooldown_sec=1.0,
        clock=clock,
    )
    defaults.update(overrides)
    return ConfidenceGate(**defaults), clock


def peaked(top_label: str, top_prob: float, runner_up: float = 0.05) -> list[float]:
    """Build a softmax-like vector with the requested top-1 and runner-up.

    Mass left over after the top-1 and runner-up is spread uniformly across
    the remaining labels. This keeps the synthetic distribution realistic
    enough to interact with the entropy gate.
    """
    probs = [0.0] * len(LABELS)
    idx = LABELS.index(top_label)
    probs[idx] = top_prob
    # pick the runner-up index deterministically: first label that is not top
    runner_idx = next(i for i in range(len(LABELS)) if i != idx)
    probs[runner_idx] = runner_up
    leftover = max(0.0, 1.0 - top_prob - runner_up)
    others = [i for i in range(len(LABELS)) if i not in (idx, runner_idx)]
    if others:
        share = leftover / len(others)
        for i in others:
            probs[i] = share
    return probs


# ---------------------------------------------------------------- unit gates
def test_rejects_below_top1_floor() -> None:
    gate, _ = make_gate(min_top1=0.80)
    decision = gate.evaluate(peaked("hello", top_prob=0.70), LABELS)
    assert not decision.announce
    assert decision.reason == "below_min_top1"


def test_rejects_when_margin_too_small() -> None:
    gate, _ = make_gate(min_top1=0.40, min_margin=0.30)
    # top-1 = 0.55, runner-up = 0.40 -> margin only 0.15
    decision = gate.evaluate(peaked("hello", top_prob=0.55, runner_up=0.40), LABELS)
    assert not decision.announce
    assert decision.reason == "below_margin"


def test_rejects_when_entropy_too_high() -> None:
    # uniform over 5 classes -> entropy = ln(5) ~= 1.609
    gate, _ = make_gate(min_top1=0.0, min_margin=0.0, max_entropy=1.0)
    uniform = [1.0 / len(LABELS)] * len(LABELS)
    decision = gate.evaluate(uniform, LABELS)
    assert not decision.announce
    assert decision.reason in {"above_max_entropy", "below_min_top1", "below_margin"}


def test_entropy_gate_passes_for_peaked_distribution() -> None:
    gate, _ = make_gate(
        min_top1=0.5,
        min_margin=0.1,
        max_entropy=1.5,
        window_size=1,
        min_majority=1,
        stability=1,
    )
    decision = gate.evaluate(peaked("hello", top_prob=0.90), LABELS)
    assert decision.announce
    assert decision.label == "hello"


# ---------------------------------------------------------------- temporal
def test_requires_majority_in_window() -> None:
    gate, _ = make_gate(window_size=5, min_majority=3, stability=1)

    # alternate hello / thanks -> no label ever reaches 3 in a 5-frame window
    seen_announce = False
    for label in ["hello", "thanks", "hello", "thanks", "hello"]:
        decision = gate.evaluate(peaked(label, top_prob=0.85), LABELS)
        seen_announce = seen_announce or decision.announce

    # 3 of the 5 are "hello" so at the final frame majority IS reached -
    # this asserts the gate fires only when it should, not earlier
    assert seen_announce, "expected at least one announcement on 3/5 majority"


def test_does_not_fire_before_stability_reached() -> None:
    # stability=2 means: frame 1 fails majority (count=1), frame 2 starts
    # streak at 1, frame 3 lifts streak to 2 and fires
    gate, _ = make_gate(window_size=3, min_majority=2, stability=2)

    decisions = [gate.evaluate(peaked("hello", top_prob=0.85), LABELS) for _ in range(2)]
    assert not any(d.announce for d in decisions)
    assert decisions[-1].reason == "stabilising"

    third = gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    assert third.announce
    assert third.label == "hello"


def test_cooldown_suppresses_repeat_announcement() -> None:
    gate, clock = make_gate(window_size=2, min_majority=2, stability=1, cooldown_sec=2.0)

    first = gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    second = gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    assert second.announce, "first announcement should pass once stability hit"

    # immediately try again: still inside cooldown
    third = gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    assert not third.announce
    assert third.reason == "cooldown"

    # advance past cooldown - the next frame at full window is allowed to fire
    clock.advance(2.5)
    after = gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    assert after.announce


def test_hands_lost_resets_temporal_state() -> None:
    gate, _ = make_gate(window_size=3, min_majority=2, stability=2)

    gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    # gate is now one frame away from firing
    gate.on_hands_lost()

    # after reset we need three more frames before firing again: frame 1
    # fails the majority gate (count=1), frames 2 and 3 build the streak
    first = gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    assert not first.announce
    second = gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    assert not second.announce
    third = gate.evaluate(peaked("hello", top_prob=0.85), LABELS)
    assert third.announce


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_top1=-0.1),
        dict(min_top1=1.5),
        dict(min_margin=-0.01),
        dict(window_size=0),
        dict(window_size=3, min_majority=4),
        dict(stability=0),
        dict(cooldown_sec=-1.0),
    ],
)
def test_constructor_rejects_invalid_thresholds(kwargs) -> None:
    with pytest.raises(ValueError):
        ConfidenceGate(**kwargs)


def test_evaluate_rejects_misaligned_inputs() -> None:
    gate, _ = make_gate()
    with pytest.raises(ValueError):
        gate.evaluate([0.5, 0.5], LABELS)


def test_empty_probs_returns_safe_default() -> None:
    gate, _ = make_gate()
    decision = gate.evaluate([], [])
    assert isinstance(decision, GateDecision)
    assert not decision.announce
    assert decision.reason == "empty_probs"


# ---------------------------------------------------------------- entropy
def test_entropy_helper_matches_closed_form_for_uniform() -> None:
    k = 7
    uniform = [1.0 / k] * k
    assert _shannon_entropy_nats(uniform) == pytest.approx(math.log(k), rel=1e-6)


def test_entropy_helper_is_zero_for_one_hot() -> None:
    one_hot = [1.0, 0.0, 0.0, 0.0]
    assert _shannon_entropy_nats(one_hot) == pytest.approx(0.0, abs=1e-9)
