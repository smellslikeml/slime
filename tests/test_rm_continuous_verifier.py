"""CPU unit tests for the ``llm_verifier`` continuous reward path.

Covers ``slime.rollout.rm_hub.continuous_verifier`` (the expectation-over-
scoring-token-logits scorer adapted from LLM-as-a-Verifier) and, crucially,
the wiring edit in ``slime.rollout.rm_hub.async_rm`` that dispatches
``rm_type == "llm_verifier"`` to it. The scorer is pure math (softmax,
expectation, normalisation, aggregation); the integration test monkeypatches
the HTTP fetch so no verifier service is required, then drives the real
``async_rm`` dispatch and asserts the reward that lands on the sample.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout.rm_hub import async_rm, continuous_verifier
from slime.rollout.rm_hub.continuous_verifier import (
    aggregate_criteria,
    aggregate_repeated,
    continuous_reward,
    expected_score,
)
from slime.utils.types import Sample


NUM_GPUS = 0


# ---------------------------------------------------------------------------
# expected_score — the core expectation over scoring-token logits
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_expected_score_is_calibrated_to_unit_interval():
    """A distribution peaked on the max token → ~1.0; on the min → ~0.0."""
    assert expected_score({"0": -5.0, "10": 5.0}) > 0.95
    assert expected_score({"0": 5.0, "10": -5.0}) < 0.05


@pytest.mark.unit
def test_expected_score_is_monotonic_in_the_top_token_mass():
    """Shifting logit mass toward the higher token strictly raises the score
    — this is the continuous signal a discrete argmax judge would flatten."""
    low = expected_score({"0": 0.0, "5": 1.0, "10": 0.0})
    high = expected_score({"0": 0.0, "5": 0.0, "10": 1.0})
    assert 0.0 <= low < high <= 1.0


@pytest.mark.unit
def test_expected_score_respects_explicit_value_map():
    """Non-numeric scoring tokens are usable via an explicit value map;
    tokens absent from the map are ignored."""
    score = expected_score(
        {"bad": 2.0, "good": -2.0, "ignore": 10.0},
        value_map={"bad": 0.0, "good": 1.0},
    )
    # Mass sits on "bad" (value 0) → score near 0 despite the huge "ignore" logit.
    assert score < 0.2


@pytest.mark.unit
def test_expected_score_raises_without_numeric_tokens():
    with pytest.raises(ValueError):
        expected_score({"foo": 1.0, "bar": 2.0})


# ---------------------------------------------------------------------------
# aggregation — repeated evaluation (variance) and criteria decomposition
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aggregate_repeated_is_the_mean():
    assert aggregate_repeated([0.2, 0.4, 0.6]) == pytest.approx(0.4)


@pytest.mark.unit
def test_aggregate_criteria_weights_emphasise_criteria():
    """Weighting correctness (1.0) above style (0.0) pulls the combined score
    up relative to the unweighted mean of 0.5."""
    assert aggregate_criteria([1.0, 0.0]) == pytest.approx(0.5)
    assert aggregate_criteria([1.0, 0.0], weights=[3.0, 1.0]) == pytest.approx(0.75)


@pytest.mark.unit
def test_aggregate_criteria_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        aggregate_criteria([1.0, 0.0], weights=[1.0])


# ---------------------------------------------------------------------------
# continuous_reward — payload-shape normalisation + full composition
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        {"0": -5.0, "10": 5.0},  # bare logprob dict
        {"score_logprobs": {"0": -5.0, "10": 5.0}},  # single criterion wrapper
        {"criteria": [{"0": -5.0, "10": 5.0}]},  # decomposed (one criterion)
        {"evaluations": [[{"0": -5.0, "10": 5.0}]]},  # repeated eval (one repeat)
        [{"0": -5.0, "10": 5.0}],  # bare list of repeats
    ],
)
def test_continuous_reward_accepts_all_payload_shapes(payload):
    assert continuous_reward(payload) == pytest.approx(expected_score({"0": -5.0, "10": 5.0}))


@pytest.mark.unit
def test_continuous_reward_composes_repeats_and_criteria():
    payload = {
        "evaluations": [
            [{"0": 0.0, "10": 4.0}, {"0": 4.0, "10": 0.0}],
            [{"0": 0.0, "10": 4.0}, {"0": 4.0, "10": 0.0}],
        ]
    }
    per_criterion = [expected_score({"0": 0.0, "10": 4.0}), expected_score({"0": 4.0, "10": 0.0})]
    expected = aggregate_repeated([aggregate_criteria(per_criterion)] * 2)
    assert continuous_reward(payload) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# integration — the async_rm dispatch edit routes to the verifier
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_async_rm_dispatches_llm_verifier(monkeypatch):
    """Exercises the wiring in ``rm_hub/__init__.py``: ``rm_type ==
    "llm_verifier"`` must reach ``continuous_verifier_reward`` and return its
    continuous score straight into the reward contract async_rm emits."""
    payload = {"evaluations": [[{"0": 0.0, "5": 0.0, "10": 2.0}], [{"0": 0.0, "5": 0.0, "10": 1.5}]]}

    async def fake_fetch(args, sample):
        return payload

    monkeypatch.setattr(continuous_verifier, "_fetch_verifier_payload", fake_fetch)

    args = SimpleNamespace(rm_type="llm_verifier", custom_rm_path=None, rm_url="http://verifier.local")
    sample = Sample(prompt="q", response="a", label="gt")

    reward = asyncio.run(async_rm(args, sample))

    assert reward == pytest.approx(continuous_reward(payload))
    assert 0.0 <= reward <= 1.0


@pytest.mark.unit
def test_async_rm_reads_per_sample_metadata_overrides(monkeypatch):
    """Per-sample metadata selects the rm_type and supplies scoring config
    (value map / criterion weights) without any new CLI flag."""
    payload = {"criteria": [{"bad": 3.0, "good": -3.0}, {"bad": -3.0, "good": 3.0}]}

    async def fake_fetch(args, sample):
        return payload

    monkeypatch.setattr(continuous_verifier, "_fetch_verifier_payload", fake_fetch)

    args = SimpleNamespace(rm_type=None, custom_rm_path=None, rm_url="http://verifier.local")
    sample = Sample(
        prompt="q",
        response="a",
        label="gt",
        metadata={
            "rm_type": "llm_verifier",
            "verifier_value_map": {"bad": 0.0, "good": 1.0},
            "verifier_criteria_weights": [3.0, 1.0],
        },
    )

    reward = asyncio.run(async_rm(args, sample))

    expected = continuous_reward(
        payload,
        value_map={"bad": 0.0, "good": 1.0},
        criteria_weights=[3.0, 1.0],
    )
    assert reward == pytest.approx(expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
