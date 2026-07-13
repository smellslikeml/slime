"""Continuous verification scoring for reward models.

Adapted from *LLM-as-a-Verifier: A General-Purpose Verification Framework*
(https://arxiv.org/abs/2607.05391v1).

Standard LM judges prompt a model to emit a single discrete score token
("give a rating from 0 to 10"), which throws away all of the probability
mass the model placed on the neighbouring scores. The paper's core insight
is to instead treat the score as a random variable and compute the
*expectation over the distribution of scoring-token logits*, yielding a
fine-grained continuous score. That single change unlocks three scaling
axes, all implemented here:

  1. score granularity   -- more scoring bins => sharper positive/negative
                            separation (``expected_score``);
  2. repeated evaluation -- averaging several sampled distributions reduces
                            variance (``aggregate_repeated``);
  3. criteria decomposition -- scoring sub-criteria independently and
                            combining reduces per-call complexity
                            (``aggregate_criteria``).

This module owns that math at full fidelity. The *auxiliary* component --
the specific verifier LLM and how it is served -- is substituted with
slime's existing remote reward-model HTTP path (``--rm-url``): the endpoint
returns per-criterion logprobs over the scoring tokens, and everything
downstream (softmax, expectation, normalisation, aggregation) happens here.
The paper's candidate-ranking algorithm, benchmark suites, and Claude Code
monitoring extension are intentionally out of scope for a per-sample reward
signal and are left to downstream work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slime.utils.types import Sample

# The endpoint may return either a single logprob dict, a list of repeats, or
# a decomposed {criterion: logprobs} structure. These keys select the shape.
_EVALUATIONS_KEY = "evaluations"
_CRITERIA_KEY = "criteria"
_LOGPROBS_KEY = "score_logprobs"


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a list of logits/logprobs."""
    if not logits:
        return []
    hi = max(logits)
    exps = [math.exp(x - hi) for x in logits]
    total = sum(exps)
    if total == 0.0:
        # Degenerate input; fall back to a uniform distribution.
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


def _token_value(token: str, value_map: Mapping[str, float] | None) -> float | None:
    """Resolve the numeric value a scoring token stands for.

    With an explicit ``value_map`` the token is looked up directly; otherwise
    numeric-looking tokens ("0", "7", "0.5") are parsed as their own value.
    Non-scoring tokens return ``None`` and are ignored.
    """
    if value_map is not None:
        return value_map.get(token)
    try:
        return float(token.strip())
    except (ValueError, AttributeError):
        return None


def expected_score(
    token_logprobs: Mapping[str, float],
    *,
    value_map: Mapping[str, float] | None = None,
) -> float:
    """Continuous score = expectation over the scoring-token distribution.

    ``token_logprobs`` maps each candidate scoring token to the logit/logprob
    the verifier assigned it. Tokens are softmax-normalised into a probability
    distribution, the expected numeric value is taken, and the result is
    rescaled to ``[0, 1]`` using the observed value range so callers get a
    calibrated reward regardless of the raw score granularity.
    """
    values: list[float] = []
    logits: list[float] = []
    for token, logprob in token_logprobs.items():
        value = _token_value(token, value_map)
        if value is None:
            continue
        values.append(value)
        logits.append(float(logprob))

    if not values:
        raise ValueError("continuous verifier received no numeric scoring tokens; pass a value_map or emit numeric token keys")

    probs = _softmax(logits)
    expected = sum(p * v for p, v in zip(probs, values, strict=False))

    lo, hi = min(values), max(values)
    if hi == lo:
        # Single scoring bin carries no gradient; report its mass directly.
        return sum(probs)
    return (expected - lo) / (hi - lo)


def aggregate_repeated(scores: list[float]) -> float:
    """Reduce variance by averaging repeated evaluations (paper axis 2)."""
    if not scores:
        raise ValueError("aggregate_repeated requires at least one score")
    return sum(scores) / len(scores)


def aggregate_criteria(
    scores: list[float],
    weights: list[float] | None = None,
) -> float:
    """Combine decomposed per-criterion scores (paper axis 3).

    Defaults to an unweighted mean; ``weights`` allows emphasising criteria
    (e.g. correctness over style). A single criterion degrades to identity.
    """
    if not scores:
        raise ValueError("aggregate_criteria requires at least one score")
    if weights is None:
        return sum(scores) / len(scores)
    if len(weights) != len(scores):
        raise ValueError("weights length must match the number of criteria")
    total = sum(weights)
    if total == 0.0:
        raise ValueError("criterion weights must not sum to zero")
    return sum(s * w for s, w in zip(scores, weights, strict=False)) / total


def _normalize_evaluations(payload) -> list[list[Mapping[str, float]]]:
    """Coerce a verifier response into ``[repeat][criterion] -> logprob dict``.

    Accepts several shapes so the endpoint stays simple:
      * ``{"0": .., "1": ..}``                       -> one repeat, one criterion
      * ``{"score_logprobs": {..}}``                 -> one repeat, one criterion
      * ``{"criteria": [{..}, {..}]}``               -> one repeat, N criteria
      * ``{"evaluations": [<repeat>, <repeat>]}``    -> N repeats (each above)
      * a bare ``list`` is treated as ``evaluations``
    """

    def as_criteria(repeat) -> list[Mapping[str, float]]:
        if isinstance(repeat, Mapping):
            if _CRITERIA_KEY in repeat:
                return as_criteria(repeat[_CRITERIA_KEY])
            if _LOGPROBS_KEY in repeat:
                return [repeat[_LOGPROBS_KEY]]
            return [repeat]
        if isinstance(repeat, list):
            return [c[_LOGPROBS_KEY] if isinstance(c, Mapping) and _LOGPROBS_KEY in c else c for c in repeat]
        raise ValueError(f"unsupported verifier repeat shape: {type(repeat).__name__}")

    if isinstance(payload, Mapping) and _EVALUATIONS_KEY in payload:
        repeats = payload[_EVALUATIONS_KEY]
    elif isinstance(payload, list):
        repeats = payload
    else:
        repeats = [payload]

    return [as_criteria(repeat) for repeat in repeats]


def continuous_reward(
    payload,
    *,
    value_map: Mapping[str, float] | None = None,
    criteria_weights: list[float] | None = None,
) -> float:
    """Turn a verifier logprob payload into one continuous reward in ``[0, 1]``.

    Composes the three scaling axes: each criterion's scoring-token
    distribution becomes an ``expected_score``, criteria are combined with
    ``aggregate_criteria``, and repeated evaluations are averaged with
    ``aggregate_repeated``.
    """
    evaluations = _normalize_evaluations(payload)
    repeat_scores = [
        aggregate_criteria(
            [expected_score(criterion, value_map=value_map) for criterion in criteria],
            weights=criteria_weights,
        )
        for criteria in evaluations
    ]
    return aggregate_repeated(repeat_scores)


async def _fetch_verifier_payload(args, sample: Sample):
    """POST the (prompt, response) to the verifier endpoint and return its JSON.

    Reuses the shared aiohttp session that ``remote_rm`` already manages, so
    no new serving infrastructure is introduced. The import is local to avoid
    a circular import with the package ``__init__``.
    """
    if getattr(args, "rm_url", None) is None:
        raise ValueError("llm_verifier rm_type requires --rm-url to be set")

    from . import _get_shared_session

    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
        "return_score_logprobs": True,
    }
    session = _get_shared_session()
    async with session.post(args.rm_url, json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


async def continuous_verifier_reward(args, sample: Sample, **kwargs) -> float:
    """Reward hook: fetch a scoring-token distribution and return its expectation.

    Wired into ``async_rm`` under ``rm_type == "llm_verifier"``. Per-sample
    scoring configuration (value map, criterion weights) may be supplied via
    ``sample.metadata`` without adding new CLI flags.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value_map = metadata.get("verifier_value_map")
    criteria_weights = metadata.get("verifier_criteria_weights")

    payload = await _fetch_verifier_payload(args, sample)
    return continuous_reward(payload, value_map=value_map, criteria_weights=criteria_weights)
