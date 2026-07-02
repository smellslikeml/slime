"""CPU tests for the CPPO token-level mask and its policy-loss wiring.

CPPO ("Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning",
https://arxiv.org/abs/2606.10968) replaces the uniform token-level trust region
with a position-weighted threshold plus a cumulative prefix budget. The mask is
applied to ``compute_policy_loss``'s output exactly as ``policy_loss_function``
wires it (``pg_loss = pg_loss * cppo_mask``), so these tests pin both the mask
semantics and that integration contract against the existing (non-new)
``slime.utils.ppo_utils`` loss.
"""

from argparse import Namespace

import pytest
import torch

from slime.utils.cumulative_prefix_mask import compute_cppo_mask
from slime.utils.ppo_utils import compute_policy_loss

NUM_GPUS = 0


def _args(early=0.5, late=1.5, prefix_budget=4.0, eps_clip=0.2):
    return Namespace(
        eps_clip=eps_clip,
        cppo_early_strictness=early,
        cppo_late_relax=late,
        cppo_prefix_budget=prefix_budget,
    )


def test_position_weighting_is_stricter_early_than_late():
    # Flatten the prefix-budget effect (huge budget => shrink ~= 1) so the keep
    # decision is driven purely by the position-weighted threshold.
    args = _args(early=0.5, late=1.5, prefix_budget=1e6)
    # Identical divergence 0.2 at both positions.
    ppo_kl = torch.tensor([0.2, -0.2])

    mask, clipfrac = compute_cppo_mask(args, ppo_kl, segment_sizes=[2])

    # t=0 threshold = eps_clip*0.5 = 0.10 < 0.2 -> dropped (stricter early);
    # t=1 threshold = eps_clip*1.5 = 0.30 > 0.2 -> kept (relaxed late).
    torch.testing.assert_close(mask, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(clipfrac, torch.tensor(0.5))


def test_prefix_budget_masks_low_divergence_token_after_drift():
    # Flat position weight isolates the cumulative-prefix-budget mechanism.
    args = _args(early=1.0, late=1.0, prefix_budget=4.0)  # budget = eps_clip*4 = 0.8
    ppo_kl = torch.tensor([0.05, 1.0, 1.0, 0.05])

    mask, _ = compute_cppo_mask(args, ppo_kl, segment_sizes=[4])

    # The final token's own divergence (0.05) is below the base threshold (0.2),
    # yet it is dropped because the prefix has already exhausted its budget.
    torch.testing.assert_close(mask, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_identical_token_kept_without_prefix_drift():
    # The same low-divergence token that the budget masked above is kept when no
    # prior drift has accumulated -- confirming the masking is prefix-dependent.
    args = _args(early=1.0, late=1.0, prefix_budget=4.0)
    mask, _ = compute_cppo_mask(args, torch.tensor([0.05]), segment_sizes=[1])
    torch.testing.assert_close(mask, torch.tensor([1.0]))


def test_prefix_budget_resets_per_sequence():
    args = _args(early=1.0, late=1.0, prefix_budget=4.0)
    # Two sequences concatenated; each starts with a low-divergence token that
    # must be kept, proving the prefix sum resets at the segment boundary.
    ppo_kl = torch.tensor([0.05, 1.0, 0.05, 1.0])

    mask, _ = compute_cppo_mask(args, ppo_kl, segment_sizes=[2, 2])

    torch.testing.assert_close(mask, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_empty_batch_returns_empty_mask():
    mask, clipfrac = compute_cppo_mask(_args(), torch.empty(0), segment_sizes=[0])
    assert mask.numel() == 0
    torch.testing.assert_close(clipfrac, torch.tensor(0.0))


def test_wires_into_compute_policy_loss_zeroing_dropped_tokens():
    # Reproduce exactly what policy_loss_function does: compute the uniform PPO
    # loss with the existing (non-new) estimator, then multiply by the CPPO mask.
    args = _args(early=0.5, late=1.5, prefix_budget=1e6)
    ppo_kl = torch.tensor([0.2, -0.2])
    advantages = torch.tensor([1.0, 1.0])

    pg_loss, _ = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip)
    cppo_mask, cppo_clipfrac = compute_cppo_mask(args, ppo_kl, segment_sizes=[2])
    masked_pg_loss = pg_loss * cppo_mask

    # Dropped token (index 0) contributes no gradient; kept token is untouched.
    assert masked_pg_loss[0].item() == 0.0
    torch.testing.assert_close(masked_pg_loss[1], pg_loss[1])
    torch.testing.assert_close(cppo_clipfrac, torch.tensor(0.5))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
