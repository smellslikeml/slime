# Cumulative Prefix-divergence Policy Optimization (CPPO) token-level masking.
#
# Adapted from "Beyond Uniform Token-Level Trust Region in LLM Reinforcement
# Learning" (CPPO, https://arxiv.org/abs/2606.10968).
#
# Standard PPO-style trust regions are position-agnostic: the same clip width is
# applied to every token independently. CPPO replaces that uniform token-level
# trust region with a masking rule built from two coupled mechanisms:
#
#   1. Position-weighted threshold -- early tokens, whose effects persist longer
#      under autoregressive generation, get a stricter divergence allowance;
#      late tokens get a relaxed one.
#   2. Cumulative prefix budget -- as a prefix accumulates divergence from the
#      rollout policy, the remaining allowance for further tokens shrinks,
#      preventing compounding drift along the sequence.
#
# The output is a per-token {0, 1} mask that multiplies the policy-gradient loss
# exactly like the existing OPSM mask in ``ppo_utils.compute_opsm_mask`` -- it
# keeps the uniform clip path as default and only gates which tokens contribute.

from argparse import Namespace

import torch


def compute_cppo_mask(
    args: Namespace,
    ppo_kl: torch.Tensor,
    segment_sizes: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the CPPO token-level keep-mask.

    Args:
        args: Configuration providing ``eps_clip`` (base trust-region width),
            ``cppo_early_strictness``, ``cppo_late_relax`` and
            ``cppo_prefix_budget``.
        ppo_kl: Concatenated per-token log-ratio ``old_log_prob - log_prob`` over
            all response tokens in the mini-batch (same tensor passed to
            ``compute_policy_loss``).
        segment_sizes: Per-sample response token counts; ``sum(segment_sizes)``
            must equal ``ppo_kl.numel()``. Used to recover per-sequence position
            so the prefix budget resets at each sequence boundary.

    Returns:
        Tuple ``(cppo_mask, cppo_clipfrac)`` where ``cppo_mask`` is a detached
        float tensor shaped like ``ppo_kl`` (1.0 = keep, 0.0 = drop) and
        ``cppo_clipfrac`` is the scalar fraction of tokens dropped.

    Note:
        Positions are read off each local segment, so they are exact when
        context parallelism is disabled (``cp_size == 1``). Under context
        parallelism a sequence is sharded across ranks and the per-shard
        position is an approximation of the global position.
    """
    eps_clip = args.eps_clip
    early = args.cppo_early_strictness
    late = args.cppo_late_relax
    budget = max(eps_clip * args.cppo_prefix_budget, 1e-8)

    with torch.no_grad():
        device = ppo_kl.device
        # Token-level divergence magnitude (|log-ratio|).
        divergence = ppo_kl.detach().abs()
        mask_segments = []

        for segment in torch.split(divergence, segment_sizes):
            length = segment.numel()
            if length == 0:
                mask_segments.append(segment.new_ones(0))
                continue

            positions = torch.arange(length, device=device, dtype=segment.dtype)
            # Position-weighted threshold: scales eps_clip from `early` at the
            # first token to `late` at the last token (monotonic in position).
            frac = positions / max(length - 1, 1)
            threshold = eps_clip * (early + (late - early) * frac)

            # Cumulative prefix budget: average divergence accumulated strictly
            # before the current token. As that average approaches `budget`, the
            # effective threshold is shrunk toward zero, masking later tokens.
            prefix_excl = torch.cumsum(segment, dim=0) - segment
            avg_prefix_drift = prefix_excl / positions.clamp(min=1.0)
            shrink = (1.0 - avg_prefix_drift / budget).clamp(min=0.0)
            effective_threshold = threshold * shrink

            mask_segments.append((segment <= effective_threshold).to(segment.dtype))

        cppo_mask = torch.cat(mask_segments, dim=0)
        cppo_clipfrac = (1.0 - cppo_mask).mean() if cppo_mask.numel() > 0 else torch.tensor(0.0, device=device)

    return cppo_mask, cppo_clipfrac
