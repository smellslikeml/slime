"""Test-internal helpers for OPD e2e coverage."""

import torch

from slime.utils.types import Sample


def post_process_megatron_server_opd_rewards(args, samples: list[Sample], **kwargs):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = []
    for reward, response_length in zip(raw_rewards, response_lengths, strict=False):
        log_probs = torch.tensor(reward["log_probs"], dtype=torch.float32)
        if response_length == 0:
            teacher_log_probs.append(log_probs[:0])
        else:
            teacher_log_probs.append(log_probs[-response_length:])

    for sample, log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = log_probs

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
