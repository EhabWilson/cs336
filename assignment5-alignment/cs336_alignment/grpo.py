import torch
from typing import Callable


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],  # "reward", "format_reward", and "answer_reward"
    rollout_responses: list[str],  # rollout_bs = n_prompts * group_size
    repeated_ground_truths: list[str],  # rollout_bs
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:   # (advantages, raw_rewards, metadata)
    n = len(rollout_responses)

    raw_rewards = []
    for rollout, gt in zip(rollout_responses, repeated_ground_truths):
        r = reward_fn(rollout, gt)
        raw_rewards.append(r["reward"])
    raw_rewards = torch.Tensor(raw_rewards)

    raw_rewards = raw_rewards.reshape(-1, group_size)
    advantages = raw_rewards - raw_rewards.mean(dim=-1, keepdim=True)
    if normalize_by_std:
        advantages = advantages / (raw_rewards.std(dim=-1, keepdim=True) + advantage_eps)
    
    metadata = {}
    return (advantages.reshape(-1), raw_rewards.reshape(-1), metadata)