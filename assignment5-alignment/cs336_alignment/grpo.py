import torch
from typing import Callable, Literal


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
    return advantages.reshape(-1), raw_rewards.reshape(-1), metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,    # (b, 1)
    policy_log_probs: torch.Tensor,             # (b, seq_len)
) -> torch.Tensor:
    return - raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,           # (b, 1)
    policy_log_probs: torch.Tensor,     # (b ,seq_len)
    old_log_probs: torch.Tensor,        # (b, seq_len)
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # (loss, metadata)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

    metadata = {
        "policy_log_probs": policy_log_probs,
        "old_log_probs": old_log_probs
    }
    return - torch.minimum(ratio * advantages, clipped_ratio * advantages), metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,     # (b, seq_len)
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,    # (b, 1) no_baseline
    advantages: torch.Tensor | None = None,     # (b, 1) reinforce_with_baseline / grpo_clip
    old_log_probs: torch.Tensor | None = None,  # (b, seq_len) grpo_clip
    cliprange: float | None = None,             # grpo_clip
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]

    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None and raw_rewards.shape[0] == policy_log_probs.shape[0]
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), metadata
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None and advantages.shape[0] == policy_log_probs.shape[0]
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), metadata
    elif loss_type == "grpo_clip":
        assert (advantages is not None 
                and advantages.shape[0] == policy_log_probs.shape[0] 
                and old_log_probs is not None 
                and old_log_probs.shape == policy_log_probs.shape
                and cliprange is not None)
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)