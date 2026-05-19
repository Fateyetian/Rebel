"""GiGPO baseline (NeurIPS 2025) — two-level grouping with observation-hash step grouping."""

from .core_gigpo import (
    episode_norm_reward,
    step_norm_reward,
    compute_gigpo_outcome_advantage,
)

__all__ = [
    'episode_norm_reward',
    'step_norm_reward',
    'compute_gigpo_outcome_advantage',
]
