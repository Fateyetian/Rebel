"""
ReBel — Rewarding Beliefs, Not Actions.

Core components:
1. HiBO (Hierarchical Belief-Observation Grouping): obs-hash primary + belief-abstract fallback
2. Competence-adaptive belief reward curriculum: SR-based decay + differential component decay
3. Structured <belief>/<think>/<action> prompting
4. Dual-layer advantage: A_total = A_episode + ω · A_step
"""

from .core_rebel import (
    compute_rebel_advantage,
    episode_norm_reward,
    step_norm_reward_by_belief,
    build_belief_group,
    canonicalize_belief,
    consistency_reward,
    progress_reward,
    exploration_reward,
    format_reward,
    compute_intrinsic_reward,
    print_rebel_summary,
    health_check,
)


__all__ = [
    'compute_rebel_advantage',
    'episode_norm_reward',
    'step_norm_reward_by_belief',
    'build_belief_group',
    'canonicalize_belief',
    'consistency_reward',
    'progress_reward',
    'exploration_reward',
    'format_reward',
    'compute_intrinsic_reward',
    'print_rebel_summary',
    'health_check',
]
