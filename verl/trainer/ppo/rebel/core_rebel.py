"""
ReBel (Reward Belief) Framework - Core Algorithm

严格按照 ReBel_RL_Algorithm_Guide.md 实现

核心创新:
1. Belief-based Grouping: 按(uid, belief_hash)分组，粒度适中(20-100组)
2. 4组件内在奖励: Consistency + Progress + Exploration (Format是独立惩罚)
3. 双层优势: A_total = A_episode + λ × A_step

公式:
- A_episode: 按uid分组归一化 (与GiGPO相同)
- A_step: 按(uid, belief_hash)分组归一化 (ReBel核心)
"""

import numpy as np
import torch
import json
import hashlib
import re
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional


# ============================================================================ #
# ===================== Belief Canonicalization ============================== #
# ============================================================================ #

def _get_stage_type_v6(subgoal: str) -> str:
    """
    V6改进: 正确优先级的阶段类型检测

    关键修复: 'find' 必须优先于 'lamp'/'light' 等词
    原因: "find lamp" 应该分类为 'find' 而非 'use'

    优先级顺序:
    1. complete - 完成状态
    2. find - 搜索阶段 (必须优先于 lamp/light)
    3. navigate - 导航阶段
    4. pickup - 拾取阶段
    5. place - 放置阶段
    6. heat/cool/clean - 状态改变阶段
    7. use - 使用阶段 (放在后面避免误匹配)
    8. interact - 交互阶段
    """
    subgoal = subgoal.lower().strip()

    # 优先级1: 完成状态
    if any(x in subgoal for x in ['complete', 'done', 'finished', 'success']):
        return 'complete'

    # 优先级2: 搜索阶段 (必须优先于 'lamp'/'light' 等词！)
    if any(x in subgoal for x in ['find', 'look for', 'search', 'locate']):
        return 'find'

    # 优先级3: 导航阶段
    if any(x in subgoal for x in ['go to', 'goto', 'navigate', 'move to']):
        return 'navigate'

    # 优先级4: 拾取阶段
    if any(x in subgoal for x in ['pick up', 'pick', 'take', 'grab']):
        return 'pickup'

    # 优先级5: 放置阶段
    if any(x in subgoal for x in ['put', 'place', 'drop']):
        return 'place'

    # 优先级6: 状态改变阶段 (heat/cool/clean)
    if any(x in subgoal for x in ['heat', 'cook', 'warm', 'microwave']):
        return 'heat'
    if any(x in subgoal for x in ['cool', 'chill', 'fridge', 'refrigerat']):
        return 'cool'
    if any(x in subgoal for x in ['clean', 'wash', 'rinse', 'sink']):
        return 'clean'

    # 优先级7: 使用阶段 (放在后面，避免 'lamp' 等词误匹配)
    if any(x in subgoal for x in ['turn on', 'use', 'toggle', 'examine']):
        return 'use'

    # 优先级8: 交互阶段
    if any(x in subgoal for x in ['open', 'close']):
        return 'interact'

    return 'other'


def canonicalize_belief(belief_state: Dict[str, Any], granularity: str = 'subgoal', gt_phase: Optional[str] = None) -> str:
    """
    将belief state转换为可哈希的规范表示

    Args:
        belief_state: 解析后的belief state字典
        granularity: 粒度级别
            - 'subgoal': 只用subgoal + status (推荐，20-50组)
            - 'medium': + found_objects统计 (50-150组)
            - 'fine': 完整belief state (100-500组)

    Returns:
        belief_hash: 16位MD5哈希字符串
    """
    if belief_state is None or not isinstance(belief_state, dict):
        return "null_belief_0000"

    try:
        task = belief_state.get('task', {}) or {}
        state = belief_state.get('state', {}) or {}

        if granularity in ('subgoal', 'default'):
            # Primary: code-extracted coarse key per paper §4.3
            # Uses task.phase (env-supervised) + task.target (stable across episode)
            canonical = {
                'phase': str(task.get('phase', '')).lower().strip(),
                'target': str(task.get('target', '')).lower().strip(),
            }

        elif granularity == 'medium':
            # + holding flag derived from state.objects
            objects = state.get('objects', {}) or {}
            has_inventory = any(
                'in_hand' in str(v).lower()
                for v in objects.values()
            ) if isinstance(objects, dict) else False

            canonical = {
                'phase': str(task.get('phase', '')).lower().strip(),
                'target': str(task.get('target', '')).lower().strip(),
                'has_inventory': has_inventory,
            }

        elif granularity == 'task_status':
            # Structured flags only (100% coverage, no text fragility)
            objects = state.get('objects', {}) or {}
            states = state.get('states', {}) or {}
            has_inventory = any(
                'in_hand' in str(v).lower()
                for v in objects.values()
            ) if isinstance(objects, dict) else False
            has_state_change = len(states) > 0 if isinstance(states, dict) else False
            phase = str(task.get('phase', '')).lower().strip()
            is_complete = phase in ('done', 'buying')

            canonical = {
                'is_complete': is_complete,
                'has_state_change': has_state_change,
                'has_inventory': has_inventory,
            }

        elif granularity == 'state_aware':
            states = state.get('states', {}) or {}
            state_types = set()
            if isinstance(states, dict):
                for obj, st in states.items():
                    st_lower = str(st).lower()
                    if any(x in st_lower for x in ['heated', 'hot', 'warm', 'cooked']):
                        state_types.add('heated')
                    elif any(x in st_lower for x in ['cooled', 'cold', 'chilled']):
                        state_types.add('cooled')
                    elif any(x in st_lower for x in ['cleaned', 'clean', 'washed']):
                        state_types.add('cleaned')

            canonical = {
                'phase': str(task.get('phase', '')).lower().strip(),
                'target': str(task.get('target', '')).lower().strip(),
                'state_types': sorted(state_types),
            }

        elif granularity == 'adaptive':
            # phase + target + holding + transform_ready flags
            objects = state.get('objects', {}) or {}
            states_map = state.get('states', {}) or {}
            has_inventory = any(
                'in_hand' in str(v).lower()
                for v in objects.values()
            ) if isinstance(objects, dict) else False
            has_state_change = len(states_map) > 0 if isinstance(states_map, dict) else False
            phase = str(task.get('phase', '')).lower().strip()
            stage_type = _get_stage_type_v6(str(task.get('subgoal', '')).lower())

            canonical = {
                'phase': phase,
                'target': str(task.get('target', '')).lower().strip(),
                'has_inventory': has_inventory,
                'has_state_change': has_state_change,
                'stage_type': stage_type,
            }

        elif granularity == 'gt_phase':
            # Hybrid: stable env-derived phase anchor + belief structural flags
            # env_phase comes from admissible_commands (ALFWorld) or page markers (WebShop)
            # Falls back to belief task.phase when gt_phase not supplied
            phase_anchor = gt_phase if gt_phase is not None else str(task.get('phase', 'unknown')).lower().strip()
            objects = state.get('objects', {}) or {}
            states_map = state.get('states', {}) or {}
            has_inventory = any(
                'in_hand' in str(v).lower()
                for v in objects.values()
            ) if isinstance(objects, dict) else False
            has_state_change = len(states_map) > 0 if isinstance(states_map, dict) else False

            # Normalize target: strip articles/punctuation so
            # "the apple", "Apple", "an apple" all map to "apple"
            # BUT keep instance numbers: "apple 1" != "apple 2" (different game objects)
            raw_target = str(task.get('target', '') or '').lower().strip()
            raw_target = re.sub(r'\b(a|an|the)\b', '', raw_target)   # remove articles
            raw_target = re.sub(r'[^a-z0-9 ]', '', raw_target)       # remove punctuation (keep digits)
            normalized_target = ' '.join(raw_target.split())          # collapse whitespace

            canonical = {
                'env_phase': phase_anchor,
                'target': normalized_target,
                'has_inventory': has_inventory,
                'has_state_change': has_state_change,
            }

        elif granularity == 'fine':
            canonical = belief_state

        else:
            canonical = {
                'phase': str(task.get('phase', '')).lower().strip(),
                'target': str(task.get('target', '')).lower().strip(),
            }

        # 生成哈希
        canonical_str = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
        belief_hash = hashlib.md5(canonical_str.encode()).hexdigest()[:16]
        return belief_hash

    except Exception as e:
        return "error_belief_000"


# ============================================================================ #
# ====================== Belief Group Building =============================== #
# ============================================================================ #

def build_belief_group(
    belief_states: np.ndarray,
    index: np.ndarray,
    granularity: str = 'subgoal',
    summarize: bool = False,
    task_types: Optional[np.ndarray] = None,
    task_aware: bool = False,
    gt_phases: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    按belief相似性将steps分组

    分组策略:
    - 默认: G = {j : belief_hash_j = belief_hash_i, uid_j = uid_i}
    - 任务感知(V2): G = {j : belief_hash_j = belief_hash_i, uid_j = uid_i, task_j = task_i}

    Args:
        belief_states: shape (batch_size,), 每个元素是belief字典
        index: shape (batch_size,), 每个step的prompt uid
        granularity: canonicalization粒度
        summarize: 是否打印分组统计
        task_types: shape (batch_size,), 每个step的任务类型 (V2新增)
        task_aware: 是否启用任务感知分组 (V2新增)

    Returns:
        belief_group_uids: shape (batch_size,), 每个step的group uid
        group_stats: 分组统计信息 (包含用于logging的metrics)
    """
    belief_group_uids = np.empty(len(belief_states), dtype=object)
    unique_indices = np.unique(index)
    group_sizes = []
    all_group_uids = set()
    task_group_counts = defaultdict(int)  # 每个任务类型的组数

    for uid in unique_indices:
        # 1. 获取该uid的所有steps
        step_indices = np.where(index == uid)[0]
        beliefs = belief_states[step_indices]

        # 获取任务类型 (如果启用任务感知)
        if task_aware and task_types is not None:
            tasks = task_types[step_indices]
        else:
            tasks = None

        # 2. 按belief hash聚类 (可选: 加入任务类型)
        clusters = defaultdict(list)
        for i, belief in enumerate(beliefs):
            gt_phase_i = gt_phases[step_indices[i]] if gt_phases is not None else None
            belief_hash = canonicalize_belief(belief, granularity, gt_phase=gt_phase_i)

            if task_aware and tasks is not None:
                # V2改进: 任务感知分组 - 只在相同任务类型内分组
                task = str(tasks[i]) if tasks is not None else "unknown"
                cluster_key = (task, belief_hash)
            else:
                # 原始方式: 只按belief分组
                cluster_key = ("all", belief_hash)

            clusters[cluster_key].append(step_indices[i])

        # 3. 分配group uid
        for cluster_key, original_indices in clusters.items():
            task, belief_hash = cluster_key
            if task_aware:
                group_uid = f"belief_{task}_{uid}_{belief_hash}"
            else:
                group_uid = f"belief_{uid}_{belief_hash}"

            all_group_uids.add(group_uid)
            group_sizes.append(len(original_indices))
            task_group_counts[task] += 1

            for idx in original_indices:
                belief_group_uids[idx] = group_uid

    # 计算统计 (用于logging)
    group_stats = {
        'num_groups': len(all_group_uids),
        'group_sizes': group_sizes,
        'mean_group_size': float(np.mean(group_sizes)) if group_sizes else 0.0,
        'median_group_size': float(np.median(group_sizes)) if group_sizes else 0.0,
        'min_group_size': int(np.min(group_sizes)) if group_sizes else 0,
        'max_group_size': int(np.max(group_sizes)) if group_sizes else 0,
        'std_group_size': float(np.std(group_sizes)) if group_sizes else 0.0,
        'single_sample_groups': sum(1 for s in group_sizes if s == 1),
        'single_sample_ratio': sum(1 for s in group_sizes if s == 1) / max(len(group_sizes), 1),
        'task_aware': task_aware,
    }

    # 如果启用任务感知，添加每个任务的组数统计
    if task_aware and task_group_counts:
        group_stats['task_group_counts'] = dict(task_group_counts)
        # 计算每个任务类型的平均组大小
        for task in task_group_counts:
            if task != "all":
                group_stats[f'num_groups_{task}'] = task_group_counts[task]

    if summarize:
        task_info = ""
        if task_aware and len(task_group_counts) > 1:
            task_info = "\n├─ Task-Aware: Enabled"
            for task, count in sorted(task_group_counts.items()):
                if task != "all":
                    task_info += f"\n│  └─ {task}: {count} groups"

        print(f"""
============================================================
ReBel Belief-Based Grouping Statistics
============================================================
Total steps: {len(belief_states)}
Number of groups: {group_stats['num_groups']}
Mean group size: {group_stats['mean_group_size']:.2f}
Median group size: {group_stats['median_group_size']:.1f}
Min group size: {group_stats['min_group_size']}
Max group size: {group_stats['max_group_size']}
Std group size: {group_stats['std_group_size']:.2f}
Single-sample groups: {group_stats['single_sample_groups']} ({group_stats['single_sample_ratio']:.1%}){task_info}
============================================================
""")

    return belief_group_uids, group_stats


# ============================================================================ #
# ====================== Episode Norm Reward ================================= #
# ============================================================================ #

def _norm_scores_by_uid(
    scores: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    traj_index: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Normalize a (batch,) score tensor by task uid, with optional trajectory equal-weighting,
    then broadcast the result to (batch, seq_len) masked by eos_mask.

    When traj_index is given: deduplicate to one score per trajectory before computing uid
    mean/std — eliminates length bias where longer trajectories dominate the normalization.
    """
    response_length = eos_mask.shape[-1]
    scores = scores.clone().float()
    id2stats: Dict[Any, tuple] = {}

    with torch.no_grad():
        if traj_index is not None:
            # 1. Deduplicate: one (uid, score) per trajectory
            traj2info: Dict[Any, tuple] = {}
            for i in range(len(scores)):
                traj = traj_index[i]
                if traj not in traj2info:
                    traj2info[traj] = (index[i], scores[i])

            # 2. Group by task uid; collect equal-weight trajectory scores
            uid2traj_scores: Dict[Any, list] = defaultdict(list)
            for traj, (uid, score) in traj2info.items():
                uid2traj_scores[uid].append(score)

            # 3. Compute per-uid mean/std over trajectories (not steps)
            for uid, traj_scores in uid2traj_scores.items():
                t = torch.stack(traj_scores)
                if len(traj_scores) == 1:
                    mean, std = torch.tensor(0.0), torch.tensor(1.0)
                else:
                    mean, std = torch.mean(t), torch.std(t)
                id2stats[uid] = (mean, std)

        else:
            # Legacy: per-row aggregation (biased toward longer trajectories)
            id2list: Dict[Any, list] = defaultdict(list)
            for i in range(len(scores)):
                id2list[index[i]].append(scores[i])
            for uid, rewards in id2list.items():
                t = torch.stack(rewards) if len(rewards) > 1 else torch.stack(rewards)
                if len(rewards) == 1:
                    mean, std = torch.tensor(0.0), torch.tensor(1.0)
                else:
                    mean, std = torch.mean(t), torch.std(t)
                id2stats[uid] = (mean, std)

        # 4. Apply normalization row-wise
        for i in range(len(scores)):
            mean, std = id2stats[index[i]]
            if remove_std:
                scores[i] = scores[i] - mean
            else:
                scores[i] = (scores[i] - mean) / (std + epsilon)

        return scores.unsqueeze(-1).expand(-1, response_length) * eos_mask


def episode_norm_reward(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    traj_index: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Episode-level advantage using trajectory equal-weight normalization.

    Extracts R_ep = sum of token_level_rewards per row, then delegates to
    _norm_scores_by_uid for uid-grouped, traj-equal-weight normalization.
    """
    scores = token_level_rewards.sum(dim=-1)  # (batch,) R_ep per row
    return _norm_scores_by_uid(scores, eos_mask, index, epsilon, remove_std, traj_index)


# ============================================================================ #
# ====================== Step Norm Reward by Belief ========================== #
# ============================================================================ #

def compute_discounted_returns(
    rebel_intrinsic_rewards: torch.Tensor,
    episode_env_rewards: torch.Tensor,
    traj_index: np.ndarray,
    gamma: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    按轨迹从后向前计算折扣回报，分别返回 env 和 belief 两路，用于 A_step 分离归一化。

    R_env[t]    = γ^(T-t) * env_reward          （仅终端 env 信号的折扣传播）
    R_belief[t] = belief_reward[t] + γ * R_belief[t+1]  （belief 奖励的累计折扣回报）

    两路分别在 belief group 内归一化后等权合并：
        A_step = normalize(R_env) + normalize(R_belief)
    再由外部的 step_advantage_w 统一缩放，避免 env_reward 量级（10）淹没 belief（~0.3/步）。

    Args:
        rebel_intrinsic_rewards: (batch,) 每步信念内在奖励
        episode_env_rewards:     (batch,) 每步所属轨迹的纯环境奖励（同轨迹所有步相同）
        traj_index:              (batch,) 轨迹 uid
        gamma:                   折扣因子

    Returns:
        R_env:    (batch,) 仅 env 奖励的折扣回报
        R_belief: (batch,) 仅 belief 奖励的折扣回报
    """
    traj2steps: Dict[Any, list] = defaultdict(list)
    for i, traj in enumerate(traj_index):
        traj2steps[traj].append(i)

    R_env    = torch.zeros(len(traj_index), dtype=rebel_intrinsic_rewards.dtype,
                           device=rebel_intrinsic_rewards.device)
    R_belief = torch.zeros(len(traj_index), dtype=rebel_intrinsic_rewards.dtype,
                           device=rebel_intrinsic_rewards.device)

    for traj, step_indices in traj2steps.items():
        step_indices.sort()
        terminal_idx = step_indices[-1]
        env_r = episode_env_rewards[step_indices[0]].item()  # same for all steps in traj
        T = len(step_indices)

        # R_env: γ^(T-1-k) * env_r for step at position k (0-indexed from start)
        for k, i in enumerate(step_indices):
            R_env[i] = (gamma ** (T - 1 - k)) * env_r

        # R_belief: cumulative discounted belief rewards (backward pass)
        g_next = 0.0
        for i in reversed(step_indices):
            r_belief = rebel_intrinsic_rewards[i].item()
            R_belief[i] = r_belief + gamma * g_next
            g_next = R_belief[i].item()

    return R_env, R_belief


def step_norm_reward_by_belief(
    step_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    belief_group_uids: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True
) -> torch.Tensor:
    """
    Step级别优势 - 在belief group内归一化

    按(uid, belief_hash)分组，计算组内归一化:
    A_step[i] = (R_intrinsic[i] - mean(R[group])) / (std + ε)

    Args:
        step_rewards: (batch,) intrinsic rewards
        eos_mask: (batch, seq_len)
        belief_group_uids: (batch,) group ids
        epsilon: 数值稳定性
        remove_std: True=mean_norm, False=mean_std_norm

    Returns:
        step_advantages: (batch, seq_len)
    """
    response_length = eos_mask.shape[-1]
    scores = step_rewards.clone()

    group2list = defaultdict(list)
    group2stats = {}

    with torch.no_grad():
        # 1. 收集每个group的rewards
        for i in range(len(scores)):
            group_uid = belief_group_uids[i]
            group2list[group_uid].append((i, scores[i]))

        # 2. 计算每个group的mean/std
        for group_uid, items in group2list.items():
            rewards = [item[1] for item in items]

            if len(rewards) == 1:
                # 单样本组: 无法与任何其他步骤比较 → 不产生梯度 (zero out)
                group2stats[group_uid] = (rewards[0], torch.tensor(1.0), True)
            else:
                t = torch.stack(rewards)
                mean, std = torch.mean(t), torch.std(t)
                group2stats[group_uid] = (mean, std, False)

        # 3. 归一化 — 单样本组强制为 0
        for i in range(len(scores)):
            group_uid = belief_group_uids[i]
            mean, std, is_singleton = group2stats[group_uid]
            if is_singleton:
                scores[i] = torch.tensor(0.0)
            elif remove_std:
                scores[i] = scores[i] - mean
            else:
                scores[i] = (scores[i] - mean) / (std + epsilon)

        # 4. 广播到所有token
        step_advantages = scores.unsqueeze(-1).expand(-1, response_length) * eos_mask

    return step_advantages


# ============================================================================ #
# ====================== Main: compute_rebel_advantage ======================= #
# ============================================================================ #

def compute_rebel_advantage(
    token_level_rewards: torch.Tensor,
    rebel_intrinsic_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    belief_states: np.ndarray,
    index: np.ndarray,
    epsilon: float = 1e-6,
    step_advantage_w: float = 0.5,
    mode: str = "mean_norm",
    belief_granularity: str = 'subgoal',
    summarize: bool = False,
    task_types: Optional[np.ndarray] = None,
    task_aware: bool = False,
    per_task_normalization: bool = False,
    conditional_norm: bool = True,
    min_samples_for_norm: int = 10,
    min_std_for_norm: float = 0.1,
    min_samples_ratio: float = 0.0,
    traj_index: Optional[np.ndarray] = None,
    gamma: float = 0.95,
    # V8新增: 任务自适应权重参数
    task_success_rates: Optional[Dict[str, float]] = None,
    use_task_weighting: bool = False,
    weight_alpha: float = 2.0,
    weight_min: float = 0.3,
    weight_max: float = 3.0,
    weight_baseline_sr: float = 0.85,
    # cumulative belief for grouping (more stable than per-step belief)
    cumulative_belief_states: Optional[np.ndarray] = None,
    # env-derived phase anchor for hybrid grouping
    gt_phases: Optional[np.ndarray] = None,
    # weight of A_belief_ep in A_episode
    alpha_belief_ep: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    ReBel优势计算主函数

    公式: A_total = A_episode + λ × A_step

    其中:
    - A_episode = A_task: normalized R_g (task outcome only, R_ep - Σr_cons ∈ {0,1})
    - A_step: R_step[t] = Σ γ^{k-t} R_ep[k] 在 (uid, belief_hash) 组内归一化
    - λ: step_advantage_w
    """
    remove_std = (mode == "mean_norm")

    # 1. Episode 优势：仅使用任务结果 R_g = episode_rewards - Σr_cons = R_env（纯任务成功信号）
    # EpisodeRewardManager 将 episode_rewards（env+intrinsic 总和）放到每步最后 token。
    # 所有步的 R_ep[i] 值相同（均等于 episode_total），因此只取一次（=赋值），不做累加。
    if traj_index is not None:
        R_ep = token_level_rewards.sum(dim=-1)  # (batch,) = episode_total, same per traj
        traj2sum_r_cons: Dict[Any, float] = defaultdict(float)
        traj2R_ep: Dict[Any, float] = {}  # 每条轨迹取一次（最后一次 = 任意一次，值相同）
        for i, traj in enumerate(traj_index):
            traj2sum_r_cons[traj] += rebel_intrinsic_rewards[i].item()
            traj2R_ep[traj] = R_ep[i].item()  # 赋值不累加：episode_total 对同轨迹所有步相同
        R_g = torch.tensor(
            [traj2R_ep[traj_index[i]] - traj2sum_r_cons[traj_index[i]] for i in range(len(traj_index))],
            dtype=token_level_rewards.dtype,
        )
        # R_g ≈ env_reward ∈ {0, 10}，不含信念奖励
        A_task = _norm_scores_by_uid(R_g, eos_mask, index, epsilon, remove_std, traj_index)
        episode_advantages = A_task
        A_belief_ep = torch.zeros_like(episode_advantages)
    else:
        # Fallback: original joint normalization of R_ep (backward compatible)
        episode_advantages = episode_norm_reward(
            token_level_rewards, eos_mask, index, epsilon, remove_std,
            traj_index=None,
        )
        A_task = episode_advantages
        A_belief_ep = torch.zeros_like(episode_advantages)

    # 2. 构建belief groups — use cumulative belief when available (Fix A)
    grouping_beliefs = cumulative_belief_states if cumulative_belief_states is not None else belief_states
    belief_group_uids, group_stats = build_belief_group(
        grouping_beliefs, index, belief_granularity, summarize,
        task_types=task_types, task_aware=task_aware, gt_phases=gt_phases,
    )

    # 3. 计算折扣回报，用于 A_step 信念组内比较（分离归一化）
    # R_env 和 R_belief 各自在 belief group 内归一化后等权合并，
    # 避免 env_reward 量级（10）淹没 belief_reward（~0.3/步）。
    if traj_index is not None:
        R_env, R_belief = compute_discounted_returns(
            rebel_intrinsic_rewards=rebel_intrinsic_rewards,
            episode_env_rewards=R_g,  # env-only reward, same per traj (computed above)
            traj_index=traj_index,
            gamma=gamma,
        )
        A_step_env    = step_norm_reward_by_belief(R_env,    eos_mask, belief_group_uids, epsilon, remove_std)
        A_step_belief = step_norm_reward_by_belief(R_belief, eos_mask, belief_group_uids, epsilon, remove_std)
        step_advantages = A_step_env + A_step_belief
    else:
        # 无 traj_index 时退化为 per-step r_cons（向后兼容）
        step_advantages = step_norm_reward_by_belief(
            rebel_intrinsic_rewards, eos_mask, belief_group_uids, epsilon, remove_std
        )

    # 4. 组合
    total_advantages = episode_advantages + step_advantage_w * step_advantages

    # 5. V2新增: 按任务归一化 (可选), V5改进: 条件归一化, V6改进: 相对阈值, V8改进: 任务自适应权重
    task_weights_applied = {}
    if per_task_normalization and task_types is not None:
        total_advantages, task_weights_applied = normalize_advantages_per_task(
            total_advantages, task_types, eos_mask, epsilon,
            min_samples_for_norm=min_samples_for_norm,
            min_std_for_norm=min_std_for_norm,
            use_conditional_norm=conditional_norm,
            min_samples_ratio=min_samples_ratio,  # V6新增
            # V8新增: 任务自适应权重参数
            task_success_rates=task_success_rates,
            use_task_weighting=use_task_weighting,
            weight_alpha=weight_alpha,
            weight_min=weight_min,
            weight_max=weight_max,
            weight_baseline_sr=weight_baseline_sr
        )

    # 6. 统计信息 (用于SwanLab/WandB logging)
    adv_details = {
        # 优势统计 — 组合后
        'episode_advantages': episode_advantages,
        'step_advantages': step_advantages,
        'episode_adv_mean': float(episode_advantages.mean().item()),
        'episode_adv_std': float(episode_advantages.std().item()),
        'step_adv_mean': float(step_advantages.mean().item()),
        'step_adv_std': float(step_advantages.std().item()),
        'total_adv_mean': float(total_advantages.mean().item()),
        'total_adv_std': float(total_advantages.std().item()),
        'rebel/using_cumulative_belief': int(cumulative_belief_states is not None),
        # 分组统计 (关键metrics)
        'belief_group_stats': group_stats,
        'rebel/num_groups': group_stats['num_groups'],
        'rebel/mean_group_size': group_stats['mean_group_size'],
        'rebel/median_group_size': group_stats['median_group_size'],
        'rebel/min_group_size': group_stats['min_group_size'],
        'rebel/max_group_size': group_stats['max_group_size'],
        'rebel/std_group_size': group_stats['std_group_size'],
        'rebel/single_sample_ratio': group_stats['single_sample_ratio'],
        # V2/V5/V6配置
        'rebel/task_aware': int(task_aware),
        'rebel/per_task_norm': int(per_task_normalization),
        'rebel/conditional_norm': int(conditional_norm),
        'rebel/min_samples_for_norm': min_samples_for_norm,
        'rebel/min_std_for_norm': min_std_for_norm,
        'rebel/min_samples_ratio': min_samples_ratio,  # V6新增
        # V8新增: 任务自适应权重
        'rebel/use_task_weighting': int(use_task_weighting),
        'rebel/weight_alpha': weight_alpha,
        'rebel/weight_baseline_sr': weight_baseline_sr,
    }

    # A_task 分布统计
    if traj_index is not None:
        adv_details['rebel/A_task_mean']       = float(A_task.mean().item())
        adv_details['rebel/A_task_std']        = float(A_task.std().item())
        adv_details['rebel/A_task_abs_mean']   = float(A_task.abs().mean().item())

    # 添加每个任务类型的组数 (如果启用任务感知)
    if task_aware and 'task_group_counts' in group_stats:
        for task, count in group_stats['task_group_counts'].items():
            if task != "all":
                adv_details[f'rebel/num_groups_{task}'] = count

    # V8新增: 记录每个任务的实际权重
    if task_weights_applied:
        for task, weight in task_weights_applied.items():
            task_short = task.replace('pick_', '').replace('_then_place_in_recep', '').replace('_obj', '').replace('_in_light', '')
            adv_details[f'rebel/task_weight_{task_short}'] = weight

    if summarize:
        task_weight_info = ""
        if task_weights_applied:
            task_weight_info = "\n├─ Task Weights (V8):"
            for task, weight in sorted(task_weights_applied.items()):
                task_short = task.replace('pick_', '').replace('_then_place_in_recep', '').replace('_obj', '').replace('_in_light', '')
                task_weight_info += f"\n│  └─ {task_short}: {weight:.2f}x"

        print(f"""
ReBel Advantage Statistics:
├─ Episode Adv Mean: {adv_details['episode_adv_mean']:.4f}
├─ Episode Adv Std: {adv_details['episode_adv_std']:.4f}
├─ Step Adv Mean: {adv_details['step_adv_mean']:.4f}
├─ Step Adv Std: {adv_details['step_adv_std']:.4f}
├─ Total Adv Mean: {adv_details['total_adv_mean']:.4f}
├─ Total Adv Std: {adv_details['total_adv_std']:.4f}
├─ Task-Aware: {task_aware}
├─ Per-Task Norm: {per_task_normalization}
└─ Task Weighting (V8): {use_task_weighting}{task_weight_info}
""")

    return total_advantages, total_advantages, adv_details


# ============================================================================ #
# ====================== V8: Task Adaptive Weighting ========================= #
# ============================================================================ #

def compute_task_adaptive_weights(
    task_success_rates: Dict[str, float],
    alpha: float = 2.0,
    min_weight: float = 0.3,
    max_weight: float = 3.0,
    baseline_sr: float = 0.85
) -> Dict[str, float]:
    """
    V8新增: 根据任务成功率计算自适应权重

    原理: 低成功率任务获得更高权重，增强其梯度贡献

    公式:
        weight(task) = clip(((1 - sr) / (1 - baseline_sr))^alpha, min_weight, max_weight)

    示例 (alpha=2.0, baseline_sr=0.85):
        pick_and_place: sr=97.8% → weight = ((1-0.978)/(1-0.85))^2 = 0.02 → clip to 0.3
        look_at:        sr=67.6% → weight = ((1-0.676)/(1-0.85))^2 = 4.68 → clip to 3.0

    Args:
        task_success_rates: {task_name: success_rate} 各任务当前成功率
        alpha: 权重调节指数，越大权重差异越明显
        min_weight: 最小权重（防止完全忽略高性能任务）
        max_weight: 最大权重（防止过度偏向单一任务）
        baseline_sr: 基线成功率，低于此值的任务获得更高权重

    Returns:
        task_weights: {task_name: weight} 各任务的梯度缩放权重
    """
    if not task_success_rates:
        return {}

    weights = {}
    denominator = 1.0 - baseline_sr

    for task, sr in task_success_rates.items():
        if denominator <= 0:
            denominator = 0.15  # 防止除零

        # 计算原始权重: 成功率越低，权重越高
        numerator = 1.0 - sr
        raw_weight = (numerator / denominator) ** alpha

        # 应用边界限制
        weight = max(min_weight, min(max_weight, raw_weight))
        weights[task] = weight

    return weights


def normalize_advantages_per_task(
    advantages: torch.Tensor,
    task_types: np.ndarray,
    eos_mask: torch.Tensor,
    epsilon: float = 1e-8,
    min_samples_for_norm: int = 10,
    min_std_for_norm: float = 0.1,
    use_conditional_norm: bool = True,
    min_samples_ratio: float = 0.0,  # V6新增: 相对阈值 (0表示不使用)
    # V8新增: 任务自适应权重参数
    task_success_rates: Optional[Dict[str, float]] = None,
    use_task_weighting: bool = False,
    weight_alpha: float = 2.0,
    weight_min: float = 0.3,
    weight_max: float = 3.0,
    weight_baseline_sr: float = 0.85
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    V8改进: 条件归一化 + 任务自适应权重

    V6策略 (保留):
    1. 样本数 >= min_samples_for_norm 且 std >= min_std_for_norm: 标准归一化 (mean=0, std=1)
    2. 样本数 >= min_samples_for_norm 但 std < min_std_for_norm: 保守归一化 (仅去均值)
    3. 样本数 < min_samples_for_norm: 使用全局统计量归一化

    V8新增: 任务自适应权重
    - 根据任务成功率计算梯度缩放权重
    - 低成功率任务(如look_at)获得更高权重，增强梯度贡献
    - 高成功率任务(如pick_and_place)降低权重，避免过拟合

    Args:
        advantages: (batch, seq_len) 原始优势
        task_types: (batch,) 任务类型
        eos_mask: (batch, seq_len) 有效token掩码
        epsilon: 数值稳定性
        min_samples_for_norm: 最小样本数阈值
        min_std_for_norm: 最小标准差阈值
        use_conditional_norm: 是否启用条件归一化
        min_samples_ratio: V6新增，相对阈值 (0表示不使用)
        task_success_rates: V8新增，各任务当前成功率
        use_task_weighting: V8新增，是否启用任务权重
        weight_alpha: V8新增，权重调节指数
        weight_min: V8新增，最小权重
        weight_max: V8新增，最大权重
        weight_baseline_sr: V8新增，基线成功率

    Returns:
        normalized: (batch, seq_len) 归一化后的优势
        task_weights: V8新增，各任务的实际缩放权重 (用于logging)
    """
    result = advantages.clone()
    unique_tasks = np.unique(task_types)
    total_samples = len(task_types)
    applied_task_weights = {}  # V8: 记录实际应用的权重

    # V8: 计算任务自适应权重
    if use_task_weighting and task_success_rates:
        task_weights = compute_task_adaptive_weights(
            task_success_rates=task_success_rates,
            alpha=weight_alpha,
            min_weight=weight_min,
            max_weight=weight_max,
            baseline_sr=weight_baseline_sr
        )
    else:
        task_weights = {}

    # 计算全局统计量 (用于小样本任务的后备归一化)
    all_valid_mask = eos_mask > 0
    if all_valid_mask.sum() > 0:
        global_mean = advantages[all_valid_mask].mean()
        global_std = advantages[all_valid_mask].std()
    else:
        global_mean = torch.tensor(0.0)
        global_std = torch.tensor(1.0)

    with torch.no_grad():
        for task in unique_tasks:
            # 找到属于该任务的样本
            mask = np.array([t == task for t in task_types])
            sample_count = mask.sum()

            if sample_count <= 1:
                continue

            # 获取该任务的优势
            task_adv = advantages[mask]
            task_eos = eos_mask[mask]

            # 只对有效token计算统计量
            valid_mask = task_eos > 0
            if valid_mask.sum() == 0:
                continue

            valid_adv = task_adv[valid_mask]
            mean = valid_adv.mean()
            std = valid_adv.std()

            # V6: 计算样本比例 (用于相对阈值判断)
            sample_ratio = sample_count / total_samples

            # V6条件归一化逻辑
            if use_conditional_norm:
                # V6: 检查是否为小样本任务 (使用绝对或相对阈值)
                is_small_sample = sample_count < min_samples_for_norm
                if min_samples_ratio > 0:
                    is_small_sample = is_small_sample or (sample_ratio < min_samples_ratio)

                if is_small_sample:
                    # 方案A: 样本太少，使用混合归一化策略
                    # V6改进: 使用混合因子而非完全使用全局统计量
                    if global_std > epsilon:
                        # 混合因子: 样本越多，越偏向任务内统计量
                        if min_samples_ratio > 0:
                            blend_factor = min(1.0, sample_ratio / min_samples_ratio)
                        else:
                            blend_factor = min(1.0, sample_count / min_samples_for_norm)

                        # 混合均值和标准差
                        blended_mean = blend_factor * mean + (1 - blend_factor) * global_mean
                        blended_std = blend_factor * std + (1 - blend_factor) * global_std

                        normalized_adv = (task_adv - blended_mean) / (blended_std + epsilon)
                    else:
                        normalized_adv = task_adv - mean
                    result[mask] = normalized_adv * task_eos

                elif std < min_std_for_norm:
                    # 方案B: 原始方差太小，使用保守归一化（仅去均值）
                    # 这保留了任务内部的自然方差信息
                    normalized_adv = task_adv - mean
                    result[mask] = normalized_adv * task_eos

                else:
                    # 方案C: 正常情况，使用标准归一化
                    normalized_adv = (task_adv - mean) / (std + epsilon)
                    result[mask] = normalized_adv * task_eos
            else:
                # 原始逻辑（兼容V4）
                if std > epsilon:
                    normalized_adv = (task_adv - mean) / (std + epsilon)
                    result[mask] = normalized_adv * task_eos

            # V8: 应用任务自适应权重
            if task_weights and task in task_weights:
                weight = task_weights[task]
                result[mask] = result[mask] * weight
                applied_task_weights[task] = weight
            else:
                applied_task_weights[task] = 1.0

    return result, applied_task_weights


# ============================================================================ #
# ====================== Intrinsic Reward Components ========================= #
# ============================================================================ #

def consistency_reward(
    belief: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None
) -> float:
    """
    评估belief与真实环境的一致性（V4增强：包含物体状态检查）

    Args:
        belief: 模型输出的belief
        ground_truth: 真实环境状态
        task_type: 任务类型（用于状态改变任务的额外检查）

    Returns:
        reward: [-0.5, 1.0]
    """
    if belief is None:
        return 0.0

    reward = 0.0
    world_model = belief.get('world_model_update', {}) or {}

    if ground_truth is None:
        # 无ground truth时，给予结构化信念的部分奖励
        found_objects = world_model.get('found_objects', {})
        if isinstance(found_objects, dict) and found_objects:
            reward += 0.1 * min(len(found_objects), 5)  # Max 0.5

        # V4新增: 对状态改变任务，检查是否记录了状态变化
        state_changes = world_model.get('state_changes', {}) or {}
        if task_type and 'heat' in str(task_type).lower():
            if any('heated' in str(v).lower() for v in state_changes.values()):
                reward += 0.2  # 正确记录了加热状态
        elif task_type and 'cool' in str(task_type).lower():
            if any('cooled' in str(v).lower() for v in state_changes.values()):
                reward += 0.2
        elif task_type and 'clean' in str(task_type).lower():
            if any('cleaned' in str(v).lower() for v in state_changes.values()):
                reward += 0.2

        return np.clip(reward, 0, 0.7)

    # 有ground truth: 验证正确性
    gt_objects = ground_truth.get('object_locations', {}) or {}
    found_objects = world_model.get('found_objects', {})

    if isinstance(found_objects, dict):
        for obj_id, believed_loc in found_objects.items():
            if not obj_id or not believed_loc:
                continue
            obj_lower = str(obj_id).lower()
            believed_lower = str(believed_loc).lower()

            for gt_obj, gt_loc in gt_objects.items():
                if obj_lower in gt_obj.lower() or gt_obj.lower() in obj_lower:
                    if believed_lower in gt_loc.lower() or gt_loc.lower() in believed_lower:
                        reward += 0.2  # 正确belief
                    else:
                        reward -= 0.1  # 错误belief
                    break

    # V4新增: 验证物体状态一致性
    gt_states = ground_truth.get('object_states', {}) or {}
    state_changes = world_model.get('state_changes', {}) or {}
    if isinstance(state_changes, dict) and gt_states:
        for obj, believed_state in state_changes.items():
            obj_lower = str(obj).lower()
            believed_state_lower = str(believed_state).lower()
            for gt_obj, gt_state in gt_states.items():
                if obj_lower in gt_obj.lower() or gt_obj.lower() in obj_lower:
                    gt_state_lower = str(gt_state).lower()
                    if believed_state_lower in gt_state_lower or gt_state_lower in believed_state_lower:
                        reward += 0.15  # 状态一致
                    else:
                        reward -= 0.1  # 状态不一致
                    break

    return np.clip(reward, -0.5, 1.0)


def progress_reward(
    belief: Dict[str, Any],
    prev_belief: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None
) -> float:
    """
    评估任务进度（V4增强：包含状态改变检测）

    Args:
        belief: 当前belief
        prev_belief: 上一步belief
        task_type: 任务类型

    Returns:
        reward: [0, 1.0]
    """
    if belief is None:
        return 0.0

    reward = 0.0
    task_progress = belief.get('task_progress_update', {}) or {}
    prev_task_progress = (prev_belief.get('task_progress_update', {}) or {}) if prev_belief else {}
    world_model = belief.get('world_model_update', {}) or {}
    prev_world_model = (prev_belief.get('world_model_update', {}) or {}) if prev_belief else {}

    # 1. 子目标完成
    curr_status = str(task_progress.get('subgoal_status', '')).lower()
    prev_status = str(prev_task_progress.get('subgoal_status', '')).lower()

    if 'complete' in curr_status and 'complete' not in prev_status:
        reward += 0.4  # 新完成子目标

    # 2. 有意义的证据
    evidence = str(task_progress.get('evidence', ''))
    if evidence and len(evidence) > 10:
        reward += 0.1

    # 3. 子目标更新
    curr_subgoal = str(task_progress.get('updated_subgoal', '')).lower().strip()
    prev_subgoal = str(prev_task_progress.get('updated_subgoal', '')).lower().strip()

    if curr_subgoal and curr_subgoal != prev_subgoal:
        reward += 0.1  # 子目标推进

    # V4新增: 状态改变检测（对heat/cool/clean任务关键）
    curr_state_changes = world_model.get('state_changes', {}) or {}
    prev_state_changes = prev_world_model.get('state_changes', {}) or {}

    if isinstance(curr_state_changes, dict) and isinstance(prev_state_changes, dict):
        # 检测新的状态改变
        new_states = set()
        for obj, state in curr_state_changes.items():
            state_lower = str(state).lower()
            prev_state = str(prev_state_changes.get(obj, '')).lower()
            if state_lower != prev_state:
                # 检测具体的状态类型
                if any(x in state_lower for x in ['heated', 'hot', 'cooked']):
                    new_states.add('heated')
                elif any(x in state_lower for x in ['cooled', 'cold', 'chilled']):
                    new_states.add('cooled')
                elif any(x in state_lower for x in ['cleaned', 'clean', 'washed']):
                    new_states.add('cleaned')

        # 根据任务类型给予奖励
        if task_type:
            task_lower = str(task_type).lower()
            if 'heat' in task_lower and 'heated' in new_states:
                reward += 0.3  # 完成了加热
            elif 'cool' in task_lower and 'cooled' in new_states:
                reward += 0.3  # 完成了冷却
            elif 'clean' in task_lower and 'cleaned' in new_states:
                reward += 0.3  # 完成了清洁
        elif new_states:
            # 无task_type时，任何新状态变化给予小奖励
            reward += 0.1 * len(new_states)

    return np.clip(reward, 0, 1.0)


def exploration_reward(
    belief: Dict[str, Any],
    prev_belief: Optional[Dict[str, Any]] = None
) -> float:
    """
    评估探索效率

    Args:
        belief: 当前belief
        prev_belief: 上一步belief

    Returns:
        reward: [-0.1, 0.5]
    """
    if belief is None:
        return 0.0

    reward = 0.0
    exploration = belief.get('exploration_map_update', {}) or {}
    prev_exploration = (prev_belief.get('exploration_map_update', {}) or {}) if prev_belief else {}

    # 1. 新访问的位置
    newly_visited = exploration.get('newly_visited', [])
    if isinstance(newly_visited, str):
        newly_visited = [newly_visited]
    if not isinstance(newly_visited, list):
        newly_visited = []

    prev_visited = prev_exploration.get('newly_visited', [])
    if isinstance(prev_visited, str):
        prev_visited = [prev_visited]
    if not isinstance(prev_visited, list):
        prev_visited = []

    new_locations = set(newly_visited) - set(prev_visited)
    reward += 0.1 * min(len(new_locations), 3)  # Max 0.3

    # 2. 避免重复探索
    world_model = belief.get('world_model_update', {}) or {}
    cleared = world_model.get('cleared_receptacles', [])
    if not isinstance(cleared, list):
        cleared = []

    for loc in newly_visited:
        if loc in cleared:
            reward -= 0.02  # 重复访问惩罚

    return np.clip(reward, -0.1, 0.5)


def format_reward(output: str, is_action_available: bool = True) -> float:
    """
    评估输出格式 - RLVMR风格惩罚机制

    检查规则:
    1. 无中文字符
    2. belief/action标签各1个
    3. 标签内容非空
    4. 标签顺序: belief → action
    5. belief是有效JSON

    Returns:
        reward: 格式正确返回小奖励，格式错误返回-1.0
    """
    # 1. 检查中文字符
    if re.search(r'[\u4e00-\u9fff]', output):
        return -1.0

    # 2. 检查action标签数量
    action_matches = re.findall(r"<action>([\s\S]*?)</action>", output, re.IGNORECASE)
    if len(action_matches) != 1:
        return -1.0

    # 3. 检查belief标签数量
    belief_matches = re.findall(r"<belief>([\s\S]*?)</belief>", output, re.IGNORECASE)
    if len(belief_matches) != 1:
        return -1.0

    # 4. 检查belief内容非空
    belief_content = belief_matches[0].strip()
    if not belief_content:
        return -1.0

    # 5. 检查标签顺序
    belief_pos = output.lower().find("<belief>")
    action_pos = output.lower().find("<action>")
    if belief_pos > action_pos:
        return -1.0

    # 6. 检查belief是否为有效JSON
    try:
        belief_json = json.loads(belief_content)
        valid_keys = ["state", "task", "prediction"]
        if not any(key in belief_json for key in valid_keys):
            return -0.5  # missing required fields
    except json.JSONDecodeError:
        return -1.0

    # 格式验证通过
    reward = 0.1
    if not is_action_available:
        reward -= 0.2

    return reward


def compute_intrinsic_reward(
    belief: Dict[str, Any],
    prev_belief: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    output: str = "",
    is_format_valid: bool = True,
    is_action_available: bool = True,
    weights: Optional[Dict[str, float]] = None,
    task_type: Optional[str] = None
) -> Tuple[float, Dict[str, float]]:
    """
    计算ReBel内在奖励（V4增强：支持task_type）

    公式: R_intrinsic = α*R_consistency + β*R_progress + γ*R_exploration

    注意: Format不是内在奖励的一部分，是独立的格式惩罚

    Args:
        belief: 当前belief
        prev_belief: 上一步belief
        ground_truth: 真实环境状态
        output: 模型输出
        is_format_valid: 格式是否有效
        is_action_available: action是否可用
        weights: 权重配置
        task_type: 任务类型（V4新增，用于状态改变任务的奖励增强）

    Returns:
        total_reward: 总内在奖励
        component_rewards: 各组件奖励
    """
    weights = weights or {
        'consistency': 0.3,
        'progress': 0.5,
        'exploration': 0.2,
    }

    # 先检查格式
    r_format = format_reward(output, is_action_available)

    component_rewards = {
        'format': r_format,
        'is_format_valid': r_format > -0.5,
    }

    # 格式无效时，跳过内在奖励计算
    if r_format <= -0.5:
        component_rewards['consistency'] = 0.0
        component_rewards['progress'] = 0.0
        component_rewards['exploration'] = 0.0
        component_rewards['intrinsic'] = 0.0
        return r_format, component_rewards  # 返回格式惩罚

    # 计算各组件（V4: 传递task_type）
    r_consistency = consistency_reward(belief, ground_truth, task_type)
    r_progress = progress_reward(belief, prev_belief, task_type)
    r_exploration = exploration_reward(belief, prev_belief)

    component_rewards['consistency'] = r_consistency
    component_rewards['progress'] = r_progress
    component_rewards['exploration'] = r_exploration

    # 加权组合
    intrinsic = (
        weights['consistency'] * r_consistency +
        weights['progress'] * r_progress +
        weights['exploration'] * r_exploration
    )

    component_rewards['intrinsic'] = intrinsic

    # 总奖励 = 内在奖励 + 格式奖励
    total_reward = intrinsic + r_format

    return total_reward, component_rewards


# ============================================================================ #
# ====================== Utility Functions =================================== #
# ============================================================================ #

def print_rebel_summary(adv_details: Dict[str, Any]):
    """打印ReBel统计摘要"""
    group_stats = adv_details.get('belief_group_stats', {})

    print(f"""
{'='*50}
ReBel Statistics Summary
{'='*50}
Belief-Based Grouping:
├─ Num Groups: {group_stats.get('num_groups', 'N/A')}
├─ Mean Group Size: {group_stats.get('mean_group_size', 0):.2f}
├─ Median Group Size: {group_stats.get('median_group_size', 0):.1f}
├─ Min/Max Size: {group_stats.get('min_group_size', 0)}/{group_stats.get('max_group_size', 0)}
└─ Single-Sample Groups: {group_stats.get('single_sample_groups', 0)}

Advantages:
├─ Episode Adv: mean={adv_details.get('episode_adv_mean', 0):.4f}, std={adv_details.get('episode_adv_std', 0):.4f}
├─ Step Adv: mean={adv_details.get('step_adv_mean', 0):.4f}, std={adv_details.get('step_adv_std', 0):.4f}
└─ Total Adv: mean={adv_details.get('total_adv_mean', 0):.4f}, std={adv_details.get('total_adv_std', 0):.4f}
{'='*50}
""")


def health_check(adv_details: Dict[str, Any]) -> Dict[str, str]:
    """健康检查"""
    warnings = {}
    group_stats = adv_details.get('belief_group_stats', {})

    num_groups = group_stats.get('num_groups', 0)
    if num_groups < 20:
        warnings['num_groups'] = f"组数太少({num_groups})，考虑使用更细的粒度"
    elif num_groups > 100:
        warnings['num_groups'] = f"组数太多({num_groups})，考虑使用更粗的粒度"

    mean_size = group_stats.get('mean_group_size', 0)
    if mean_size < 10:
        warnings['group_size'] = f"平均组大小太小({mean_size:.1f})，归一化效果可能不佳"
    elif mean_size > 50:
        warnings['group_size'] = f"平均组大小太大({mean_size:.1f})，考虑使用更细的粒度"

    single_ratio = group_stats.get('single_sample_groups', 0) / max(num_groups, 1)
    if single_ratio > 0.5:
        warnings['single_samples'] = f"单样本组占比太高({single_ratio:.1%})"

    return warnings
