#!/bin/bash
# =============================================================================
# ReBel base training script
# =============================================================================
# ReBel (Reinforcement Learning with Belief-State Enhancement) 正式实验的统一入口。
# 支持多种子批量运行，通过环境变量控制实验配置。
#
# 必需环境变量:
#   EXP_ID          - 实验编号 (M1-M5 主实验, A1-A6 消融)
#   EXP_NAME        - 实验名称 (如 grpo_baseline, rebel_full)
#   ADV_ESTIMATOR   - advantage 估计器 (grpo/gigpo/rebel)
#   USE_REBEL_PROMPT - 是否使用 <belief> 提示格式 (true/false)
#
# 可选环境变量 (有默认值):
#   USE_TRAINING_TRICKS  - 是否启用训练稳定化 (true/false, 默认 false)
#   USE_ADV_TRICKS       - 是否启用 advantage tricks (true/false, 默认 false)
#   USE_BELIEF_REWARD    - 是否使用内在信念奖励 (true/false, 默认 false)
#   USE_RESULT_REWARD    - 是否使用结果奖励 (true/false, 默认 true)
#   USE_BELIEF_DECAY     - 是否启用信念奖励衰减 (true/false, 默认 false)
#   DECAY_METHOD         - 衰减方法 (cosine/adaptive, 默认 cosine)
#   USE_ADAPTIVE_DECAY   - 是否启用自适应衰减 (true/false, 默认 false)
#   USE_DIFFERENTIAL_DECAY - 是否启用差异化组件衰减 (true/false, 默认 true)
#   NUM_GPUS             - GPU 数量 (默认 8)
#   EPOCHS               - 训练轮次 (默认 100)
#   SEED                 - 随机种子 (默认 42)
#   MODEL_PATH           - 起始模型路径 (默认自动查找 SFT checkpoint)
# =============================================================================

set -e

# ======================== 项目根目录 (自动检测，无需硬编码) ========================
# 脚本位于 examples/{trainer}/_base_alfworld.sh，上两级即为根目录
VERL_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${VERL_AGENT_DIR}:${PYTHONPATH:-}"

# ======================== 基本参数 ========================
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}
PPO_MICRO_BSZ_PER_GPU=${PPO_MICRO_BSZ_PER_GPU:-8}
TRAIN_BSZ=${TRAIN_BSZ:-16}
# 存档/验证频率（单位：epoch）；脚本将自动乘以 steps_per_epoch 转换为 global_steps
SAVE_FREQ_EPOCHS=${SAVE_FREQ_EPOCHS:-10}
TEST_FREQ_EPOCHS=${TEST_FREQ_EPOCHS:-5}

# ======================== 实验标识 ========================
EXP_ID=${EXP_ID:?"ERROR: EXP_ID is required (M1-M5/A1-A6)"}
EXP_NAME=${EXP_NAME:?"ERROR: EXP_NAME is required"}
ADV_ESTIMATOR=${ADV_ESTIMATOR:?"ERROR: ADV_ESTIMATOR is required (grpo/gigpo/rebel)"}
USE_REBEL_PROMPT=${USE_REBEL_PROMPT:?"ERROR: USE_REBEL_PROMPT is required (true/false)"}

# ======================== 可选配置 ========================
USE_TRAINING_TRICKS=${USE_TRAINING_TRICKS:-false}
USE_ADV_TRICKS=${USE_ADV_TRICKS:-false}
USE_BELIEF_REWARD=${USE_BELIEF_REWARD:-false}
USE_RESULT_REWARD=${USE_RESULT_REWARD:-true}
USE_BELIEF_DECAY=${USE_BELIEF_DECAY:-false}
STEP_ADV_W=${STEP_ADV_W:-0.5}  # step advantage weight; set to 0 to ablate A_step
DECAY_METHOD=${DECAY_METHOD:-cosine}
# Adaptive decay control
USE_ADAPTIVE_DECAY=${USE_ADAPTIVE_DECAY:-false}
USE_DIFFERENTIAL_DECAY=${USE_DIFFERENTIAL_DECAY:-true}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# ======================== SwanLab 配置 ========================
# 默认 local 模式（无需联网），日志写入 RESULTS_DIR/swanlog
# 如需上传云端：SWANLAB_MODE=cloud SWANLAB_API_KEY=<your_key> bash xxx.sh
export SWANLAB_MODE=${SWANLAB_MODE:-local}
if [ "${SWANLAB_MODE}" = "cloud" ]; then
    export SWANLAB_API_KEY=${SWANLAB_API_KEY:?"ERROR: SWANLAB_API_KEY is required for cloud mode. Get it from https://swanlab.cn"}
fi
# SWANLAB_LOG_DIR 将在 RESULTS_DIR 确定后（数据发现阶段）导出

# ======================== 模型路径 ========================
find_sft_model() {
    local path
    path=$(find "${VERL_AGENT_DIR}/checkpoints/sft/alfworld" -maxdepth 2 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "Qwen/Qwen2.5-1.5B-Instruct"
}

MODEL_PATH=${MODEL_PATH:-$(find_sft_model)}

# ======================== SFT Model Validation ========================
if [ "$MODEL_PATH" = "Qwen/Qwen2.5-1.5B-Instruct" ]; then
    echo ""
    echo "  WARNING: No ALFWorld SFT checkpoint found, falling back to base model."
    echo "  Run SFT first: bash scripts/sft_alfworld.sh"
    echo "  Or override:   MODEL_PATH=<sft_checkpoint_path> bash $0"
    echo ""
    sleep 3
fi

# ======================== 输出路径 ========================
RESULTS_BASE="${VERL_AGENT_DIR}/results/alfworld"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_EXP_NAME="${EXP_ID}_${EXP_NAME}_seed${SEED}"
RESULTS_DIR="${RESULTS_BASE}/${FULL_EXP_NAME}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/checkpoints"

# ======================== 训练 Tricks 参数 ========================
if [ "$USE_TRAINING_TRICKS" = "true" ]; then
    CLIP_RATIO_LOW=0.2
    CLIP_RATIO_HIGH=0.28
    ENTROPY_COEFF=0.001
    ENTROPY_PROTECTION_ENABLE=True
    ENTROPY_PROTECTION_METHOD=clip_cov
    CLIP_COV_LB=0.0
    CLIP_COV_UB=0.3
    INVALID_ACTION_PENALTY=True
    INVALID_ACTION_PENALTY_COEF=0.1
else
    CLIP_RATIO_LOW=0.2
    CLIP_RATIO_HIGH=0.2    # 对称裁剪
    ENTROPY_COEFF=0.001
    ENTROPY_PROTECTION_ENABLE=False
    ENTROPY_PROTECTION_METHOD=clip_cov
    CLIP_COV_LB=0.0
    CLIP_COV_UB=0.3
    INVALID_ACTION_PENALTY=False
    INVALID_ACTION_PENALTY_COEF=0.0
fi

# ======================== Advantage Tricks 参数 ========================
if [ "$USE_ADV_TRICKS" = "true" ]; then
    USE_TASK_WEIGHTING="true"
    WEIGHT_ALPHA=2.0
    WEIGHT_MIN=0.3
    WEIGHT_MAX=3.0
    TASK_WEIGHTING_WARMUP=20
    MIN_SAMPLES_RATIO=0.15
else
    USE_TASK_WEIGHTING="false"
    WEIGHT_ALPHA=2.0
    WEIGHT_MIN=0.3
    WEIGHT_MAX=3.0
    TASK_WEIGHTING_WARMUP=20
    MIN_SAMPLES_RATIO=0.0
fi

# ======================== Belief Decay 参数 ========================
DECAY_WARMUP_EPOCHS=3
DECAY_START_EPOCH=5
DECAY_END_EPOCH=40
DECAY_MIN_WEIGHT=0.05
# Adaptive decay parameters
DECAY_TARGET_SR=0.90
DECAY_ALPHA=2.0
# Differential component decay rates
PROGRESS_DECAY_RATE=0.7
CONSISTENCY_DECAY_RATE=1.0
EXPLORATION_DECAY_RATE=2.0

# # Override decay rates if differential decay is disabled
if [ "$USE_DIFFERENTIAL_DECAY" = "false" ]; then
    PROGRESS_DECAY_RATE=1.0
    CONSISTENCY_DECAY_RATE=1.0
    EXPLORATION_DECAY_RATE=1.0
fi

# ======================== 打印配置 ========================
echo "═══════════════════════════════════════════════════════════════════"
echo "  ReBel experiment: ${EXP_ID} - ${EXP_NAME}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Config:"
echo "  - ID:               ${EXP_ID}"
echo "  - 名称:             ${EXP_NAME}"
echo "  - Advantage:        ${ADV_ESTIMATOR}"
echo "  - 提示格式:         $([ "$USE_REBEL_PROMPT" = "true" ] && echo '<belief>' || echo '<think>')"
echo "  - Belief reward:    ${USE_BELIEF_REWARD}"
echo "  - Result reward:    ${USE_RESULT_REWARD}"
echo ""
echo "基本参数:"
echo "  - 种子:             ${SEED}"
echo "  - GPU:              ${NUM_GPUS}"
echo "  - Epochs:           ${EPOCHS}"
echo "  - 模型路径:         ${MODEL_PATH}"
echo "  - 结果目录:         ${RESULTS_DIR}"
echo ""
echo "───────────────────────────────────────────────────────────────────"

cd "${VERL_AGENT_DIR}"

# ======================== 准备数据 ========================
# 固定路径（与 old M5 一致）：train=16 行, test=128 行
# 注意：绝对不能使用 NFS 上的 3553 行大集合，否则：
#   steps/epoch=222 → 总步数 22200 × 1100s ≈ 282 天；首次 val 在第 14 天
_ALF_TRAIN="$HOME/data/verl-agent/alfworld/text/train.parquet"
_ALF_TEST="$HOME/data/verl-agent/alfworld/text/test.parquet"
_ALF_EXPECTED_VAL_ROWS=128

if [ ! -f "$_ALF_TRAIN" ]; then
    echo "[Data] ALFWorld parquet not found, generating (train=16, test=128)..."
    mkdir -p "$HOME/data/verl-agent/alfworld"
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' \
        --local_dir "$HOME/data/verl-agent/alfworld" \
        --train_data_size 16 \
        --val_data_size ${_ALF_EXPECTED_VAL_ROWS} 2>/dev/null || true
else
    _ACTUAL_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$_ALF_TEST')))" 2>/dev/null || echo 0)
    if [ "$_ACTUAL_ROWS" != "$_ALF_EXPECTED_VAL_ROWS" ]; then
        echo "[Data] test.parquet has ${_ACTUAL_ROWS} rows (need ${_ALF_EXPECTED_VAL_ROWS}), regenerating..."
        python3 -m examples.data_preprocess.prepare \
            --mode 'text' \
            --local_dir "$HOME/data/verl-agent/alfworld" \
            --train_data_size 16 \
            --val_data_size ${_ALF_EXPECTED_VAL_ROWS} 2>/dev/null || true
    fi
fi

echo "[Data] train: $_ALF_TRAIN"
echo "[Data] test:  $_ALF_TEST"

# ======================== 计算 steps/epoch → 转换存档频率 ========================
TRAIN_DATA_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${_ALF_TRAIN}')))")
STEPS_PER_EPOCH=$(python3 -c "import math; print(max(1, math.ceil(${TRAIN_DATA_ROWS}/${TRAIN_BSZ})))")
SAVE_FREQ_STEPS=$(( SAVE_FREQ_EPOCHS * STEPS_PER_EPOCH ))
TEST_FREQ_STEPS=$(( TEST_FREQ_EPOCHS * STEPS_PER_EPOCH ))

# SwanLab 本地日志目录（确定 RESULTS_DIR 之后才能设置）
export SWANLAB_LOG_DIR="${RESULTS_DIR}/swanlog"

echo "[Schedule] train_rows=${TRAIN_DATA_ROWS}, steps/epoch=${STEPS_PER_EPOCH}"
echo "[Schedule] total_steps=$((EPOCHS * STEPS_PER_EPOCH))  (${EPOCHS} epochs × ${STEPS_PER_EPOCH} steps)"
echo "[Schedule] save_freq=${SAVE_FREQ_STEPS} steps (every ${SAVE_FREQ_EPOCHS} epochs)"
echo "[Schedule] test_freq=${TEST_FREQ_STEPS} steps (every ${TEST_FREQ_EPOCHS} epochs)"
echo "[SwanLab]  mode=${SWANLAB_MODE}, logdir=${SWANLAB_LOG_DIR}"
echo ""

# ======================== 构建训练命令 ========================
# 基础参数 (所有实验共享)
BASE_ARGS=(
    "algorithm.adv_estimator=${ADV_ESTIMATOR}"
    "data.train_files=${_ALF_TRAIN}"
    "data.val_files=${_ALF_TEST}"
    "data.train_batch_size=${TRAIN_BSZ}"
    "data.val_batch_size=128"
    "data.max_prompt_length=6000"
    "data.max_response_length=1024"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.clip_ratio=0.2"
    "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
    "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=0.01"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=256"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BSZ_PER_GPU}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.65"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.free_cache_engine=False"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "algorithm.use_kl_in_reward=False"
    "algorithm.kl_penalty=kl"
    "algorithm.kl_ctrl.type=fixed"
    "algorithm.kl_ctrl.kl_coef=0.001"
    "env.env_name=alfworld/AlfredTWEnv"
    "env.seed=${SEED}"
    "env.max_steps=30"
    "env.rollout.n=16"
    "env.alfworld.generalization_level=0"
    "env.alfworld.meta_think=True"
    "env.use_teacher_planner=True"
    "trainer.critic_warmup=0"
    "trainer.logger=['console','swanlab']"
    "trainer.project_name=${SWANLAB_PROJECT:-ReBel}"
    "trainer.experiment_name=${FULL_EXP_NAME}"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "trainer.nnodes=1"
    "trainer.save_freq=${SAVE_FREQ_STEPS}"
    "trainer.test_freq=${TEST_FREQ_STEPS}"
    "trainer.total_epochs=${EPOCHS}"
    "trainer.default_local_dir=${RESULTS_DIR}/checkpoints"
    "trainer.val_before_train=True"
    "trainer.max_actor_ckpt_to_keep=3"
)

# Invalid action penalty
if [ "$INVALID_ACTION_PENALTY" = "True" ]; then
    BASE_ARGS+=(
        "actor_rollout_ref.actor.use_invalid_action_penalty=True"
        "actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF}"
    )
fi

# ReBel 提示格式相关参数
if [ "$USE_REBEL_PROMPT" = "true" ]; then
    BASE_ARGS+=(
        "algorithm.rebel.enable=True"
        "env.alfworld.use_rebel=True"
        "env.alfworld.prompt_template_type=explicit_task_type"
    )

    # ReBel 核心参数 (当 rebel.enable=True 时需要提供)
    BASE_ARGS+=(
        "algorithm.rebel.belief_granularity=gt_phase"
        "algorithm.rebel.step_advantage_w=${STEP_ADV_W}"
        "algorithm.rebel.mode=mean_norm"
        "algorithm.rebel.task_aware_grouping=true"
        "algorithm.rebel.per_task_normalization=true"
        "algorithm.rebel.conditional_norm=true"
        "algorithm.rebel.min_samples_for_norm=10"
        "algorithm.rebel.min_std_for_norm=0.2"
        "algorithm.rebel.min_samples_ratio=${MIN_SAMPLES_RATIO}"
        "algorithm.rebel.entropy_protection.enable=${ENTROPY_PROTECTION_ENABLE}"
        "algorithm.rebel.entropy_protection.method=${ENTROPY_PROTECTION_METHOD}"
        "algorithm.rebel.entropy_protection.clip_cov_lb=${CLIP_COV_LB}"
        "algorithm.rebel.entropy_protection.clip_cov_ub=${CLIP_COV_UB}"
        "algorithm.rebel.alpha=0.40"
        "algorithm.rebel.beta=0.30"
        "algorithm.rebel.gamma=0.20"
        "algorithm.rebel.delta=0.10"
    )

    # HiBO-specific parameter (min obs group size for fallback)
    BASE_ARGS+=(
        "algorithm.rebel.min_obs_group_size=2"
    )

    # 信念奖励和结果奖励开关
    BASE_ARGS+=(
        "algorithm.rebel.use_belief_reward=${USE_BELIEF_REWARD}"
        "algorithm.rebel.use_result_reward=${USE_RESULT_REWARD}"
    )

    # Advantage tricks (task weighting)
    BASE_ARGS+=(
        "algorithm.rebel.use_task_weighting=${USE_TASK_WEIGHTING}"
        "algorithm.rebel.weight_alpha=${WEIGHT_ALPHA}"
        "algorithm.rebel.weight_min=${WEIGHT_MIN}"
        "algorithm.rebel.weight_max=${WEIGHT_MAX}"
        "algorithm.rebel.weight_baseline_sr=0.85"
        "algorithm.rebel.task_weighting_warmup_epochs=${TASK_WEIGHTING_WARMUP}"
    )

    # Belief reward decay
    BASE_ARGS+=(
        "algorithm.rebel.belief_reward_decay.enable=${USE_BELIEF_DECAY}"
        "algorithm.rebel.belief_reward_decay.method=${DECAY_METHOD}"
        "algorithm.rebel.belief_reward_decay.warmup_epochs=${DECAY_WARMUP_EPOCHS}"
        "algorithm.rebel.belief_reward_decay.decay_start_epoch=${DECAY_START_EPOCH}"
        "algorithm.rebel.belief_reward_decay.decay_end_epoch=${DECAY_END_EPOCH}"
        "algorithm.rebel.belief_reward_decay.min_weight=${DECAY_MIN_WEIGHT}"
        # Adaptive decay parameters
        "algorithm.rebel.belief_reward_decay.adaptive=${USE_ADAPTIVE_DECAY}"
        "algorithm.rebel.belief_reward_decay.target_sr=${DECAY_TARGET_SR}"
        "algorithm.rebel.belief_reward_decay.alpha=${DECAY_ALPHA}"
        # Differential component decay rates
        "algorithm.rebel.belief_reward_decay.progress_decay_rate=${PROGRESS_DECAY_RATE}"
        "algorithm.rebel.belief_reward_decay.consistency_decay_rate=${CONSISTENCY_DECAY_RATE}"
        "algorithm.rebel.belief_reward_decay.exploration_decay_rate=${EXPLORATION_DECAY_RATE}"
    )
else
    BASE_ARGS+=(
        "algorithm.rebel.enable=False"
        "env.alfworld.use_rebel=False"
    )
fi

# GiGPO / HiBO 特定参数 (both need gamma and step params)
if [ "$ADV_ESTIMATOR" = "gigpo" ] || [ "$ADV_ESTIMATOR" = "rebel" ]; then
    BASE_ARGS+=(
        "algorithm.gamma=0.95"
        "algorithm.gigpo.step_advantage_w=${STEP_ADV_W}"
        "algorithm.gigpo.mode=mean_norm"
    )
fi

# ======================== 执行训练 ========================
echo ""
echo "开始训练..."
echo ""

python3 -m verl.trainer.main_ppo \
    "${BASE_ARGS[@]}" \
    2>&1 | tee "${RESULTS_DIR}/training.log"

# ── Checkpoint cleanup: keep only last 3 (epochs 80/90/100) ──────────────────
_CKPT_DIR="${RESULTS_DIR}/checkpoints"
readarray -t _ALL_CKPTS < <(find "$_CKPT_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V)
_N_KEEP=3
if [ "${#_ALL_CKPTS[@]}" -gt "$_N_KEEP" ]; then
    _N_DELETE=$(( ${#_ALL_CKPTS[@]} - _N_KEEP ))
    echo "[Cleanup] Removing ${_N_DELETE} early checkpoints (keeping last ${_N_KEEP})..."
    for _d in "${_ALL_CKPTS[@]:0:$_N_DELETE}"; do
        echo "  rm: $_d"
        rm -rf "$_d"
    done
    echo "[Cleanup] Done. Remaining checkpoints:"
    find "$_CKPT_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  实验完成: ${EXP_ID} - ${EXP_NAME} (seed=${SEED})"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "结果保存在: ${RESULTS_DIR}"
echo ""
