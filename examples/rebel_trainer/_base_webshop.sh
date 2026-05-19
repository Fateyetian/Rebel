#!/bin/bash
# =============================================================================
# ReBel WebShop base training script
# =============================================================================
# Unified entry point for WebShop ReBel experiments.
# Mirrors: _base_alfworld.sh (ALFWorld version)
#
# Key differences from ALFWorld:
#   - env.env_name=Webshop
#   - env.max_steps=15 (WebShop episodes are shorter)
#   - data.max_response_length=512 (WebShop responses shorter)
#   - No gamefile/generalization_level config
#   - MODEL_PATH points to WebShop SFT checkpoint
# =============================================================================

set -e

# ======================== Repo Root (mirrors _base_alfworld.sh) ========================
# Use $0 (not BASH_SOURCE) for portability on /ml_platform/bin/bash
# Script lives at examples/{trainer}/_base_webshop.sh → /../.. = repo root
VERL_AGENT_DIR="${VERL_AGENT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
export VERL_AGENT_DIR
export PYTHONPATH="${VERL_AGENT_DIR}:${PYTHONPATH:-}"

# ======================== Basic Parameters ========================
NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}

# ======================== Experiment ID ========================
EXP_ID=${EXP_ID:?"ERROR: EXP_ID is required (M1-M5/A1-A6)"}
EXP_NAME=${EXP_NAME:?"ERROR: EXP_NAME is required"}
ADV_ESTIMATOR=${ADV_ESTIMATOR:?"ERROR: ADV_ESTIMATOR is required (grpo/gigpo/rebel)"}
USE_REBEL_PROMPT=${USE_REBEL_PROMPT:?"ERROR: USE_REBEL_PROMPT is required (true/false)"}

# ======================== Optional Config ========================
USE_TRAINING_TRICKS=${USE_TRAINING_TRICKS:-false}
USE_ADV_TRICKS=${USE_ADV_TRICKS:-false}
USE_BELIEF_REWARD=${USE_BELIEF_REWARD:-false}
USE_RESULT_REWARD=${USE_RESULT_REWARD:-true}
USE_BELIEF_DECAY=${USE_BELIEF_DECAY:-false}
STEP_ADV_W=${STEP_ADV_W:-0.5}
DECAY_METHOD=${DECAY_METHOD:-cosine}
USE_ADAPTIVE_DECAY=${USE_ADAPTIVE_DECAY:-false}
USE_DIFFERENTIAL_DECAY=${USE_DIFFERENTIAL_DECAY:-true}
ROLLOUT_N=${ROLLOUT_N:-16}
SAVE_TRAJECTORIES=${SAVE_TRAJECTORIES:-true}  # Save rollout trajectories for visualization

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export SWANLAB_MODE=${SWANLAB_MODE:-local}

# ======================== Model Path ========================
find_sft_model() {
    local path
    path=$(find "${VERL_AGENT_DIR}/checkpoints/sft/webshop" -maxdepth 2 -type d -name 'global_step_*' 2>/dev/null | sort -t_ -k3 -n | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "Qwen/Qwen2.5-1.5B-Instruct"
}

MODEL_PATH=${MODEL_PATH:-$(find_sft_model)}

# ======================== SFT Model Validation ========================
if [ "$MODEL_PATH" = "Qwen/Qwen2.5-1.5B-Instruct" ]; then
    echo ""
    echo "  WARNING: No WebShop SFT checkpoint found, falling back to base model."
    echo "  Run SFT first: bash scripts/sft_webshop.sh"
    echo "  Or override:   MODEL_PATH=<sft_checkpoint_path> bash $0"
    echo ""
    sleep 3
fi

# ======================== Output Path ========================
RESULTS_BASE="${VERL_AGENT_DIR}/results/webshop"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_EXP_NAME="${EXP_ID}_${EXP_NAME}_seed${SEED}"
RESULTS_DIR="${RESULTS_BASE}/${FULL_EXP_NAME}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/checkpoints"

# ======================== Training Tricks ========================
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
    CLIP_RATIO_HIGH=0.2
    ENTROPY_COEFF=0.001
    ENTROPY_PROTECTION_ENABLE=False
    ENTROPY_PROTECTION_METHOD=clip_cov
    CLIP_COV_LB=0.0
    CLIP_COV_UB=0.3
    INVALID_ACTION_PENALTY=False
    INVALID_ACTION_PENALTY_COEF=0.0
fi

# ======================== Advantage Tricks ========================
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

# ======================== Belief Decay ========================
DECAY_WARMUP_EPOCHS=3
DECAY_START_EPOCH=5
DECAY_END_EPOCH=40
DECAY_MIN_WEIGHT=0.05
DECAY_TARGET_SR=0.90
DECAY_ALPHA=2.0
PROGRESS_DECAY_RATE=0.7
CONSISTENCY_DECAY_RATE=1.0
EXPLORATION_DECAY_RATE=2.0

if [ "$USE_DIFFERENTIAL_DECAY" = "false" ]; then
    PROGRESS_DECAY_RATE=1.0
    CONSISTENCY_DECAY_RATE=1.0
    EXPLORATION_DECAY_RATE=1.0
fi

# ======================== Print Config ========================
echo "==================================================================="
echo "  ReBel WebShop experiment: ${EXP_ID} - ${EXP_NAME}"
echo "==================================================================="
echo ""
echo "Config:"
echo "  - ID:               ${EXP_ID}"
echo "  - Name:             ${EXP_NAME}"
echo "  - Advantage:        ${ADV_ESTIMATOR}"
echo "  - Prompt:           $([ "$USE_REBEL_PROMPT" = "true" ] && echo '<belief>' || echo '<think>')"
echo "  - Belief reward:    ${USE_BELIEF_REWARD}"
echo "  - Result reward:    ${USE_RESULT_REWARD}"
echo ""
echo "Parameters:"
echo "  - Seed:             ${SEED}"
echo "  - GPUs:             ${NUM_GPUS}"
echo "  - Epochs:           ${EPOCHS}"
echo "  - Rollout N:        ${ROLLOUT_N}  (total procs: $((16 * ROLLOUT_N)))"
echo "  - Max response len: ${MAX_RESPONSE_LENGTH:-1536}  (was 512, SFT avg=1054)"
echo "  - Max prompt len:   ${MAX_PROMPT_LENGTH:-3000}"
echo "  - Model:            ${MODEL_PATH}"
echo "  - Results:          ${RESULTS_DIR}"
echo ""
echo "-------------------------------------------------------------------"

cd "${VERL_AGENT_DIR}"

# ======================== Prepare Data ========================
# val_batch_size=500 requires test.parquet to have exactly 500 rows.
# Check row count and regenerate if needed.
_EXPECTED_VAL_ROWS=500
_TEST_PARQUET="${VERL_AGENT_DIR}/processed_data/rl/webshop/test.parquet"
_NEED_REGEN=false
if [ ! -f "${VERL_AGENT_DIR}/processed_data/rl/webshop/train.parquet" ]; then
    _NEED_REGEN=true
elif [ -f "$_TEST_PARQUET" ]; then
    _ACTUAL_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$_TEST_PARQUET')))" 2>/dev/null || echo 0)
    if [ "$_ACTUAL_ROWS" != "$_EXPECTED_VAL_ROWS" ]; then
        echo "[Data] test.parquet has ${_ACTUAL_ROWS} rows (need ${_EXPECTED_VAL_ROWS}). Regenerating..."
        _NEED_REGEN=true
    fi
else
    _NEED_REGEN=true
fi
if [ "$_NEED_REGEN" = "true" ]; then
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size 16 \
        --val_data_size ${_EXPECTED_VAL_ROWS} 2>/dev/null || true
fi

# ======================== Compute save/test frequency ========================
# With train_data_size=16 and train_batch_size=16, steps_per_epoch=1.
_TRAIN_PARQUET="${VERL_AGENT_DIR}/processed_data/rl/webshop/train.parquet"
_WEB_TRAIN_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$_TRAIN_PARQUET')))" 2>/dev/null || echo 16)
_WEB_STEPS_PER_EPOCH=$(python3 -c "import math; print(max(1, math.ceil(${_WEB_TRAIN_ROWS}/16)))")
_WEB_SAVE_FREQ=$(( 10 * _WEB_STEPS_PER_EPOCH ))   # save every 10 epochs
_WEB_TEST_FREQ=$(( 5  * _WEB_STEPS_PER_EPOCH ))   # validate every 5 epochs
echo "[Schedule] WebShop: train_rows=${_WEB_TRAIN_ROWS}, steps/epoch=${_WEB_STEPS_PER_EPOCH}"
echo "[Schedule] save_freq=${_WEB_SAVE_FREQ} steps (every 10 epochs), test_freq=${_WEB_TEST_FREQ} steps (every 5 epochs)"

# ======================== Build Training Command ========================
# Response length: SFT data averages ~1054 tokens (p90=1436).
# Previously 512 which truncated 99.8% of outputs before <action> tag.
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-6144}   # webshop prompts: search page ~1500, detail page ~3500, history累积可达4000+; 5120偶尔溢出(实测5272)

BASE_ARGS=(
    "algorithm.adv_estimator=${ADV_ESTIMATOR}"
    "data.train_files=${VERL_AGENT_DIR}/processed_data/rl/webshop/train.parquet"
    "data.val_files=${VERL_AGENT_DIR}/processed_data/rl/webshop/test.parquet"
    "data.train_batch_size=16"
    "data.val_batch_size=500"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
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
    "actor_rollout_ref.actor.ppo_mini_batch_size=128"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.8"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.free_cache_engine=False"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "algorithm.use_kl_in_reward=False"
    "algorithm.kl_penalty=kl"
    "algorithm.kl_ctrl.type=fixed"
    "algorithm.kl_ctrl.kl_coef=0.001"
    "env.env_name=Webshop"
    "env.seed=${SEED}"
    "env.max_steps=15"
    "env.rollout.n=${ROLLOUT_N}"
    "trainer.critic_warmup=0"
    "trainer.logger=['console','swanlab']"
    "trainer.project_name=ReBel_WebShop"
    "trainer.experiment_name=${FULL_EXP_NAME}"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "trainer.nnodes=1"
    "trainer.save_freq=${_WEB_SAVE_FREQ}"
    "trainer.test_freq=${_WEB_TEST_FREQ}"
    "trainer.total_epochs=${EPOCHS}"
    "trainer.default_local_dir=${RESULTS_DIR}/checkpoints"
    "trainer.val_before_train=True"
    "trainer.max_actor_ckpt_to_keep=3"
    "+trainer.save_trajectories=${SAVE_TRAJECTORIES:-false}"
    "+trainer.trajectory_save_dir=${VERL_AGENT_DIR}/results/trajectories/${TIMESTAMP}_${FULL_EXP_NAME}"
)

# Invalid action penalty
if [ "$INVALID_ACTION_PENALTY" = "True" ]; then
    BASE_ARGS+=(
        "actor_rollout_ref.actor.use_invalid_action_penalty=True"
        "actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF}"
    )
fi

# ReBel prompt format
if [ "$USE_REBEL_PROMPT" = "true" ]; then
    BASE_ARGS+=(
        "algorithm.rebel.enable=True"
    )

    BASE_ARGS+=(
        "algorithm.rebel.belief_granularity=gt_phase"
        "algorithm.rebel.step_advantage_w=${STEP_ADV_W}"
        "algorithm.rebel.mode=mean_norm"
        "algorithm.rebel.task_aware_grouping=false"
        "algorithm.rebel.per_task_normalization=false"
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

    BASE_ARGS+=(
        "algorithm.rebel.min_obs_group_size=2"
    )

    BASE_ARGS+=(
        "algorithm.rebel.use_belief_reward=${USE_BELIEF_REWARD}"
        "algorithm.rebel.use_result_reward=${USE_RESULT_REWARD}"
    )

    BASE_ARGS+=(
        "algorithm.rebel.use_task_weighting=${USE_TASK_WEIGHTING}"
        "algorithm.rebel.weight_alpha=${WEIGHT_ALPHA}"
        "algorithm.rebel.weight_min=${WEIGHT_MIN}"
        "algorithm.rebel.weight_max=${WEIGHT_MAX}"
        "algorithm.rebel.weight_baseline_sr=0.85"
        "algorithm.rebel.task_weighting_warmup_epochs=${TASK_WEIGHTING_WARMUP}"
    )

    BASE_ARGS+=(
        "algorithm.rebel.belief_reward_decay.enable=${USE_BELIEF_DECAY}"
        "algorithm.rebel.belief_reward_decay.method=${DECAY_METHOD}"
        "algorithm.rebel.belief_reward_decay.warmup_epochs=${DECAY_WARMUP_EPOCHS}"
        "algorithm.rebel.belief_reward_decay.decay_start_epoch=${DECAY_START_EPOCH}"
        "algorithm.rebel.belief_reward_decay.decay_end_epoch=${DECAY_END_EPOCH}"
        "algorithm.rebel.belief_reward_decay.min_weight=${DECAY_MIN_WEIGHT}"
        "algorithm.rebel.belief_reward_decay.adaptive=${USE_ADAPTIVE_DECAY}"
        "algorithm.rebel.belief_reward_decay.target_sr=${DECAY_TARGET_SR}"
        "algorithm.rebel.belief_reward_decay.alpha=${DECAY_ALPHA}"
        "algorithm.rebel.belief_reward_decay.progress_decay_rate=${PROGRESS_DECAY_RATE}"
        "algorithm.rebel.belief_reward_decay.consistency_decay_rate=${CONSISTENCY_DECAY_RATE}"
        "algorithm.rebel.belief_reward_decay.exploration_decay_rate=${EXPLORATION_DECAY_RATE}"
    )
else
    BASE_ARGS+=(
        "algorithm.rebel.enable=False"
    )
fi

# GiGPO / HiBO specific parameters
if [ "$ADV_ESTIMATOR" = "gigpo" ] || [ "$ADV_ESTIMATOR" = "rebel" ]; then
    BASE_ARGS+=(
        "algorithm.gamma=0.95"
        "algorithm.gigpo.step_advantage_w=${STEP_ADV_W}"
        "algorithm.gigpo.mode=mean_norm"
    )
fi

# ======================== Execute Training ========================
echo ""
echo "Starting training..."
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
echo "==================================================================="
echo "  Experiment complete: ${EXP_ID} - ${EXP_NAME} (seed=${SEED})"
echo "==================================================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
