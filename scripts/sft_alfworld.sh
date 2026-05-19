#!/bin/bash
# =============================================================================
# SFT cold-start pipeline — ALFWorld
# =============================================================================
# Stages:
#   1. Annotate raw trajectories with teacher LLM   → annotated_trajs.jsonl
#   2. Convert annotated trajs into SFT pairs       → train.jsonl
#   3. Preprocess SFT pairs to parquet              → train.parquet
#   4. SFT training (3 epochs)
#
# Usage: bash scripts/sft_alfworld.sh
# =============================================================================

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

VERL_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${VERL_AGENT_DIR}:${PYTHONPATH:-}"

# ─── Paths ────────────────────────────────────────────────────────────────────
ENV=alfworld
RAW_DATA="${VERL_AGENT_DIR}/data/alfworld_rebel/rebel_coldstart.json"
ANNOTATED_DATA="${VERL_AGENT_DIR}/data/alfworld_sft_annotated/annotated_trajs.jsonl"
SFT_PAIRS="${VERL_AGENT_DIR}/data/alfworld_sft/train.jsonl"
LOCAL_DIR="${VERL_AGENT_DIR}/processed_data/sft/alfworld"

# Backbone model (prefer local HF cache, otherwise pull from the Hub)
LOCAL_MODEL="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
if [ -d "$LOCAL_MODEL" ]; then
    MODEL_NAME="$LOCAL_MODEL"
else
    MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
fi

# SFT checkpoint — stored under project dir for portability
CHECKPOINT_DIR="${VERL_AGENT_DIR}/checkpoints/sft/alfworld/qwen1.5b_rebel_sft"
EXPERIMENT_NAME=qwen1.5b_rebel_cold-start_alfworld

# Teacher LLM for annotation (override via env vars)
TEACHER_API_BASE=${TEACHER_API_BASE:-https://api.openai.com/v1}
TEACHER_API_KEY=${TEACHER_API_KEY:-<set-your-api-key>}
TEACHER_MODEL=${TEACHER_MODEL:-gpt-4o}
TEACHER_WORKERS=${TEACHER_WORKERS:-8}

echo "==================== ALFWorld SFT Training (v3) ===================="
echo "  Project root:    $VERL_AGENT_DIR"
echo "  Raw data:        $RAW_DATA"
echo "  Annotated data:  $ANNOTATED_DATA"
echo "  SFT pairs:       $SFT_PAIRS"
echo "  SFT parquet:     $LOCAL_DIR"
echo "  Base model:      $MODEL_NAME"
echo "  Checkpoint:      $CHECKPOINT_DIR"
echo "====================================================================="

# ─── Stage 1: Annotate trajectories ──────────────────────────────────────────
if [ -f "$ANNOTATED_DATA" ]; then
    ANNOTATED_LINES=$(wc -l < "$ANNOTATED_DATA")
    echo ""
    echo "Stage 1: Annotation already exists ($ANNOTATED_LINES episodes). Skipping."
else
    echo ""
    echo "Stage 1: Annotating trajectories with teacher LLM ($TEACHER_MODEL)..."
    mkdir -p "$(dirname "$ANNOTATED_DATA")"

    python3 "${VERL_AGENT_DIR}/scripts/generate_rebel_sft_data.py" \
        --stage annotate \
        --env alfworld \
        --input "$RAW_DATA" \
        --annotated "$ANNOTATED_DATA" \
        --api_base "$TEACHER_API_BASE" \
        --api_key "$TEACHER_API_KEY" \
        --model "$TEACHER_MODEL" \
        --workers "$TEACHER_WORKERS"

    [ -f "$ANNOTATED_DATA" ] || { echo "Error: Annotation failed."; exit 1; }
    echo "✓ Annotation done: $(wc -l < "$ANNOTATED_DATA") episodes"
fi

# ─── Stage 2: Convert annotated trajs → SFT pairs ────────────────────────────
if [ -f "$SFT_PAIRS" ]; then
    SFT_LINES=$(wc -l < "$SFT_PAIRS")
    echo ""
    echo "Stage 2: SFT pairs already exist ($SFT_LINES pairs). Skipping."
else
    echo ""
    echo "Stage 2: Converting annotated trajs to SFT pairs..."
    mkdir -p "$(dirname "$SFT_PAIRS")"

    python3 "${VERL_AGENT_DIR}/scripts/generate_rebel_sft_data.py" \
        --stage convert \
        --env alfworld \
        --annotated "$ANNOTATED_DATA" \
        --output "$SFT_PAIRS"

    [ -f "$SFT_PAIRS" ] || { echo "Error: Conversion failed."; exit 1; }
    echo "✓ Conversion done: $(wc -l < "$SFT_PAIRS") SFT pairs"
fi

# ─── Stage 3: Preprocess to parquet ──────────────────────────────────────────
echo ""
echo "Stage 3: Preprocessing SFT pairs to parquet..."
mkdir -p "$LOCAL_DIR"

python3 -m examples.data_preprocess.cold_start_data \
    --local_dir="$LOCAL_DIR" \
    --data_source="$SFT_PAIRS"

[ -f "$LOCAL_DIR/train.parquet" ] || { echo "Error: Preprocessing failed."; exit 1; }
echo "✓ Preprocessing done: $LOCAL_DIR/train.parquet"

# ─── Stage 4: SFT training ───────────────────────────────────────────────────
echo ""
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-4}
echo "Stage 4: SFT training (3 epochs) on ${NUM_GPUS} GPU(s)..."
mkdir -p "$CHECKPOINT_DIR"

# ─── SwanLab setup (mirrors RL scripts) ──────────────────────────────────────
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
if [ -z "${SWANLAB_API_KEY:-}" ]; then
    echo "WARNING: SWANLAB_API_KEY not set. SwanLab cloud logging will fail."
    echo "  export SWANLAB_API_KEY=<your_key>  # get from https://swanlab.cn"
    export SWANLAB_MODE=disabled
fi
if ss -tlnp 2>/dev/null | grep -q ":7890"; then
    export http_proxy=http://127.0.0.1:7890
    export https_proxy=http://127.0.0.1:7890
    export HTTP_PROXY=http://127.0.0.1:7890
    export HTTPS_PROXY=http://127.0.0.1:7890
    export no_proxy=127.0.0.1,localhost
    echo "[Proxy] Clash detected on :7890, proxy enabled"
fi
export SWANLAB_LOG_DIR="${CHECKPOINT_DIR}/swanlog"

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${LOCAL_DIR}/train.parquet" \
    data.val_files="${LOCAL_DIR}/train.parquet" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=4096 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain="${MODEL_NAME}" \
    trainer.default_hdfs_dir=null \
    trainer.project_name=ReBel_SFT \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.logger=['console','swanlab'] \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true

echo ""
echo "✓ SFT training completed."
echo "  Checkpoint: $CHECKPOINT_DIR"

# ─── Find latest checkpoint ──────────────────────────────────────────────────
LATEST_CKPT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
    | sort -V | tail -1)

echo ""
echo "==================== SFT Complete ===================="
echo "  Base model:       $MODEL_NAME"
echo "  SFT checkpoint:   ${LATEST_CKPT:-$CHECKPOINT_DIR}"
echo "  (Pass to RL via MODEL_PATH=<above>)"
echo "======================================================="
