#!/bin/bash
# =============================================================================
# Trajectory annotation — ALFWorld
# =============================================================================
# Stage 1 of the SFT cold-start pipeline: use a teacher LLM to annotate raw
# expert trajectories with structured <belief>/<think>/<action> segments.
#
# Usage:
#   bash scripts/annotate_alfworld.sh
#
# Override via env vars:
#   TEACHER_API_BASE  — OpenAI-compatible endpoint (default: env-set)
#   TEACHER_API_KEY   — API key
#   TEACHER_MODEL     — teacher model name (default: gpt-4o)
#   TEACHER_WORKERS   — concurrent workers (default: 8)
# =============================================================================

set -e

VERL_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${VERL_AGENT_DIR}:${PYTHONPATH:-}"

TEACHER_API_BASE=${TEACHER_API_BASE:-https://api.openai.com/v1}
TEACHER_API_KEY=${TEACHER_API_KEY:-<set-your-api-key>}
TEACHER_MODEL=${TEACHER_MODEL:-gpt-4o}
TEACHER_WORKERS=${TEACHER_WORKERS:-8}

python3 "${VERL_AGENT_DIR}/scripts/generate_rebel_sft_data.py" \
    --stage both \
    --env alfworld \
    --input     "${VERL_AGENT_DIR}/data/alfworld_rebel/rebel_coldstart.json" \
    --annotated "${VERL_AGENT_DIR}/data/alfworld_sft_annotated/annotated_trajs.jsonl" \
    --output    "${VERL_AGENT_DIR}/data/alfworld_sft/train.jsonl" \
    --api_base  "${TEACHER_API_BASE}" \
    --api_key   "${TEACHER_API_KEY}" \
    --model     "${TEACHER_MODEL}" \
    --workers   "${TEACHER_WORKERS}"
