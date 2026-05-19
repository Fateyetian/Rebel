#!/bin/bash
# =============================================================================
# GiGPO — ALFWorld (initialised from ReBel SFT checkpoint)
# =============================================================================
# Two-level grouping baseline (GiGPO, NeurIPS 2025) with <think> prompting:
# observation-hash step grouping on top of episode-level GRPO.
#
# Usage:    bash run_alfworld.sh
# Seeded:   SEED=42 bash run_alfworld.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_ID=gigpo \
EXP_NAME=gigpo_alfworld \
ADV_ESTIMATOR=gigpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=false \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/_base_alfworld.sh"
