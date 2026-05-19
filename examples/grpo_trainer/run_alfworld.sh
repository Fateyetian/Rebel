#!/bin/bash
# =============================================================================
# GRPO — ALFWorld (initialised from ReBel SFT checkpoint)
# =============================================================================
# Vanilla GRPO baseline with <think> prompting, episode-level advantage,
# symmetric clipping, no belief supervision.
#
# Usage:    bash run_alfworld.sh
# Seeded:   SEED=42 bash run_alfworld.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_ID=grpo \
EXP_NAME=grpo_alfworld \
ADV_ESTIMATOR=grpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=false \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/_base_alfworld.sh"
