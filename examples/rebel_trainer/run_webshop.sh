#!/bin/bash
# =============================================================================
# ReBel — WebShop
# =============================================================================
# Belief-anchor step grouping (HiBO) + belief-consistency reward
# + structured <belief>/<think>/<action> prompting.
#
# Usage:    bash run_webshop.sh
# Seeded:   SEED=42 bash run_webshop.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_ID=rebel \
EXP_NAME=rebel_webshop \
ADV_ESTIMATOR=rebel \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=true \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=true \
DECAY_METHOD=cosine \
USE_ADAPTIVE_DECAY=true \
USE_DIFFERENTIAL_DECAY=true \
bash "${SCRIPT_DIR}/_base_webshop.sh"
