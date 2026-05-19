#!/bin/bash
# =============================================================================
# RL training — ALFWorld
# =============================================================================
# Thin wrapper that dispatches to a method-specific trainer in examples/.
#
# Usage:
#   bash scripts/rl_alfworld.sh                 # default: ReBel
#   METHOD=rebel bash scripts/rl_alfworld.sh    # ReBel (ours)
#   METHOD=grpo  bash scripts/rl_alfworld.sh    # GRPO baseline
#   METHOD=gigpo bash scripts/rl_alfworld.sh    # GiGPO baseline
#   SEED=42 METHOD=rebel bash scripts/rl_alfworld.sh
# =============================================================================

set -e

METHOD=${METHOD:-rebel}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "${METHOD}" in
    rebel) TRAINER_SCRIPT="${REPO_ROOT}/examples/rebel_trainer/run_alfworld.sh" ;;
    grpo)  TRAINER_SCRIPT="${REPO_ROOT}/examples/grpo_trainer/run_alfworld.sh"  ;;
    gigpo) TRAINER_SCRIPT="${REPO_ROOT}/examples/gigpo_trainer/run_alfworld.sh" ;;
    *) echo "ERROR: unknown METHOD='${METHOD}' (expected: rebel | grpo | gigpo)" >&2; exit 1 ;;
esac

echo "[RL] method=${METHOD}  env=alfworld  → ${TRAINER_SCRIPT}"
bash "${TRAINER_SCRIPT}"
