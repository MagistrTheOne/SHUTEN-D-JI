#!/bin/bash
# SHUTEN-DŌJI Training Script via LLaMA Factory
# Team: NULLXES DAI
#
# Prerequisites:
#   pip install llamafactory
#   Data generated via: python scripts/generate_data.py --export
#
# Usage:
#   bash scripts/train.sh phase1    # supervised trajectory learning
#   bash scripts/train.sh phase2    # tool-use learning
#   bash scripts/train.sh phase3    # DPO preference optimization
#   bash scripts/train.sh phase4    # PPO reinforcement learning

set -e

PHASE=${1:-"phase1"}
echo "[SHUTEN-DŌJI] Training Phase: $PHASE"
echo "[NULLXES DAI] ════════════════════════════"

case $PHASE in
    phase1)
        echo "Phase 1: Supervised Trajectory Learning"
        llamafactory-cli train configs/training/phase1_sft.yaml
        ;;
    phase2)
        echo "Phase 2: Tool-Use Learning"
        llamafactory-cli train configs/training/phase2_tools.yaml
        ;;
    phase3)
        echo "Phase 3: DPO Preference Optimization"
        llamafactory-cli train configs/training/phase3_dpo.yaml
        ;;
    phase4)
        echo "Phase 4: PPO Reinforcement Learning"
        llamafactory-cli train configs/training/phase4_ppo.yaml
        ;;
    all)
        echo "Running all phases sequentially..."
        bash scripts/train.sh phase1
        bash scripts/train.sh phase2
        bash scripts/train.sh phase3
        bash scripts/train.sh phase4
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: bash scripts/train.sh [phase1|phase2|phase3|phase4|all]"
        exit 1
        ;;
esac

echo "[SHUTEN-DŌJI] Phase $PHASE complete."
