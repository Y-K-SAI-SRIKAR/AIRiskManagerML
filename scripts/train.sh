#!/usr/bin/env bash

set -e

echo "=========================================="
echo "========== TRAINING PIPELINE ============="
echo "=========================================="

python -m src.training.train

echo ""
echo "Training pipeline completed successfully."