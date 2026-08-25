#!/usr/bin/env bash

set -e

echo "=========================================="
echo "========== EVALUATION PIPELINE ==========="
echo "=========================================="

python -m src.evaluation.evaluate

echo ""
echo "Evaluation pipeline completed successfully."