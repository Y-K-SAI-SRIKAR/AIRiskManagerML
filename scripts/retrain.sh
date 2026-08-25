#!/usr/bin/env bash

set -e

echo "=========================================="
echo "========== RETRAIN PIPELINE =============="
echo "=========================================="

python -m src.training.retrain

echo ""
echo "Retraining and promotion pipeline completed successfully."