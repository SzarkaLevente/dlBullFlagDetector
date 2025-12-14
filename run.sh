#!/bin/bash

set -euo pipefail

echo "[run.sh] Starting ful pipeline run at $(date --iso-8601=seconds)"

python src/01-data-preprocessing.py
python src/02-training.py
python src/03-evaluation.py
python src/04-inference.py

echo "[run.sh] Pipeline finished ad $(date --iso-8601=seconds)"