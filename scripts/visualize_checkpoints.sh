#!/usr/bin/env bash
set -euo pipefail

# In bảng ROUGE-L F1 của tất cả checkpoint trong eval result.
#
# Cách chạy:
#   bash eval_mta/scripts/visualize_checkpoints.sh
#   bash eval_mta/scripts/visualize_checkpoints.sh /path/to/eval_results/qwen1.5-1.8B-to-gpt2-120M
#   bash eval_mta/scripts/visualize_checkpoints.sh /path/to/eval_results/qwen1.5-1.8B-to-gpt2-120M/mta_all_word
#
# Nếu chạy từ /mnt/phongdq/projects thì default bên dưới sẽ trỏ vào eval_results
# của ResidualKD_MTA/Multi-Level-OT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EVAL_ROOT="/mnt/phongdq/projects/ResidualKD_MTA/Multi-Level-OT/eval_results/qwen1.5-1.8B-to-gpt2-120M/"
EVAL_ROOT="${1:-${DEFAULT_EVAL_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "======================================================"
echo " Visualize checkpoints"
echo " Eval root : ${EVAL_ROOT}"
echo "======================================================"

"${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_checkpoints.py" \
    --eval_root "${EVAL_ROOT}"
