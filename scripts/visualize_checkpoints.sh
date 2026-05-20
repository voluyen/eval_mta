#! /bin/bash
# Trực quan hóa điểm ROUGE-L F1 của tất cả checkpoint trong eval_outputs/
# Chạy từ thư mục gốc dự án: bash eval_mta/scripts/visualize_checkpoints.sh

EVAL_ROOT="${1:-eval_mta/eval_outputs}"
OUTPUT_DIR="${2:-eval_mta/plots}"

echo "======================================================"
echo " Visualize checkpoints"
echo " Eval root : ${EVAL_ROOT}"
echo " Output dir: ${OUTPUT_DIR}"
echo "======================================================"

python eval_mta/scripts/visualize_checkpoints.py \
    --eval_root "${EVAL_ROOT}" \
    --output_dir "${OUTPUT_DIR}"
