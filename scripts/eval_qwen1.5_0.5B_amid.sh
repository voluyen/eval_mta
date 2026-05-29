#! /bin/bash
# Đánh giá TẤT CẢ các checkpoint trong CKPT_ROOT, chia song song trên nhiều GPU.
# Chế độ nạp model (full / lora) đặt qua biến PEFT bên dưới.
# Bỏ qua checkpoint nào đã đánh giá xong (đủ 4 metric ROUGE-L).

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
# Thư mục cha chứa các checkpoint của một experiment (mỗi checkpoint là 1 step).
CKPT_ROOT="checkpoints/amid_mta_ckpt/qwen1.5-0.5B#amid/ab_pr_0.5_0.5_16_1e-4"
# Chế độ: "full" (checkpoint là full-model) hoặc "lora" (checkpoint là adapter)
PEFT="full"
# Base model dùng khi PEFT="lora" (full-model thì bỏ qua)
BASE_MODEL="Qwen/Qwen1.5-0.5B"
TOKENIZER="Qwen/Qwen1.5-0.5B"
BATCH_SIZE=64
TASK="qwen_amid_mta"
# Danh sách GPU chạy song song
DEVICES=("cuda:0" "cuda:1")
# DEVICES=("cuda:0")

# ==== Hàm đánh giá 1 checkpoint trên 1 device ====
eval_ckpt() {
    local CKPT_DIR="$1"
    local DEVICE="$2"

    local CKPT_NAME=$(basename "${CKPT_DIR}")
    local CKPT_REL="${CKPT_DIR#${CKPT_ROOT}/}"
    local OUTPUT_DIR="${BASE_PATH}/eval_outputs/${TASK}/${CKPT_REL}"
    mkdir -p "${OUTPUT_DIR}"

    local LOG_FILE="${OUTPUT_DIR}/eval.log"
    if [ -f "${LOG_FILE}" ]; then
        local DONE=$(grep -cE "^(S-NI|Dolly|Self-Instruct|Vicuna) ROUGE-L F1:" "${LOG_FILE}" 2>/dev/null || echo 0)
        if [ "${DONE}" -eq 4 ]; then
            echo "  [${DEVICE} | ${CKPT_NAME}] SKIP (đã đánh giá xong)"
            return
        else
            echo "  [${DEVICE} | ${CKPT_NAME}] Log không đầy đủ (${DONE}/4) — chạy lại"
            rm -f "${LOG_FILE}"
        fi
    fi

    local OPTS=""
    OPTS+=" --train_data ${BASE_PATH}/data/dolly/train.jsonl"
    OPTS+=" --val_data ${BASE_PATH}/data/dolly/dev.jsonl"
    OPTS+=" --test_data ${BASE_PATH}/data/dolly/valid.jsonl"
    OPTS+=" --teacher_layers_mapping 32"
    OPTS+=" --student_encoder_layers_finetuned 22"
    OPTS+=" --val_batch_size ${BATCH_SIZE}"
    OPTS+=" --student_device ${DEVICE}"
    OPTS+=" --output_dir ${OUTPUT_DIR}"
    OPTS+=" --seed ${SEED}"
    OPTS+=" --tokenizer ${TOKENIZER}"

    # Nạp model theo chế độ PEFT
    if [ "${PEFT}" = "lora" ]; then
        OPTS+=" --model_path ${BASE_MODEL}"
        OPTS+=" --lora_path ${CKPT_DIR}"
    else
        OPTS+=" --model_path ${CKPT_DIR}"
    fi

    echo "  [${DEVICE} | ${CKPT_NAME}] Đang đánh giá -> ${LOG_FILE}"
    python src/run_eval.py ${OPTS} >> "${LOG_FILE}" 2>&1

    local EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "  [${DEVICE} | ${CKPT_NAME}] DONE ✓"
    else
        echo "  [${DEVICE} | ${CKPT_NAME}] FAILED (exit ${EXIT_CODE}) — xem ${LOG_FILE}"
    fi
}

# ==== Worker: chạy tuần tự danh sách checkpoint được giao cho 1 GPU ====
worker() {
    local DEVICE="$1"; shift
    for CKPT_DIR in "$@"; do
        eval_ckpt "${CKPT_DIR}" "${DEVICE}"
    done
    echo "  [${DEVICE}] === đã xong tất cả checkpoint được giao ==="
}

echo "======================================================"
echo " Đánh giá tất cả checkpoint trong: ${CKPT_ROOT}"
echo " Batch: ${BATCH_SIZE} | GPU: ${DEVICES[*]}"
echo "======================================================"

# Lấy danh sách checkpoint theo step, tương tự eval_opt_2.7B.sh.
mapfile -t CKPTS < <(
    find "${CKPT_ROOT}" -maxdepth 1 -mindepth 1 -type d | \
    while read d; do
        if [ -f "$d/config.json" ] || [ -f "$d/adapter_config.json" ]; then
            echo "$d"
        fi
    done | sort -t/ -k1 -rV
)

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "Không tìm thấy checkpoint nào trong ${CKPT_ROOT}"
    exit 1
fi

NGPU=${#DEVICES[@]}

# Chia checkpoint round-robin cho từng GPU rồi chạy song song
PIDS=()
for ((g=0; g<NGPU; g++)); do
    SUBSET=()
    for ((i=g; i<${#CKPTS[@]}; i+=NGPU)); do
        SUBSET+=("${CKPTS[$i]}")
    done
    worker "${DEVICES[$g]}" "${SUBSET[@]}" &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

echo ""
echo "======================================================"
echo " Hoàn thành đánh giá tất cả checkpoint trên ${NGPU} GPU."
echo "======================================================"
