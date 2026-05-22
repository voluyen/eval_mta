#! /bin/bash
# Đánh giá TẤT CẢ các checkpoint trong thư mục LoRA của distillm 1.3B<-6.7B,
# CHIA SONG SONG TRÊN 2 GPU (mỗi GPU đánh giá một nửa số checkpoint).
# Bỏ qua checkpoint nào đã đánh giá xong (đủ 4 metric ROUGE-L).

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="facebook/opt-1.3b"
TOKENIZER="facebook/opt-1.3b"
# Thư mục cha chứa các checkpoint (mỗi checkpoint là 1 thư mục con tên là số bước)
LORA_ROOT="distillm-master/results/opt/train/distillm_1.3B_6.7B"
BATCH_SIZE=128
# Danh sách GPU dùng để chạy song song
# DEVICES=("cuda:0" "cuda:1")
DEVICES=("cuda:0")

# ==== Hàm đánh giá 1 checkpoint trên 1 device ====
eval_ckpt() {
    local CKPT_DIR="$1"
    local DEVICE="$2"

    local CKPT_STEP=$(basename "${CKPT_DIR}")
    local OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}-distillm-${CKPT_STEP}"
    mkdir -p "${OUTPUT_DIR}"

    local LOG_FILE="${OUTPUT_DIR}/eval.log"
    if [ -f "${LOG_FILE}" ]; then
        local DONE=$(grep -cE "^(S-NI|Dolly|Self-Instruct|Vicuna) ROUGE-L F1:" "${LOG_FILE}" 2>/dev/null || echo 0)
        if [ "${DONE}" -eq 4 ]; then
            echo "  [${DEVICE} | Step ${CKPT_STEP}] SKIP (đã đánh giá xong)"
            return
        else
            echo "  [${DEVICE} | Step ${CKPT_STEP}] Log không đầy đủ (${DONE}/4) — chạy lại"
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
    OPTS+=" --model_path ${MODEL_PATH}"
    OPTS+=" --lora_path ${CKPT_DIR}"
    OPTS+=" --tokenizer ${TOKENIZER}"

    echo "  [${DEVICE} | Step ${CKPT_STEP}] Đang đánh giá -> ${LOG_FILE}"
    python src/run_eval.py ${OPTS} >> "${LOG_FILE}" 2>&1

    local EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "  [${DEVICE} | Step ${CKPT_STEP}] DONE ✓"
    else
        echo "  [${DEVICE} | Step ${CKPT_STEP}] FAILED (exit ${EXIT_CODE}) — xem ${LOG_FILE}"
    fi
}

# ==== Worker: chạy tuần tự danh sách checkpoint được giao cho 1 GPU ====
worker() {
    local DEVICE="$1"
    shift
    for CKPT_DIR in "$@"; do
        eval_ckpt "${CKPT_DIR}" "${DEVICE}"
    done
    echo "  [${DEVICE}] === đã xong tất cả checkpoint được giao ==="
}

echo "======================================================"
echo " Đánh giá tất cả checkpoint trong: ${LORA_ROOT}"
echo " Model: ${MODEL_PATH} | Batch: ${BATCH_SIZE}"
echo " GPU song song: ${DEVICES[*]}"
echo "======================================================"

# Lấy danh sách checkpoint (tên là số nguyên), sắp xếp theo số bước
mapfile -t CKPTS < <(find "${LORA_ROOT}" -maxdepth 1 -mindepth 1 -type d \
    | grep -E '/[0-9]+$' | sort -t/ -k1 -V)

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "Không tìm thấy checkpoint nào trong ${LORA_ROOT}"
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

# Chờ tất cả worker hoàn thành
for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

echo ""
echo "======================================================"
echo " Hoàn thành đánh giá tất cả checkpoint trên ${NGPU} GPU."
echo "======================================================"
