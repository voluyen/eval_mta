#! /bin/bash
# Đánh giá tất cả các checkpoint trong thư mục results/
# Tự động phát hiện model, tokenizer, và LoRA từ args.json của mỗi experiment

SEED=42
BASE_PATH=.
RESULTS_DIR="${BASE_PATH}/results"
BATCH_SIZE=256
DEVICE="cuda:0"

# Map model_path → tokenizer HuggingFace
get_tokenizer() {
    local model_path="$1"
    case "$model_path" in
        *gpt2*)         echo "openai-community/gpt2" ;;
        *opt-1.3b*)     echo "facebook/opt-1.3b" ;;
        *opt-6.7b*)     echo "facebook/opt-6.7b" ;;
        *Qwen1.5-0.5B*) echo "Qwen/Qwen1.5-0.5B" ;;
        *Qwen1.5-1.8B*) echo "Qwen/Qwen1.5-1.8B" ;;
        *llama*)        echo "$model_path" ;;
        *)              echo "$model_path" ;;
    esac
}

echo "======================================================"
echo " Bắt đầu đánh giá tất cả checkpoint trong: ${RESULTS_DIR}"
echo " Batch size: ${BATCH_SIZE} | Device: ${DEVICE}"
echo "======================================================"

# Tìm tất cả file args.json trong results (mỗi file tương ứng 1 experiment)
find "$RESULTS_DIR" -name "args.json" | sort | while read ARGS_FILE; do
    TRAIN_DIR=$(dirname "$ARGS_FILE")

    # Đọc thông tin từ args.json
    BASE_MODEL=$(python3 -c "
import json, sys
d = json.load(open('$ARGS_FILE'))
print(d.get('model_path', ''))
" 2>/dev/null)

    PEFT=$(python3 -c "
import json, sys
d = json.load(open('$ARGS_FILE'))
v = d.get('peft', None)
print(v if v and v != 'None' else '')
" 2>/dev/null)

    TOKENIZER=$(get_tokenizer "$BASE_MODEL")

    echo ""
    echo "------------------------------------------------------"
    echo " Experiment : ${TRAIN_DIR}"
    echo " Base model : ${BASE_MODEL}"
    echo " PEFT       : ${PEFT:-none}"
    echo " Tokenizer  : ${TOKENIZER}"
    echo "------------------------------------------------------"

    # Tìm tất cả thư mục checkpoint (tên là số nguyên) trong experiment
    find "$TRAIN_DIR" -maxdepth 1 -mindepth 1 -type d | \
        grep -E '/[0-9]+$' | sort -t/ -k1 -V | while read CKPT_DIR; do

        CKPT_STEP=$(basename "$CKPT_DIR")
        # Đường dẫn tương đối từ BASE_PATH
        CKPT_REL="${CKPT_DIR#${BASE_PATH}/}"
        EXP_REL="${TRAIN_DIR#${BASE_PATH}/}"
        OUTPUT_DIR="${BASE_PATH}/eval_outputs/${EXP_REL}/${CKPT_STEP}"

        mkdir -p "${OUTPUT_DIR}"

        # Kiểm tra checkpoint đã đánh giá đầy đủ chưa (đủ 4 dòng metric cuối)
        LOG_FILE="${OUTPUT_DIR}/eval.log"
        if [ -f "${LOG_FILE}" ]; then
            DONE=$(grep -cE "^(S-NI|Dolly|Self-Instruct|Vicuna) ROUGE-L F1:" "${LOG_FILE}" 2>/dev/null || echo 0)
            if [ "${DONE}" -eq 4 ]; then
                echo "  [Step ${CKPT_STEP}] SKIP (đã đánh giá xong)"
                continue
            else
                echo "  [Step ${CKPT_STEP}] Log không đầy đủ (${DONE}/4 metric) — chạy lại"
                rm -f "${LOG_FILE}"
            fi
        fi

        OPTS=""
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

        if [ "$PEFT" = "lora" ]; then
            # LoRA: base model là HF hub, checkpoint là adapter
            OPTS+=" --model_path ${BASE_MODEL}"
            OPTS+=" --lora_path ${CKPT_DIR}"
        else
            # Full model: checkpoint chứa toàn bộ model
            OPTS+=" --model_path ${CKPT_DIR}"
        fi

        echo ""
        echo "  [Step ${CKPT_STEP}] Đang đánh giá: ${CKPT_REL}"
        echo "  Output: ${OUTPUT_DIR}/eval.log"

        python src/run_eval.py ${OPTS} >> "${OUTPUT_DIR}/eval.log" 2>&1

        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "  [Step ${CKPT_STEP}] DONE ✓"
        else
            echo "  [Step ${CKPT_STEP}] FAILED (exit code: ${EXIT_CODE}) — xem log tại ${OUTPUT_DIR}/eval.log"
        fi
    done
done

echo ""
echo "======================================================"
echo " Hoàn thành đánh giá tất cả checkpoint."
echo "======================================================"
