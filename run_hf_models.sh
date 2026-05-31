#! /bin/bash
# Evaluate HuggingFace models directly (no checkpoint download needed).
# Output is stored under eval_outputs/hf_models/<org>/<model_name>/ to avoid
# any collision with residual/rerun/amid outputs.

set -euo pipefail

SEED=42
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# run_eval.py uses relative paths ./data/... so must run from BASE_PATH
cd "${BASE_PATH}"
OUTPUT_ROOT="${BASE_PATH}/eval_outputs/hf_models"

DEVICE_LIST="${DEVICE_LIST:-cuda:1}"
read -r -a DEVICES <<< "${DEVICE_LIST}"

# Model list: "<hf_model_id>|<batch_size>"
# Batch rules: <1B → 128, <2B → 64, else → 32
HF_MODELS=(
    "HoangTran223/MCW_KD_GPTXL_SFT|16"
    "HoangTran223/MCW_KD_TinyLLama_SFT|16"
    "MiniLLM/SFT-gpt2-120M|128"
    "MiniLLM/SFT-gpt2-340M|128"
    "MiniLLM/SFT-OPT-1.3B|64"
    "MiniLLM/teacher-gpt2-1.5B|64"
    "VoCuc/Qwen1.5_1.8B_SFT|64"
    "MiniLLM/SFT-OPT-2.7B|32"
    "VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT|16"
    "VoCuc/Mistral7B_Dolly_SFT|16"
    "MiniLLM/SFT-OPT-6.7B|16"
)

if ! command -v python >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        set +u
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate mta
        set -u
    fi
fi

is_eval_done() {
    local log_file="$1"
    local done_count=0
    if [ -f "${log_file}" ]; then
        done_count="$(grep -cE "^(S-NI|Dolly|Self-Instruct|Vicuna) ROUGE-L F1:" "${log_file}" 2>/dev/null || true)"
    fi
    [ "${done_count:-0}" -eq 4 ]
}

eval_hf_model() {
    local model_id="$1"
    local batch_size="$2"
    local device="$3"

    # Convert "org/name" → "org/name" for directory (keep slash as subdir)
    local output_dir="${OUTPUT_ROOT}/${model_id}"
    local log_file="${output_dir}/eval.log"

    mkdir -p "${output_dir}"

    if is_eval_done "${log_file}"; then
        echo "  [${device} | ${model_id}] SKIP (already evaluated)"
        return 0
    fi

    if [ -f "${log_file}" ]; then
        echo "  [${device} | ${model_id}] Incomplete log, re-running"
        rm -f "${log_file}"
    fi

    local opts=""
    opts+=" --train_data ${BASE_PATH}/data/dolly/train.jsonl"
    opts+=" --val_data ${BASE_PATH}/data/dolly/dev.jsonl"
    opts+=" --test_data ${BASE_PATH}/data/dolly/valid.jsonl"
    opts+=" --teacher_layers_mapping 32"
    opts+=" --student_encoder_layers_finetuned 22"
    opts+=" --val_batch_size ${batch_size}"
    opts+=" --student_device ${device}"
    opts+=" --output_dir ${output_dir}"
    opts+=" --seed ${SEED}"
    opts+=" --model_path ${model_id}"
    opts+=" --tokenizer ${model_id}"

    echo "  [${device} | ${model_id}] batch=${batch_size} -> ${log_file}"
    set +e
    python "${BASE_PATH}/src/run_eval.py" ${opts} >> "${log_file}" 2>&1
    local exit_code=$?
    set -e

    if [ ${exit_code} -eq 0 ]; then
        echo "  [${device} | ${model_id}] DONE"
    else
        echo "  [${device} | ${model_id}] FAILED (exit ${exit_code}) - see ${log_file}"
    fi
    return ${exit_code}
}

worker() {
    local device="$1"
    local failed=0
    shift
    local entry model_id batch_size
    for entry in "$@"; do
        model_id="${entry%%|*}"
        batch_size="${entry##*|}"
        if ! eval_hf_model "${model_id}" "${batch_size}" "${device}"; then
            failed=1
        fi
    done
    echo "  [${device}] === done all assigned models ==="
    return ${failed}
}

echo ""
echo "======================================================"
echo " Evaluate HuggingFace models"
echo " Output root : ${OUTPUT_ROOT}"
echo " GPUs        : ${DEVICES[*]}"
echo " Models      : ${#HF_MODELS[@]}"
echo "======================================================"

NGPU=${#DEVICES[@]}
PIDS=()
for ((g=0; g<NGPU; g++)); do
    SUBSET=()
    for ((i=g; i<${#HF_MODELS[@]}; i+=NGPU)); do
        SUBSET+=("${HF_MODELS[$i]}")
    done
    if [ "${#SUBSET[@]}" -gt 0 ]; then
        worker "${DEVICES[$g]}" "${SUBSET[@]}" &
        PIDS+=("$!")
    fi
done

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAILED=1
    fi
done

if [ ${FAILED} -ne 0 ]; then
    echo "[ERROR] One or more model evals failed. Check the corresponding eval.log."
    exit 1
fi

echo ""
echo "======================================================"
echo " All HF model evaluations complete."
echo "======================================================"
