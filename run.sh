#! /bin/bash
# Download rerun checkpoints from Hugging Face, then evaluate the final epoch10
# checkpoint of each experiment.

set -euo pipefail

SEED=42
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_REPO="VoCuc/mta-dskd"
DOWNLOAD_ROOT="${BASE_PATH}/checkpoints"
OUTPUT_ROOT="${BASE_PATH}/eval_outputs"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICES=("${DEVICES[@]:-cuda:2}")
DWA_CKPTS=(
    "dwa_mta/gpt2/gpt2-medium/span_dwa_kd/criterion=dwa_kd__skewed_reverse_kl-bf16__teacher=Qwen1.5_1.8B_SFT_Dolly__kd^rate=0.5__kd^temp=2.0__epoch=10__bsz=16x2x1=32__lr=0.0005__proj^lr=0.001/epoch10_step3570"
    "dwa_mta/gpt2/gpt2-xl/span_dwa_kd/criterion=dwa_kd__skewed_reverse_kl-lora-rank=256-alpha=8-dropout=0.1-bf16__teacher=Qwen2.5-7B-Instruct-Dolly-SFT__kd^rate=0.5__kd^temp=2.0__epoch=10__bsz=16x2x1=32__lr=0.001__proj^lr=0.001/epoch10_step3570"
    "dwa_mta/gpt2/gpt2-base/dwa_kd_word_level/criterion=dwa_kd__skewed_reverse_kl-bf16__teacher=Qwen1.5_1.8B_SFT_Dolly__kd^rate=0.5__kd^temp=2.0__epoch=10__bsz=16x2x1=32__lr=0.0005__proj^lr=0.001/epoch10_step3570"
)
CHECKPOINT_ROOTS=(
    "${DOWNLOAD_ROOT}/rerun"
    "${DOWNLOAD_ROOT}/dwa_mta"
)

if ! command -v python >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        # Make the script usable from a fresh shell where conda is not active.
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate mta
    fi
fi

install_requirements() {
    echo "======================================================"
    echo " Install requirements"
    echo " File: ${BASE_PATH}/requirements.txt"
    echo "======================================================"

    python -m pip install -r "${BASE_PATH}/requirements.txt"
}

check_hf_cli() {
    if ! command -v hf >/dev/null 2>&1; then
        echo "[ERROR] Không tìm thấy lệnh hf sau khi install requirements."
        echo "Hãy kiểm tra env Python hoặc chạy: conda activate mta"
        exit 1
    fi
}

download_checkpoints() {
    echo "======================================================"
    echo " Download checkpoints"
    echo " Repo       : ${HF_REPO}"
    echo " HF folders : rerun/ + selected dwa_mta checkpoints"
    echo " Local root : ${DOWNLOAD_ROOT}"
    echo "======================================================"

    local include_args=("--include" "rerun/**")
    for ckpt_path in "${DWA_CKPTS[@]}"; do
        include_args+=("--include" "${ckpt_path}/**")
    done

    hf download "${HF_REPO}" "${include_args[@]}" \
        --local-dir "${DOWNLOAD_ROOT}" \
        --max-workers 4
}

is_eval_done() {
    local log_file="$1"
    local done_count=0
    if [ -f "${log_file}" ]; then
        done_count="$(grep -cE "^(S-NI|Dolly|Self-Instruct|Vicuna) ROUGE-L F1:" "${log_file}" 2>/dev/null || true)"
    fi
    [ "${done_count:-0}" -eq 4 ]
}

eval_ckpt() {
    local ckpt_dir="$1"
    local device="$2"
    local ckpt_rel="${ckpt_dir#${DOWNLOAD_ROOT}/}"
    local output_dir="${OUTPUT_ROOT}/${ckpt_rel}"
    local log_file="${output_dir}/eval.log"

    mkdir -p "${output_dir}"

    if is_eval_done "${log_file}"; then
        echo "  [${device} | ${ckpt_rel}] SKIP (đã đánh giá xong)"
        return
    fi

    if [ -f "${log_file}" ]; then
        echo "  [${device} | ${ckpt_rel}] Log chưa đủ metric, chạy lại"
        rm -f "${log_file}"
    fi

    local opts=""
    opts+=" --train_data ${BASE_PATH}/data/dolly/train.jsonl"
    opts+=" --val_data ${BASE_PATH}/data/dolly/dev.jsonl"
    opts+=" --test_data ${BASE_PATH}/data/dolly/valid.jsonl"
    opts+=" --teacher_layers_mapping 32"
    opts+=" --student_encoder_layers_finetuned 22"
    opts+=" --val_batch_size ${BATCH_SIZE}"
    opts+=" --student_device ${device}"
    opts+=" --output_dir ${output_dir}"
    opts+=" --seed ${SEED}"

    if [ -f "${ckpt_dir}/adapter_config.json" ]; then
        local base_model=""
        case "${ckpt_dir}" in
            *gpt2-xl*)
                base_model="openai-community/gpt2-xl"
                ;;
            *gpt2-medium*)
                base_model="openai-community/gpt2-medium"
                ;;
            *gpt2-base*|*gpt2*)
                base_model="openai-community/gpt2"
                ;;
            *opt-1.3B*|*opt-1.3b*)
                base_model="facebook/opt-1.3b"
                ;;
            *tinyllama*|*TinyLlama*)
                base_model="model_hub/tinyllama/tinyllama-1.1b-3T"
                ;;
        esac

        if [ -z "${base_model}" ]; then
            echo "  [${device} | ${ckpt_rel}] FAILED: Không xác định được base model cho LoRA checkpoint."
            return 1
        fi

        opts+=" --model_path ${base_model}"
        opts+=" --lora_path ${ckpt_dir}"
        opts+=" --tokenizer ${ckpt_dir}"
    else
        opts+=" --model_path ${ckpt_dir}"
        opts+=" --tokenizer ${ckpt_dir}"
    fi

    echo "  [${device} | ${ckpt_rel}] Đang đánh giá -> ${log_file}"
    set +e
    python "${BASE_PATH}/src/run_eval.py" ${opts} >> "${log_file}" 2>&1
    local exit_code=$?
    set -e

    if [ ${exit_code} -eq 0 ]; then
        echo "  [${device} | ${ckpt_rel}] DONE"
    else
        echo "  [${device} | ${ckpt_rel}] FAILED (exit ${exit_code}) - xem ${log_file}"
    fi
    return ${exit_code}
}

worker() {
    local device="$1"
    local failed=0
    shift
    for ckpt_dir in "$@"; do
        if ! eval_ckpt "${ckpt_dir}" "${device}"; then
            failed=1
        fi
    done
    echo "  [${device}] === đã xong các checkpoint được giao ==="
    return ${failed}
}

install_requirements
check_hf_cli
download_checkpoints

echo ""
echo "======================================================"
echo " Find final checkpoints"
echo " Checkpoint roots:"
printf "  %s\n" "${CHECKPOINT_ROOTS[@]}"
echo " Output root    : ${OUTPUT_ROOT}"
echo " Batch          : ${BATCH_SIZE}"
echo " GPUs           : ${DEVICES[*]}"
echo "======================================================"

mapfile -t CKPTS < <(
    DOWNLOAD_ROOT="${DOWNLOAD_ROOT}" CHECKPOINT_ROOTS="$(printf "%s\n" "${CHECKPOINT_ROOTS[@]}")" python - <<'PY'
import os
import re
from pathlib import Path

download_root = Path(os.environ["DOWNLOAD_ROOT"])
checkpoint_roots = [Path(p) for p in os.environ["CHECKPOINT_ROOTS"].splitlines() if p]
best = {}

for ckpt_root in checkpoint_roots:
    if not ckpt_root.exists():
        continue

    for marker_name in ("config.json", "adapter_config.json"):
        for marker in ckpt_root.rglob(marker_name):
            ckpt_dir = marker.parent
            match = re.match(r"epoch(9|10)(?:_|$)", ckpt_dir.name)
            if not match:
                continue

            epoch = int(match.group(1))
            exp_dir = ckpt_dir.parent
            rel = ckpt_dir.relative_to(download_root)
            prev = best.get(exp_dir)
            if prev is None or epoch > prev[0]:
                best[exp_dir] = (epoch, rel, ckpt_dir)

for _, _, ckpt_dir in sorted(best.values(), key=lambda item: str(item[1])):
    print(ckpt_dir)
PY
)

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "Không tìm thấy checkpoint epoch10/epoch9 nào trong ${CKPT_ROOT}"
    exit 1
fi

printf "Tìm thấy %d checkpoint cuối (ưu tiên epoch10, fallback epoch9):\n" "${#CKPTS[@]}"
printf "  %s\n" "${CKPTS[@]}"

NGPU=${#DEVICES[@]}
PIDS=()
for ((g=0; g<NGPU; g++)); do
    SUBSET=()
    for ((i=g; i<${#CKPTS[@]}; i+=NGPU)); do
        SUBSET+=("${CKPTS[$i]}")
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
    echo "[ERROR] Có ít nhất một checkpoint eval bị lỗi. Xem eval.log tương ứng."
    exit 1
fi

echo ""
echo "======================================================"
echo " Hoàn thành đánh giá checkpoint epoch10."
echo "======================================================"
