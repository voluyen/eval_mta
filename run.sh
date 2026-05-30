#! /bin/bash
# Download rerun checkpoints from Hugging Face, then evaluate the final epoch10
# checkpoint of each experiment.

set -euo pipefail

SEED=42
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_REPO="VoCuc/mta-dskd"
AMID_REPO="VoCuc/amid-mta"
RESIDUAL_REPO="phuocsang/residual-mta"
DOWNLOAD_ROOT="${BASE_PATH}/checkpoints"
OUTPUT_ROOT="${BASE_PATH}/eval_outputs"
BATCH_SIZE="${BATCH_SIZE:-64}"
# Space-separated GPU list. Example:
#   DEVICE_LIST="cuda:0 cuda:1" bash run.sh
DEVICE_LIST="${DEVICE_LIST:-cuda:1}"
read -r -a DEVICES <<< "${DEVICE_LIST}"
DWA_CKPTS=(
    "dwa_mta/gpt2/gpt2-medium/span_dwa_kd/criterion=dwa_kd__skewed_reverse_kl-bf16__teacher=Qwen1.5_1.8B_SFT_Dolly__kd^rate=0.5__kd^temp=2.0__epoch=10__bsz=16x2x1=32__lr=0.0005__proj^lr=0.001/epoch10_step3570"
    "dwa_mta/gpt2/gpt2-xl/span_dwa_kd/criterion=dwa_kd__skewed_reverse_kl-lora-rank=256-alpha=8-dropout=0.1-bf16__teacher=Qwen2.5-7B-Instruct-Dolly-SFT__kd^rate=0.5__kd^temp=2.0__epoch=10__bsz=16x2x1=32__lr=0.001__proj^lr=0.001/epoch10_step3570"
    "dwa_mta/gpt2/gpt2-base/dwa_kd_word_level/criterion=dwa_kd__skewed_reverse_kl-bf16__teacher=Qwen1.5_1.8B_SFT_Dolly__kd^rate=0.5__kd^temp=2.0__epoch=10__bsz=16x2x1=32__lr=0.0005__proj^lr=0.001/epoch10_step3570"
)
RESIDUAL_CKPTS=(
    "results/gpt2/train/spanresidual_paper_A_0.1B_qwen1.8B/14290"
    "results/gpt2/train/spanresidual_mta_A_0.1B_qwen1.8B_5e-5/11432"
    "results/gpt2/train/spanresidual_paper_B_0.35B_qwen1.8B/14290"
    "results/gpt2/train/spanresidual_mta_B_0.35B_qwen1.8B/10003"
    "results/gpt2/train/spanresidual_paper_E_1.5B_qwen2.5-7B/14290"
    "results/gpt2/train/spanresidual_mta_E_1.5B_qwen2.5-7B/11432"
    "results/llama/train/spanresidual_paper_tinyllama-1.1B_mistral-7B/14290"
    "results/llama/train/spanresidual_mta_tinyllama-1.1B_mistral-7B_5e-4/8574"
    "results/opt/train/spanresidual_paper_opt-2.7B_qwen2.5-7B/14290"
    "results/opt/train/spanresidual_mta_opt-2.7B_qwen2.5-7B_5e-4/14290"
)
CHECKPOINT_ROOTS=(
    # "${DOWNLOAD_ROOT}/rerun"
    # "${DOWNLOAD_ROOT}/dwa_mta"
    "${DOWNLOAD_ROOT}/rerun/tinyllama"
    "${DOWNLOAD_ROOT}/amid_mta"
    "${DOWNLOAD_ROOT}/residual_mta/results"

)

if ! command -v python >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        # Make the script usable from a fresh shell where conda is not active.
        set +u
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate mta
        set -u
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

    echo "======================================================"
    echo " Download AMID checkpoints"
    echo " Repo       : ${AMID_REPO}"
    echo " Local dir  : ${DOWNLOAD_ROOT}/amid_mta"
    echo "======================================================"

    hf download "${AMID_REPO}" \
        --local-dir "${DOWNLOAD_ROOT}/amid_mta" \
        --max-workers 4

    echo "======================================================"
    echo " Download residual MTA checkpoints"
    echo " Repo       : ${RESIDUAL_REPO}"
    echo " Local dir  : ${DOWNLOAD_ROOT}/residual_mta"
    echo "======================================================"

    local residual_include_args=()
    for ckpt_path in "${RESIDUAL_CKPTS[@]}"; do
        residual_include_args+=("--include" "${ckpt_path}/**")
    done

    hf download "${RESIDUAL_REPO}" "${residual_include_args[@]}" \
        --repo-type dataset \
        --local-dir "${DOWNLOAD_ROOT}/residual_mta" \
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
            *gpt2-xl*|*1.5B*)
                base_model="openai-community/gpt2-xl"
                ;;
            *gpt2-medium*|*0.35B*|*0.4B*)
                base_model="openai-community/gpt2-medium"
                ;;
            *gpt2-base*|*gpt2*)
                base_model="openai-community/gpt2"
                ;;
            *opt-2.7B*|*opt-2.7b*)
                base_model="facebook/opt-2.7b"
                ;;
            *opt-1.3B*|*opt-1.3b*)
                base_model="facebook/opt-1.3b"
                ;;
            *tinyllama*|*TinyLlama*)
                if [ -d "${BASE_PATH}/model_hub/tinyllama/tinyllama-1.1b-3T" ]; then
                    base_model="${BASE_PATH}/model_hub/tinyllama/tinyllama-1.1b-3T"
                elif [ -d "${BASE_PATH}/model_hub/tinyllama/tinyllama-1.1B" ]; then
                    base_model="${BASE_PATH}/model_hub/tinyllama/tinyllama-1.1B"
                else
                    base_model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
                fi
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
            epoch_match = re.match(r"epoch(9|10)(?:_|$)", ckpt_dir.name)
            step_match = re.fullmatch(r"\d+", ckpt_dir.name)
            if not epoch_match and not step_match:
                continue

            # Priority: epoch10 > epoch9 > numeric step directories like 3570.
            if epoch_match:
                priority = int(epoch_match.group(1)) * 1_000_000
            else:
                priority = int(step_match.group(0))

            exp_dir = ckpt_dir.parent
            rel = ckpt_dir.relative_to(download_root)
            prev = best.get(exp_dir)
            if prev is None or priority > prev[0]:
                best[exp_dir] = (priority, rel, ckpt_dir)

for _, _, ckpt_dir in sorted(best.values(), key=lambda item: str(item[1])):
    print(ckpt_dir)
PY
)

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "Không tìm thấy checkpoint phù hợp trong các CHECKPOINT_ROOTS."
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
echo " Hoàn thành đánh giá checkpoint."
echo "======================================================"
