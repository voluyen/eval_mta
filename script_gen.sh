#!/bin/bash
# Downloads 4 checkpoints from HuggingFace to ckpt_gen/, then generates
# dolly eval responses for each checkpoint.
#
# Output format (one JSON per line in gen_outputs/<name>/dolly_gen.jsonl):
#   {"prompt": "Below is an instruction...\n\n### Response:\n", "generated_text": "<model response>"}
#
# Env vars (override defaults):
#   DEVICE_LIST="cuda:0 cuda:1"   -- space-separated GPUs; ckpts are round-robin assigned
#   BATCH_SIZE=16
#   SKIP_DOWNLOAD=1               -- skip hf download (assume already done)

set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_ROOT="${BASE_PATH}/ckpt_gen"
OUTPUT_ROOT="${BASE_PATH}/gen_outputs"
DOLLY_EVAL="${BASE_PATH}/data/dolly/valid.jsonl"
BATCH_SIZE="${BATCH_SIZE:-16}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
DEVICE_LIST="${DEVICE_LIST:-cuda:0}"
read -r -a DEVICES <<< "${DEVICE_LIST}"

# Base model for LoRA checkpoints (all ckpts here are gpt2-base variants)
BASE_MODEL="openai-community/gpt2"

# ---------------------------------------------------------------------------
# Checkpoint registry: NAME -> (HF_REPO, HF_PATH_IN_REPO)
# ---------------------------------------------------------------------------
CKPT_NAMES=("csd" "spancsd" "amid" "span_dskdv2")

declare -A CKPT_REPO=(
    ["csd"]="baesad/mta_checkpoints"
    ["spancsd"]="baesad/mta_checkpoints"
    ["amid"]="VoCuc/amid-mta-ckpt"
    ["span_dskdv2"]="VoCuc/mta-dskd"
)

declare -A CKPT_HF_PATH=(
    ["csd"]="csd/gpt2-base/csd/2856"
    ["spancsd"]="csd/gpt2-base/spancsd/2856"
    ["amid"]="turn2/gpt2-base#amid/ab_pr_0.5_0.5_16_1e-4/3570"
    ["span_dskdv2"]="rerun/gpt2/gpt2-base/mta_dskd_v2_eta_full/skewed_reverse_kl-bf16__teacher_qwen1.5__kd^rate0.5__kd^temp2.0__epoch10__bsz16x2x1x1_32__lr0.0005__proj^lr0.001/epoch10_step3570_loss5.8625_rougel24.8647"
)

# ---------------------------------------------------------------------------

if ! command -v python >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        set +u
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate mta
        set -u
    fi
fi

check_hf_cli() {
    if ! command -v hf >/dev/null 2>&1; then
        echo "[ERROR] hf CLI not found. Install with: pip install 'huggingface_hub[cli]'"
        exit 1
    fi
}

download_ckpt() {
    local name="$1"
    local repo="${CKPT_REPO[$name]}"
    local hf_path="${CKPT_HF_PATH[$name]}"
    local local_dir="${CKPT_ROOT}/${name}"
    local ckpt_dir="${local_dir}/${hf_path}"

    if [ -f "${ckpt_dir}/config.json" ] || [ -f "${ckpt_dir}/adapter_config.json" ]; then
        echo "  [${name}] Already downloaded → ${ckpt_dir}"
        return
    fi

    echo "  [${name}] Downloading ${repo}  path: ${hf_path}"
    hf download "${repo}" \
        --include "${hf_path}/**" \
        --local-dir "${local_dir}" \
        --max-workers 4
    echo "  [${name}] Download done."
}

gen_ckpt() {
    local name="$1"
    local device="$2"
    local hf_path="${CKPT_HF_PATH[$name]}"
    local ckpt_dir="${CKPT_ROOT}/${name}/${hf_path}"
    local output_file="${OUTPUT_ROOT}/${name}/dolly_gen.jsonl"

    if [ -f "${output_file}" ]; then
        echo "  [${device} | ${name}] Already generated → ${output_file}"
        return
    fi

    if [ ! -f "${ckpt_dir}/config.json" ] && [ ! -f "${ckpt_dir}/adapter_config.json" ]; then
        echo "  [${device} | ${name}] ERROR: checkpoint not found at ${ckpt_dir}"
        return 1
    fi

    local py_args=""
    py_args+=" --data_path ${DOLLY_EVAL}"
    py_args+=" --output_file ${output_file}"
    py_args+=" --device ${device}"
    py_args+=" --batch_size ${BATCH_SIZE}"
    py_args+=" --tokenizer ${ckpt_dir}"
    py_args+=" --temperature 1.0"
    py_args+=" --top_p 1.0"

    if [ -f "${ckpt_dir}/adapter_config.json" ]; then
        echo "  [${device} | ${name}] LoRA checkpoint → base: ${BASE_MODEL}"
        py_args+=" --model_path ${BASE_MODEL}"
        py_args+=" --lora_path ${ckpt_dir}"
    else
        py_args+=" --model_path ${ckpt_dir}"
    fi

    echo "  [${device} | ${name}] Generating → ${output_file}"
    # shellcheck disable=SC2086
    python "${BASE_PATH}/src/run_gen.py" ${py_args}
    echo "  [${device} | ${name}] Done."
}

worker() {
    local device="$1"
    shift
    local failed=0
    for name in "$@"; do
        if ! gen_ckpt "${name}" "${device}"; then
            failed=1
        fi
    done
    return "${failed}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "======================================================"
echo " script_gen.sh"
echo " CKPT root : ${CKPT_ROOT}"
echo " Output    : ${OUTPUT_ROOT}"
echo " Data      : ${DOLLY_EVAL}"
echo " GPUs      : ${DEVICES[*]}"
echo " Batch     : ${BATCH_SIZE}"
echo "======================================================"

check_hf_cli

if [ "${SKIP_DOWNLOAD}" -ne 1 ]; then
    echo ""
    echo "--- Download checkpoints ---"
    for name in "${CKPT_NAMES[@]}"; do
        download_ckpt "${name}"
    done
fi

echo ""
echo "--- Generate dolly responses ---"

NGPU=${#DEVICES[@]}
PIDS=()
for ((g=0; g<NGPU; g++)); do
    SUBSET=()
    for ((i=g; i<${#CKPT_NAMES[@]}; i+=NGPU)); do
        SUBSET+=("${CKPT_NAMES[$i]}")
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

if [ "${FAILED}" -ne 0 ]; then
    echo "[ERROR] One or more generation jobs failed."
    exit 1
fi

echo ""
echo "======================================================"
echo " All done. Outputs:"
for name in "${CKPT_NAMES[@]}"; do
    echo "   ${OUTPUT_ROOT}/${name}/dolly_gen.jsonl"
done
echo "======================================================"
