#! /bin/bash
# Create .venv and install all eval dependencies.
# Run once: bash setup_env.sh
# After this, run_hf_models.sh will use .venv automatically.

set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${BASE_PATH}/.venv"

echo "======================================================"
echo " Creating virtual environment"
echo " Path: ${VENV_DIR}"
echo "======================================================"

python3 -m venv "${VENV_DIR}"

echo "======================================================"
echo " Installing dependencies"
echo " torch >= 2.6.0  (CUDA 12.4)"
echo "======================================================"

"${VENV_DIR}/bin/pip" install --upgrade pip

# Install torch with CUDA 12.4 support (for H200)
"${VENV_DIR}/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements
"${VENV_DIR}/bin/pip" install -r "${BASE_PATH}/requirements.txt"

echo ""
echo "======================================================"
echo " Done. Python: $("${VENV_DIR}/bin/python" --version)"
echo " Torch:  $("${VENV_DIR}/bin/python" -c 'import torch; print(torch.__version__)')"
echo "======================================================"
