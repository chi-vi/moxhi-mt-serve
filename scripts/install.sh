#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 <cpu|cuda|rocm>" >&2
  echo "Optional: VENV_DIR=.venv PYTHON_BIN=python3 PYTORCH_INDEX_URL=... $0 <mode>" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
mode=$1

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
venv_dir=${VENV_DIR:-$project_dir/.venv}
python_bin=${PYTHON_BIN:-python3}

if [[ ! -x "$venv_dir/bin/python" ]]; then
  if ! "$python_bin" -m venv "$venv_dir"; then
    echo "Khong tao duoc virtualenv. Tren Debian/Ubuntu hay cai python3-venv." >&2
    exit 1
  fi
fi

venv_python=$venv_dir/bin/python
if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
  echo "Virtualenv $venv_dir khong co pip; hay sua hoac xoa environment nay roi thu lai." >&2
  exit 1
fi
"$venv_python" -m pip install --upgrade pip

case "$mode" in
  cpu)
    "$venv_python" -m pip install -e "$project_dir[cpu,server]"
    "$venv_python" -c 'import ctranslate2; print("CTranslate2", ctranslate2.__version__)'
    ;;
  cuda)
    pytorch_index_url=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
    "$venv_python" -m pip install torch --index-url "$pytorch_index_url"
    "$venv_python" -m pip install -e "$project_dir[server]"
    if [[ ${SKIP_DEVICE_CHECK:-0} != 1 ]]; then
      "$venv_python" - <<'PY'
import torch
assert torch.version.cuda is not None and torch.version.hip is None, "PyTorch wheel khong phai CUDA"
assert torch.cuda.is_available(), "Khong tim thay GPU NVIDIA CUDA"
print("CUDA", torch.version.cuda, torch.cuda.get_device_name(0))
PY
    fi
    ;;
  rocm)
    pytorch_index_url=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.1}
    "$venv_python" -m pip install torch --index-url "$pytorch_index_url"
    "$venv_python" -m pip install -e "$project_dir[server]"
    if [[ ${SKIP_DEVICE_CHECK:-0} != 1 ]]; then
      "$venv_python" - <<'PY'
import torch
assert torch.version.hip is not None, "PyTorch wheel khong phai ROCm/HIP"
assert torch.cuda.is_available(), "Khong tim thay GPU AMD ROCm/HIP"
print("ROCm/HIP", torch.version.hip, torch.cuda.get_device_name(0))
PY
    fi
    ;;
  *)
    usage
    ;;
esac

echo "Da cai dat mode '$mode' vao $venv_dir"
