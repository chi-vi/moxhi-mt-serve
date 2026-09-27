#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 <cpu|cuda|rocm> [server options]" >&2
  echo "Example: $0 cpu --host 0.0.0.0 --port 8000" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
mode=$1
shift

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
venv_dir=${VENV_DIR:-$project_dir/.venv}
venv_python=$venv_dir/bin/python

if [[ ! -x "$venv_python" ]]; then
  echo "Khong tim thay virtualenv $venv_dir; hay chay scripts/install.sh $mode" >&2
  exit 1
fi

case "$mode" in
  cpu) module=moxhimt_runner.server_cpu ;;
  cuda) module=moxhimt_runner.server_cuda ;;
  rocm) module=moxhimt_runner.server_rocm ;;
  *) usage ;;
esac

exec "$venv_python" -m "$module" "$@"
