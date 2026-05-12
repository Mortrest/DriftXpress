#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${RELEASE_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_release.sh <dataset> <method> [imagenet_ratio] [extra args...]

Datasets:
  cifar10 | cifar100 | svhn | imagenet | imagenet32

Methods:
  standard-drifting | standard | normal
  driftxpress | xpress | nystrom

Examples:
  ./release/scripts/run_release.sh cifar10 standard-drifting
  ./release/scripts/run_release.sh cifar100 driftxpress
  ./release/scripts/run_release.sh imagenet standard-drifting 10
  ./release/scripts/run_release.sh imagenet driftxpress 10 --deadline-epoch 9999999999
EOF
}

has_flag() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" || "${arg}" == "${needle}="* ]]; then
      return 0
    fi
  done
  return 1
}

normalize_dataset() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    cifar10) printf '%s\n' "cifar10" ;;
    cifar100) printf '%s\n' "cifar100" ;;
    svhn) printf '%s\n' "svhn" ;;
    imagenet|imagenet32) printf '%s\n' "imagenet" ;;
    *)
      printf 'Unknown dataset: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

normalize_method() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    standard-drifting|standard_drifting|standard|normal) printf '%s\n' "standard-drifting" ;;
    driftxpress|xpress|nystrom|nyström) printf '%s\n' "driftxpress" ;;
    *)
      printf 'Unknown method: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

default_master_port() {
  case "$1:$2" in
    imagenet:standard-drifting) printf '%s\n' "29610" ;;
    imagenet:driftxpress) printf '%s\n' "29611" ;;
    cifar10:standard-drifting) printf '%s\n' "29612" ;;
    cifar10:driftxpress) printf '%s\n' "29613" ;;
    cifar100:standard-drifting) printf '%s\n' "29614" ;;
    cifar100:driftxpress) printf '%s\n' "29615" ;;
    svhn:standard-drifting) printf '%s\n' "29616" ;;
    svhn:driftxpress) printf '%s\n' "29617" ;;
    *)
      printf 'No default port for %s %s\n' "$1" "$2" >&2
      exit 1
      ;;
  esac
}

default_output_dir() {
  local dataset_key="$1"
  local method_key="$2"
  local imagenet_ratio="$3"
  local output_dataset="${dataset_key}"
  local output_method="normal"
  if [[ "${method_key}" == "driftxpress" ]]; then
    output_method="xpress"
  fi
  if [[ "${dataset_key}" == "imagenet" ]]; then
    output_dataset="imagenet32"
    local ratio_tag="${imagenet_ratio//./p}"
    printf '%s\n' "${REPO_DIR}/outputs/release/${output_dataset}/${output_method}_ratio_${ratio_tag}"
    return
  fi
  printf '%s\n' "${REPO_DIR}/outputs/release/${output_dataset}/${output_method}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

DATASET_KEY="$(normalize_dataset "$1")"
METHOD_KEY="$(normalize_method "$2")"
shift 2

IMAGENET_RATIO="${IMAGENET_TRAIN_SAMPLE_RATIO:-10}"
if [[ "${DATASET_KEY}" == "imagenet" && $# -gt 0 && "${1}" != --* ]]; then
  IMAGENET_RATIO="$1"
  shift
fi

EXTRA_ARGS=("$@")
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-$(default_master_port "${DATASET_KEY}" "${METHOD_KEY}")}"
OUTPUT_DIR="${OUTPUT_DIR:-$(default_output_dir "${DATASET_KEY}" "${METHOD_KEY}" "${IMAGENET_RATIO}")}"
MANAGED_OUTPUT_DIR="${OUTPUT_DIR}"

cmd=(
  torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  release/run_training.py
  --dataset "${DATASET_KEY}"
  --method "${METHOD_KEY}"
)

if ! has_flag --output-dir "${EXTRA_ARGS[@]}"; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
else
  MANAGED_OUTPUT_DIR=""
fi

if [[ "${DATASET_KEY}" == "imagenet" ]]; then
  if ! has_flag --data-source "${EXTRA_ARGS[@]}"; then
    cmd+=(--data-source "${DATA_SOURCE:-data/Imagenet32_train.zip}")
  fi
  if [[ "${METHOD_KEY}" == "standard-drifting" ]]; then
    if ! has_flag --train-sample-ratio "${EXTRA_ARGS[@]}"; then
      cmd+=(--train-sample-ratio "${IMAGENET_RATIO}")
    fi
  else
    IMAGENET_XPRESS_CLASS_RATIO="${IMAGENET_XPRESS_CLASS_RATIO:-${IMAGENET_RATIO}}"
    if ! has_flag --imagenet32-class-ratio "${EXTRA_ARGS[@]}"; then
      cmd+=(--imagenet32-class-ratio "${IMAGENET_XPRESS_CLASS_RATIO}")
    fi
  fi
else
  if ! has_flag --data-root "${EXTRA_ARGS[@]}"; then
    cmd+=(--data-root "${DATA_ROOT:-./data}")
  fi
fi

cmd+=("${EXTRA_ARGS[@]}")

if [[ -n "${MANAGED_OUTPUT_DIR}" ]]; then
  mkdir -p "${MANAGED_OUTPUT_DIR}"
fi
cd "${REPO_DIR}"

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

PYTHONPATH=. "${cmd[@]}"
