#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${RELEASE_DIR}/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  run_release_suite_30min.sh [imagenet_ratio]

Runs the release suite sequentially for ImageNet, CIFAR10, CIFAR100, and SVHN.
Each Standard Drifting and DriftXpress variant gets its own 30-minute deadline.
EOF
  exit 0
fi

RUN_MINUTES="${RUN_MINUTES:-30}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29610}"
IMAGENET_TRAIN_SAMPLE_RATIO="${1:-${IMAGENET_TRAIN_SAMPLE_RATIO:-10}}"
IMAGENET_XPRESS_CLASS_RATIO="${IMAGENET_XPRESS_CLASS_RATIO:-${IMAGENET_TRAIN_SAMPLE_RATIO}}"
RATIO_TAG="${IMAGENET_TRAIN_SAMPLE_RATIO//./p}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${REPO_DIR}/outputs/release/normal_xpress_suite_30min_${RUN_TAG}}"

mkdir -p "${OUTPUT_BASE_DIR}"

make_deadline_epoch() {
  python - "${RUN_MINUTES}" <<'PY'
import sys
import time

run_minutes = float(sys.argv[1])
print(time.time() + run_minutes * 60.0)
PY
}

run_variant() {
  local label="$1"
  local master_port="$2"
  local output_dir="$3"
  local dataset="$4"
  local method="$5"
  shift 5

  local deadline_epoch
  deadline_epoch="$(make_deadline_epoch)"

  echo
  echo "=== ${label} ==="
  echo "  nproc_per_node=${NPROC_PER_NODE}"
  echo "  master_port=${master_port}"
  echo "  run_minutes=${RUN_MINUTES}"
  echo "  deadline_epoch=${deadline_epoch}"

  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  MASTER_PORT="${master_port}" \
  OUTPUT_DIR="${output_dir}" \
  IMAGENET_TRAIN_SAMPLE_RATIO="${IMAGENET_TRAIN_SAMPLE_RATIO}" \
  IMAGENET_XPRESS_CLASS_RATIO="${IMAGENET_XPRESS_CLASS_RATIO}" \
  bash "${SCRIPT_DIR}/run_release.sh" "${dataset}" "${method}" "$@" --deadline-epoch "${deadline_epoch}"
}

run_variant \
  "imagenet/Standard Drifting" \
  "$((MASTER_PORT_BASE + 0))" \
  "${OUTPUT_BASE_DIR}/imagenet32/normal_ratio_${RATIO_TAG}" \
  imagenet \
  standard-drifting \
  "${IMAGENET_TRAIN_SAMPLE_RATIO}"

run_variant \
  "imagenet/DriftXpress" \
  "$((MASTER_PORT_BASE + 1))" \
  "${OUTPUT_BASE_DIR}/imagenet32/xpress_ratio_${RATIO_TAG}" \
  imagenet \
  driftxpress \
  "${IMAGENET_XPRESS_CLASS_RATIO}"

run_variant \
  "cifar10/Standard Drifting" \
  "$((MASTER_PORT_BASE + 2))" \
  "${OUTPUT_BASE_DIR}/cifar10/normal" \
  cifar10 \
  standard-drifting

run_variant \
  "cifar10/DriftXpress" \
  "$((MASTER_PORT_BASE + 3))" \
  "${OUTPUT_BASE_DIR}/cifar10/xpress" \
  cifar10 \
  driftxpress

run_variant \
  "cifar100/Standard Drifting" \
  "$((MASTER_PORT_BASE + 4))" \
  "${OUTPUT_BASE_DIR}/cifar100/normal" \
  cifar100 \
  standard-drifting

run_variant \
  "cifar100/DriftXpress" \
  "$((MASTER_PORT_BASE + 5))" \
  "${OUTPUT_BASE_DIR}/cifar100/xpress" \
  cifar100 \
  driftxpress

run_variant \
  "svhn/Standard Drifting" \
  "$((MASTER_PORT_BASE + 6))" \
  "${OUTPUT_BASE_DIR}/svhn/normal" \
  svhn \
  standard-drifting

run_variant \
  "svhn/DriftXpress" \
  "$((MASTER_PORT_BASE + 7))" \
  "${OUTPUT_BASE_DIR}/svhn/xpress" \
  svhn \
  driftxpress
