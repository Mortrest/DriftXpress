#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_CONFIG_PATH="${RELEASE_CONFIG_PATH:-${SCRIPT_DIR}/release_profiles.yaml}"

load_release_profile() {
  local dataset_key="$1"
  local env_prefix="$2"

  eval "$(
    python - "${RELEASE_CONFIG_PATH}" "${dataset_key}" "${env_prefix}" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
requested_key = sys.argv[2].strip().lower()
env_prefix = sys.argv[3].strip().upper()
config = yaml.safe_load(config_path.read_text())

datasets = config.get("datasets", {})
selected_key = None
for key, payload in datasets.items():
    aliases = {str(alias).strip().lower() for alias in payload.get("aliases", [])}
    aliases.add(str(key).strip().lower())
    if requested_key in aliases:
        selected_key = key
        break

if selected_key is None:
    raise SystemExit(f"Unknown release dataset profile: {requested_key}")

payload = datasets[selected_key]
common = payload.get("common", {})
standard = payload.get("standard_drifting", {})
driftxpress = payload.get("driftxpress", {})

def emit(name, value):
    if value is None:
        return
    print(f"{name}={shlex.quote(str(value))}")

emit(f"{env_prefix}_DATASET_KEY", selected_key)

for key, env_suffix in [
    ("data_root", "DATA_ROOT"),
    ("data_source", "DATA_SOURCE"),
    ("encoder", "ENCODER"),
    ("encoder_size", "ENCODER_SIZE"),
    ("pool_size", "POOL_SIZE"),
    ("more_features", "MORE_FEATURES"),
    ("temps", "TEMPS"),
    ("steps", "STEPS"),
    ("evaluate_every", "EVALUATE_EVERY"),
]:
    emit(f"{env_prefix}_{env_suffix}", common.get(key))

standard_map = {
    "batch_size": "BATCH_SIZE",
    "real_batch_size": "REAL_BATCH_SIZE",
    "real_feature_batch_size": "REAL_FEATURE_BATCH_SIZE",
    "fid_eval_weights": "FID_EVAL_WEIGHTS",
}
for key, env_suffix in standard_map.items():
    value = standard.get(key)
    emit(f"{env_prefix}_STANDARD_DRIFTING_{env_suffix}", value)
    emit(f"{env_prefix}_NORMAL_{env_suffix}", value)

driftxpress_map = {
    "batch_size": "BATCH_SIZE",
    "feature_batch_size": "FEATURE_BATCH_SIZE",
    "fid_eval_weights": "FID_EVAL_WEIGHTS",
    "class_ratio": "CLASS_RATIO",
    "landmark_strategy": "LANDMARK_STRATEGY",
    "landmarks_per_class": "LANDMARKS_PER_CLASS",
    "landmark_seed": "LANDMARK_SEED",
    "kmeans_iters": "KMEANS_ITERS",
    "ridge": "RIDGE",
    "repulsion": "REPULSION",
    "shard_by_class": "SHARD_BY_CLASS",
    "classes_per_shard": "CLASSES_PER_SHARD",
    "folded_exact_attraction": "FOLDED_EXACT_ATTRACTION",
    "restrict_training_to_selected_classes": "RESTRICT_TRAINING_TO_SELECTED_CLASSES",
}
for key, env_suffix in driftxpress_map.items():
    value = driftxpress.get(key)
    emit(f"{env_prefix}_DRIFTXPRESS_{env_suffix}", value)
    emit(f"{env_prefix}_XPRESS_{env_suffix}", value)
PY
  )"
}
