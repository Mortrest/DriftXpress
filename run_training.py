from __future__ import annotations

import argparse
import sys

from common.entrypoints import has_flag, normalize_method_name, run_module
from common.release_profiles import dataset_cli_name, get_release_profile


def _inject_value(argv: list[str], flag: str, value: str | int | float | None) -> list[str]:
    if value is None or has_flag(argv, flag):
        return argv
    return [argv[0], flag, str(value), *argv[1:]]


def _inject_bool(argv: list[str], flag: str, enabled: bool, negative_flag: str | None = None) -> list[str]:
    if has_flag(argv, flag) or (negative_flag and has_flag(argv, negative_flag)):
        return argv
    chosen = flag if enabled else negative_flag
    if chosen is None:
        return argv
    return [argv[0], chosen, *argv[1:]]


def _apply_common_defaults(argv: list[str], profile: dict) -> list[str]:
    common = profile.get("common", {})
    for key, flag in [
        ("data_root", "--data-root"),
        ("data_source", "--data-source"),
        ("encoder", "--encoder"),
        ("encoder_size", "--encoder-size"),
        ("pool_size", "--pool-size"),
        ("temps", "--temps"),
        ("steps", "--steps"),
        ("evaluate_every", "--evaluate-every"),
    ]:
        argv = _inject_value(argv, flag, common.get(key))
    argv = _inject_bool(argv, "--more-features", bool(common.get("more_features", 0)))
    return argv


def _apply_standard_drifting_defaults(argv: list[str], profile: dict) -> list[str]:
    defaults = profile.get("standard_drifting", {})
    for key, flag in [
        ("batch_size", "--batch-size"),
        ("real_batch_size", "--real-batch-size"),
        ("real_feature_batch_size", "--real-feature-batch-size"),
        ("fid_eval_weights", "--fid-eval-weights"),
    ]:
        argv = _inject_value(argv, flag, defaults.get(key))
    return argv


def _apply_driftxpress_defaults(argv: list[str], profile: dict) -> list[str]:
    defaults = profile.get("driftxpress", {})
    for key, flag in [
        ("batch_size", "--batch-size"),
        ("feature_batch_size", "--feature-batch-size"),
        ("fid_eval_weights", "--fid-eval-weights"),
        ("class_ratio", "--imagenet32-class-ratio"),
        ("landmark_strategy", "--nystrom-landmark-strategy"),
        ("landmarks_per_class", "--nystrom-landmarks-per-class"),
        ("landmark_seed", "--nystrom-landmark-seed"),
        ("kmeans_iters", "--nystrom-kmeans-iters"),
        ("ridge", "--nystrom-ridge"),
        ("repulsion", "--nystrom-repulsion"),
        ("classes_per_shard", "--nystrom-classes-per-shard"),
    ]:
        argv = _inject_value(argv, flag, defaults.get(key))
    argv = _inject_bool(
        argv,
        "--nystrom-shard-by-class",
        bool(defaults.get("shard_by_class", 0)),
        "--no-nystrom-shard-by-class",
    )
    argv = _inject_bool(
        argv,
        "--nystrom-folded-exact-attraction",
        bool(defaults.get("folded_exact_attraction", 0)),
        "--no-nystrom-folded-exact-attraction",
    )
    argv = _inject_bool(
        argv,
        "--restrict-training-to-selected-classes",
        bool(defaults.get("restrict_training_to_selected_classes", 0)),
        "--no-restrict-training-to-selected-classes",
    )
    return argv


def _resolve_backend(dataset_name: str, method_name: str) -> str:
    if method_name == "standard_drifting":
        return "trainers.standard_drifting"
    return "train_nystrom"


def build_delegate_argv(argv: list[str], dataset_name: str, method_name: str) -> list[str]:
    profile = get_release_profile(dataset_name)
    delegate = argv[:]
    delegate = _inject_value(delegate, "--dataset", dataset_cli_name(dataset_name))
    delegate = _apply_common_defaults(delegate, profile)
    if method_name == "standard_drifting":
        delegate = _apply_standard_drifting_defaults(delegate, profile)
    else:
        delegate = _apply_driftxpress_defaults(delegate, profile)
    return delegate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    known, remaining = parser.parse_known_args()

    method_name = normalize_method_name(known.method)
    backend = _resolve_backend(known.dataset, method_name)
    delegate_argv = build_delegate_argv([sys.argv[0], *remaining], known.dataset, method_name)
    run_module(backend, delegate_argv)


if __name__ == "__main__":
    main()
