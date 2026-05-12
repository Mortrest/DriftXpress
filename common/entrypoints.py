from __future__ import annotations

import sys


STANDARD_DRIFTING_ALIASES = {"standard", "standard_drifting", "standard-drifting", "normal"}
DRIFTXPRESS_ALIASES = {"driftxpress", "xpress", "nystrom", "nyström"}


def normalize_method_name(method_name: str) -> str:
    normalized = str(method_name).strip().lower()
    if normalized in STANDARD_DRIFTING_ALIASES:
        return "standard_drifting"
    if normalized in DRIFTXPRESS_ALIASES:
        return "driftxpress"
    raise ValueError(f"Unknown method: {method_name}")


def has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv[1:])


def inject_flag(argv: list[str], flag: str, value: str | None = None) -> list[str]:
    if has_flag(argv, flag):
        return argv
    if value is None:
        return [argv[0], flag, *argv[1:]]
    return [argv[0], flag, value, *argv[1:]]


def require_equal_dataset(argv: list[str], dataset_name: str) -> list[str]:
    for idx, arg in enumerate(argv[1:], start=1):
        if arg == "--dataset":
            if idx + 1 >= len(argv):
                raise SystemExit("--dataset requires a value.")
            value = argv[idx + 1].strip().lower()
            if value != dataset_name:
                raise SystemExit(f"{argv[0]} only supports --dataset {dataset_name}; got {value!r}.")
            return argv
        if arg.startswith("--dataset="):
            value = arg.split("=", 1)[1].strip().lower()
            if value != dataset_name:
                raise SystemExit(f"{argv[0]} only supports --dataset {dataset_name}; got {value!r}.")
            return argv
    return inject_flag(argv, "--dataset", dataset_name)


def run_module(module_name: str, argv: list[str]) -> None:
    import runpy

    sys.argv = argv
    runpy.run_module(module_name, run_name="__main__")
