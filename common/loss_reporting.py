from __future__ import annotations


def count_feature_slots(feature_groups) -> int:
    total_slots = 0
    for features, _ in feature_groups:
        if features.ndim < 3:
            raise ValueError(
                "Expected feature groups with shape [batch, slots, channels] for loss logging."
            )
        total_slots += int(features.shape[1])
    if total_slots <= 0:
        raise ValueError("Feature slot count must be positive.")
    return total_slots


def scaled_loss_for_logging(loss, feature_groups) -> tuple[float, int]:
    slot_count = count_feature_slots(feature_groups)
    return float(loss.detach().item()) / float(slot_count), slot_count


def format_scientific(value: float) -> str:
    return f"{float(value):.6e}"
