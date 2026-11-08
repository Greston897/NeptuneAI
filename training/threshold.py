"""Dependency-free conservative threshold selection and binary metrics."""

from __future__ import annotations

from collections.abc import Sequence


def binary_metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float]:
    if len(predictions) != len(labels) or len(labels) == 0:
        raise ValueError("predictions and labels must be non-empty and equally sized")
    tp = sum(bool(p) and label == 1 for p, label in zip(predictions, labels))
    fp = sum(bool(p) and label == 0 for p, label in zip(predictions, labels))
    fn = sum(not bool(p) and label == 1 for p, label in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def select_threshold(
    probabilities: Sequence[float], labels: Sequence[int], target_precision: float
) -> tuple[float, dict[str, float]]:
    """Maximize recall among thresholds that meet target precision; default to .99."""
    if len(probabilities) != len(labels) or len(probabilities) == 0:
        raise ValueError("probabilities and labels must be non-empty and equally sized")
    best: tuple[float, float] | None = None  # recall, threshold
    for index in range(100):
        threshold = 0.5 + index * (0.49 / 99)
        predicted = [probability >= threshold for probability in probabilities]
        tp = sum(p and label == 1 for p, label in zip(predicted, labels))
        fp = sum(p and label == 0 for p, label in zip(predicted, labels))
        fn = sum((not p) and label == 1 for p, label in zip(predicted, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision >= target_precision and tp + fp and (best is None or recall > best[0]):
            best = (recall, threshold)

    threshold = best[1] if best else 0.99
    predicted = [probability >= threshold for probability in probabilities]
    tp = sum(p and label == 1 for p, label in zip(predicted, labels))
    fp = sum(p and label == 0 for p, label in zip(predicted, labels))
    fn = sum((not p) and label == 1 for p, label in zip(predicted, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return threshold, {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_precision": float(target_precision),
    }
