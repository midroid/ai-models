"""Probability metrics for choice, score, and noul decisions.

NLL applies the sokudan floor of 5e-5 and renormalises. Accuracy, AUROC, and
slot counts use the returned probabilities without that floor.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from laya_jp.pins import ECE_BINS, PROB_FLOOR


def _as_float(value) -> float:
    return float(value)


def floor_and_renormalise(probs: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    out = np.clip(np.asarray(probs, dtype=np.float64), floor, None)
    totals = out.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("probability row sums to 0")
    return out / totals


def accuracy(pred: list, gold: list) -> float:
    if not pred:
        raise ValueError("empty predictions")
    return float(sum(p == g for p, g in zip(pred, gold)) / len(pred))


def balanced_accuracy(pred: list, gold: list) -> float:
    labels = sorted(set(gold))
    recalls = []
    for label in labels:
        mask = [g == label for g in gold]
        total = sum(mask)
        if total == 0:
            continue
        hit = sum(p == g for p, g, keep in zip(pred, gold, mask) if keep)
        recalls.append(hit / total)
    if not recalls:
        raise ValueError("no gold labels")
    return float(sum(recalls) / len(recalls))


def multiclass_counts(probs: list[dict], gold: list, classes: list) -> dict:
    matrix = np.zeros((len(probs), len(classes)), dtype=np.float64)
    labels = []
    for row, (dist, label) in enumerate(zip(probs, gold)):
        if label not in classes:
            raise ValueError(f"gold {label!r} is not in {classes}")
        labels.append(classes.index(label))
        for col, name in enumerate(classes):
            matrix[row, col] = _as_float(dist.get(name, 0.0))
    return {"probs": matrix, "labels": np.asarray(labels, dtype=np.int64)}


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    floored = floor_and_renormalise(probs)
    picked = floored[np.arange(len(labels)), labels]
    return float(-np.log(picked).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> float:
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0:
            in_bin = confidence <= hi
        else:
            in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        total += (count / len(confidence)) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(total)


def binary_probs(p_true: list[float]) -> np.ndarray:
    p = np.clip(np.asarray(p_true, dtype=np.float64), 0.0, 1.0)
    return np.stack([1.0 - p, p], axis=1)


def auroc(scores: list[float], labels: list[int]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    for end in range(1, len(s) + 1):
        if end == len(s) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _round(value: float, digits: int = 6):
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def score_choice(rows: list[dict], classes: list[str]) -> dict:
    probs = [row["probs"] for row in rows]
    gold = [row["gold"] for row in rows]
    pred = [row["pred"] for row in rows]
    packed = multiclass_counts(probs, gold, classes)
    matrix = packed["probs"]
    # Four-decimal API output may not sum to 1. Renormalise for Brier and ECE.
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    matrix = matrix / totals
    return {
        "n": len(rows),
        "accuracy": _round(accuracy(pred, gold)),
        "balanced_accuracy": _round(balanced_accuracy(pred, gold)),
        "ece": _round(ece(matrix, packed["labels"])),
        "brier": _round(brier(matrix, packed["labels"])),
        "nll": _round(nll(matrix, packed["labels"])),
    }


def score_ordinal(rows: list[dict], n_levels: int) -> dict:
    classes = [str(i) for i in range(n_levels)]
    probs = [row["probs"] for row in rows]
    gold_index = [int(row["gold_index"]) for row in rows]
    gold_labels = [str(i) for i in gold_index]
    pred_index = [int(row["pred_index"]) for row in rows]
    packed = multiclass_counts(probs, gold_labels, classes)
    matrix = packed["probs"]
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    matrix = matrix / totals
    expected = np.array([float(row["expected_score"]) for row in rows], dtype=np.float64)
    gold = np.asarray(gold_index, dtype=np.float64)
    slot0 = sum(i == 0 for i in pred_index)
    return {
        "n": len(rows),
        "accuracy": _round(accuracy(pred_index, gold_index)),
        "balanced_accuracy": _round(balanced_accuracy(pred_index, gold_index)),
        "mae_argmax": _round(float(np.abs(np.asarray(pred_index) - gold).mean())),
        "mae_expectation": _round(float(np.abs(expected - gold).mean())),
        "ece": _round(ece(matrix, packed["labels"])),
        "brier": _round(brier(matrix, packed["labels"])),
        "nll": _round(nll(matrix, packed["labels"])),
        "slot0_count": slot0,
        "slot0_rate": _round(slot0 / len(rows)),
    }


def score_noul(rows: list[dict], threshold: float = 0.5) -> dict:
    p_true = [float(row["p_true"]) for row in rows]
    gold = [1 if row["gold"] else 0 for row in rows]
    pred = [1 if p >= threshold else 0 for p in p_true]
    matrix = binary_probs(p_true)
    totals = matrix.sum(axis=1, keepdims=True)
    matrix = matrix / totals
    labels = np.asarray(gold, dtype=np.int64)
    true_rows = [(p, g) for p, g in zip(pred, gold) if g == 1]
    false_rows = [(p, g) for p, g in zip(pred, gold) if g == 0]
    true_acc = sum(p == g for p, g in true_rows) / len(true_rows) if true_rows else float("nan")
    false_acc = sum(p == g for p, g in false_rows) / len(false_rows) if false_rows else float("nan")
    fp = sum(p == 1 and g == 0 for p, g in zip(pred, gold))
    fn = sum(p == 0 and g == 1 for p, g in zip(pred, gold))
    return {
        "n": len(rows),
        "accuracy": _round(accuracy(pred, gold)),
        "balanced_accuracy": _round(balanced_accuracy(pred, gold)),
        "true_accuracy": _round(true_acc),
        "false_accuracy": _round(false_acc),
        "false_positive_rate": _round(fp / len(false_rows) if false_rows else float("nan")),
        "false_negative_rate": _round(fn / len(true_rows) if true_rows else float("nan")),
        "auroc": _round(auroc(p_true, gold)),
        "ece": _round(ece(matrix, labels)),
        "brier": _round(brier(matrix, labels)),
        "nll": _round(nll(matrix, labels)),
    }


def group_rows(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["qtype"]].append(record)
    return grouped
