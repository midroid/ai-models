"""Toy rows for the metric definitions. No checkpoint."""

from laya_jp.metrics import auroc, balanced_accuracy, floor_and_renormalise, nll, score_noul
import numpy as np


def test_floor_keeps_a_hard_zero_finite():
    floored = floor_and_renormalise(np.array([[0.0, 1.0]]))
    assert floored[0, 0] > 0
    assert abs(floored.sum() - 1) < 1e-9
    value = nll(np.array([[0.0, 1.0]]), np.array([0]))
    assert value < 20


def test_auroc_is_half_for_tied_scores():
    assert abs(auroc([0.2, 0.2, 0.2, 0.2], [0, 1, 0, 1]) - 0.5) < 1e-9


def test_balanced_accuracy_catches_always_false():
    gold = [True, True, False, False]
    pred = [False, False, False, False]
    assert balanced_accuracy(pred, gold) == 0.5
    rows = [
        {"p_true": 0.1, "gold": True},
        {"p_true": 0.2, "gold": True},
        {"p_true": 0.1, "gold": False},
        {"p_true": 0.2, "gold": False},
    ]
    scored = score_noul(rows)
    assert scored["accuracy"] == 0.5
    assert scored["true_accuracy"] == 0.0
    assert scored["false_accuracy"] == 1.0
