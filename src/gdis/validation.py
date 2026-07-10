from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .result import GDISResult


def safe_correlation(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(spearmanr(x, y).correlation if method == "spearman" else pearsonr(x, y)[0])


def roc_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[y_true == 1]
    negative = scores[y_true == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = pd.Series(combined).rank(method="average").values
    positive_ranks = ranks[:len(positive)]
    return float((np.sum(positive_ranks) - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def best_threshold(y_true, scores, threshold_count=1001) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    best = None
    for threshold in np.linspace(0.0, 1.0, threshold_count):
        predicted = (scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (predicted == 1))
        tn = np.sum((y_true == 0) & (predicted == 0))
        fp = np.sum((y_true == 0) & (predicted == 1))
        fn = np.sum((y_true == 1) & (predicted == 0))
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        accuracy = (tp + tn) / len(y_true)
        balanced = 0.5 * (sensitivity + specificity)
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-12)
        current = {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
        if best is None or current["balanced_accuracy"] > best["balanced_accuracy"]:
            best = current
    return best


def validate_against_reference(result: GDISResult, reference):
    reference = np.asarray(reference, dtype=float)
    if len(reference) != len(result.gdis):
        raise ValueError("Reference length must equal the number of GDIS values.")
    return {
        "pearson": safe_correlation(result.gdis, reference, "pearson"),
        "spearman": safe_correlation(result.gdis, reference, "spearman"),
    }
