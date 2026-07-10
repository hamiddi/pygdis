import numpy as np
from gdis.validation import best_threshold, roc_auc


def test_validation_metrics():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert best_threshold(y_true, scores)["balanced_accuracy"] > 0.99
    assert roc_auc(y_true, scores) > 0.99
