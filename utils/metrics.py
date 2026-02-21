"""
Evaluation Metrics for ABC-CapsNet.

Implements:
    - Equal Error Rate (EER)
    - Accuracy
    - Precision, Recall, F1-Score
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER).

    EER is the point where False Acceptance Rate (FAR) equals
    False Rejection Rate (FRR) on the ROC curve.

    Args:
        labels: Ground truth binary labels (0 = real, 1 = fake).
        scores: Prediction scores (higher = more likely fake).

    Returns:
        eer: Equal Error Rate as a percentage.
        threshold: The threshold at which EER occurs.
    """
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find the point where FPR = FNR
    try:
        eer = brentq(lambda x: interp1d(fpr, fpr)(x) - interp1d(fpr, fnr)(x), 0.0, 1.0)
        threshold_idx = np.nanargmin(np.abs(fpr - eer))
        threshold = thresholds[threshold_idx]
    except ValueError:
        # Fallback: find closest point
        abs_diff = np.abs(fpr - fnr)
        min_idx = np.nanargmin(abs_diff)
        eer = (fpr[min_idx] + fnr[min_idx]) / 2
        threshold = thresholds[min_idx]

    return eer * 100, threshold  # Return as percentage


def compute_accuracy(labels, predictions):
    """
    Compute classification accuracy.

    Args:
        labels: Ground truth labels.
        predictions: Predicted labels.

    Returns:
        accuracy: Accuracy as a percentage.
    """
    return accuracy_score(labels, predictions) * 100


def compute_metrics(labels, predictions, scores=None):
    """
    Compute comprehensive evaluation metrics.

    Args:
        labels: Ground truth labels.
        predictions: Predicted labels.
        scores: Prediction scores for EER computation (optional).

    Returns:
        dict with accuracy, precision, recall, f1, and optionally EER.
    """
    metrics = {
        "accuracy": compute_accuracy(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0) * 100,
        "recall": recall_score(labels, predictions, zero_division=0) * 100,
        "f1": f1_score(labels, predictions, zero_division=0) * 100,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }

    if scores is not None:
        eer, threshold = compute_eer(labels, scores)
        metrics["eer"] = eer
        metrics["eer_threshold"] = threshold

    return metrics
