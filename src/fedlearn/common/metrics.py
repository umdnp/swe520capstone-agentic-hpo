from __future__ import annotations

import logging
from typing import Any

import numpy as np
from flwr.common import MetricRecord
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from fedlearn.common.model import CLASSES

logger = logging.getLogger(__name__)


def compute_binary_metrics(model, X, y) -> dict[str, float]:
    """
    Compute binary model metrics (accuracy, log loss, ROC-AUC) with failure flags.
    """
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))

    labels = getattr(model, "classes_", None)
    if labels is None and hasattr(model, "named_steps"):
        clf = model.named_steps.get("classifier")
        labels = getattr(clf, "classes_", None)

    if labels is None:
        labels = CLASSES

    log_loss_failed = 0.0
    try:
        y_proba = model.predict_proba(X)
        loss = float(log_loss(y, y_proba, labels=labels))
    except (ValueError, AttributeError, NotFittedError):
        loss = float("nan")
        log_loss_failed = 1.0

    roc_auc, roc_auc_failed = compute_roc_auc(y, model, X)

    return {
        "accuracy": acc,
        "loss": loss,
        "roc_auc": roc_auc,
        "log-loss-failed": log_loss_failed,
        "roc-auc-failed": roc_auc_failed,
    }


def compute_roc_auc(y_true, model, X) -> tuple[float, float]:
    """
    Compute ROC-AUC safely for binary classification.

    Returns:
        (roc_auc, failed_flag). If the score cannot be computed, a fallback value of (0.5, 1.0) is returned.
    """
    # AUC requires both classes
    if len(np.unique(y_true)) < 2:
        return 0.5, 1.0

    # try probability scores first
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X)[:, 1]
            return float(roc_auc_score(y_true, y_score)), 0.0
        except ValueError:
            pass

    # fallback to decision scores
    if hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X)
            return float(roc_auc_score(y_true, y_score)), 0.0
        except ValueError:
            pass

    return 0.5, 1.0


def metricrecord_to_dict(mrec: MetricRecord) -> dict[str, Any]:
    """
    Convert a Flower MetricRecord into a plain dict.
    """
    try:
        return dict(mrec)
    except (TypeError, ValueError):
        if hasattr(mrec, "to_dict"):
            return mrec.to_dict()
        if hasattr(mrec, "as_dict"):
            return mrec.as_dict()

        logger.warning("Unable to convert MetricRecord to dict; storing raw representation.")
        return {"_raw": repr(mrec)}
