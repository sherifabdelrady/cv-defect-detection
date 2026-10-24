"""
defectdetect.evaluator
======================
Standardised evaluation for anomaly detection — image-level and pixel-level.

Metrics implemented
-------------------
Image-level:
  - ROC-AUC (primary metric for anomaly detection)
  - Average Precision (AP) — area under precision-recall curve
  - Best F1 score + optimal threshold
  - Confusion matrix at the optimal threshold

Pixel-level (when ground-truth masks available):
  - Pixel-level ROC-AUC
  - Pixel-level AP
  - IoU at the optimal threshold
  - Dice coefficient

These match the evaluation protocol in MVTec AD and PatchCore papers.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class AnomalyEvaluator:
    """
    Compute anomaly detection metrics from scores and binary labels.

    Usage
    -----
    >>> evaluator = AnomalyEvaluator()
    >>> evaluator.fit(image_scores, image_labels)  # 0=normal, 1=anomaly
    >>> results = evaluator.image_level_metrics()
    >>> evaluator.fit_pixel(pixel_scores_flat, pixel_labels_flat)
    >>> pixel_results = evaluator.pixel_level_metrics()
    """

    def __init__(self) -> None:
        self._img_scores: Optional[np.ndarray] = None
        self._img_labels: Optional[np.ndarray] = None
        self._px_scores:  Optional[np.ndarray] = None
        self._px_labels:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "AnomalyEvaluator":
        """
        Set image-level scores and binary labels.

        Parameters
        ----------
        scores : np.ndarray, shape (N,) — anomaly score per image
        labels : np.ndarray, shape (N,) — 0=normal, 1=anomaly
        """
        self._img_scores = np.asarray(scores, dtype=np.float32)
        self._img_labels = np.asarray(labels, dtype=np.int32)
        return self

    def fit_pixel(self, scores: np.ndarray, labels: np.ndarray) -> "AnomalyEvaluator":
        """
        Set flattened pixel-level scores and binary labels.

        Parameters
        ----------
        scores : np.ndarray, shape (N_px,) — anomaly score per pixel
        labels : np.ndarray, shape (N_px,) — 0=normal, 1=anomaly (from GT mask)
        """
        self._px_scores = np.asarray(scores, dtype=np.float32)
        self._px_labels = np.asarray(labels, dtype=np.int32)
        return self

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def image_level_metrics(self) -> dict:
        """
        Compute image-level ROC-AUC, AP, best F1, and optimal threshold.

        Returns
        -------
        dict with keys: roc_auc, ap, best_f1, best_threshold, confusion_matrix
        """
        if self._img_scores is None:
            raise RuntimeError("Call fit() first.")

        roc_auc = roc_auc_score(self._img_labels, self._img_scores)
        ap      = average_precision_score(self._img_labels, self._img_scores)

        # Optimal threshold: maximise F1
        prec, rec, thresholds = precision_recall_curve(
            self._img_labels, self._img_scores
        )
        f1s     = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = int(np.argmax(f1s[:-1]))  # last element has no threshold
        best_f1  = float(f1s[best_idx])
        best_thr = float(thresholds[best_idx])

        preds = (self._img_scores >= best_thr).astype(int)
        cm    = confusion_matrix(self._img_labels, preds).tolist()

        results = {
            "roc_auc":         round(roc_auc, 4),
            "average_precision": round(ap, 4),
            "best_f1":         round(best_f1, 4),
            "best_threshold":  round(best_thr, 6),
            "confusion_matrix": cm,
            "n_normal":        int((self._img_labels == 0).sum()),
            "n_anomaly":       int((self._img_labels == 1).sum()),
        }
        logger.info("Image-level — ROC-AUC=%.4f  AP=%.4f  F1=%.4f  threshold=%.6f",
                    roc_auc, ap, best_f1, best_thr)
        return results

    def pixel_level_metrics(self) -> dict:
        """
        Compute pixel-level ROC-AUC and AP.

        Returns
        -------
        dict with keys: roc_auc, ap, best_iou, dice
        """
        if self._px_scores is None:
            raise RuntimeError("Call fit_pixel() first.")

        roc_auc = roc_auc_score(self._px_labels, self._px_scores)
        ap      = average_precision_score(self._px_labels, self._px_scores)

        # Best IoU / Dice at optimal threshold
        prec, rec, thresholds = precision_recall_curve(self._px_labels, self._px_scores)
        f1s      = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = int(np.argmax(f1s[:-1]))
        best_thr = float(thresholds[best_idx])
        preds    = (self._px_scores >= best_thr).astype(int)

        tp = int(((preds == 1) & (self._px_labels == 1)).sum())
        fp = int(((preds == 1) & (self._px_labels == 0)).sum())
        fn = int(((preds == 0) & (self._px_labels == 1)).sum())

        iou  = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        results = {
            "pixel_roc_auc": round(roc_auc, 4),
            "pixel_ap":      round(ap, 4),
            "best_iou":      round(iou, 4),
            "dice":          round(dice, 4),
            "best_threshold": round(best_thr, 6),
        }
        logger.info("Pixel-level — ROC-AUC=%.4f  AP=%.4f  IoU=%.4f  Dice=%.4f",
                    roc_auc, ap, iou, dice)
        return results

    def roc_curve_data(self) -> tuple:
        """Return (fpr, tpr, thresholds) for ROC curve plotting."""
        return roc_curve(self._img_labels, self._img_scores)

    def pr_curve_data(self) -> tuple:
        """Return (precision, recall, thresholds) for PR curve plotting."""
        return precision_recall_curve(self._img_labels, self._img_scores)
