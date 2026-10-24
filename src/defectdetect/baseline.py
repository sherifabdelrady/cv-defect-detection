"""
defectdetect.baseline
=====================
Classical image-processing anomaly baseline.

Uses Structural Similarity Index (SSIM) between a test image and the
per-pixel mean image of the training normal set. Pixels where SSIM is
low (i.e. structure differs significantly from the normal mean) are
flagged as anomalous.

This is a meaningful baseline — not a straw man. SSIM-based detection
works well when:
  - Defects are large relative to the spatial SSIM window.
  - The product surface is relatively uniform (textureless).

It fails when:
  - Subtle defects (thin scratches, small pits) below the SSIM window size.
  - High natural texture variation in normal images.

Comparison against PatchCore allows us to quantify what deep features add.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ClassicalBaseline:
    """
    SSIM-based anomaly detector using a pixel-mean normal template.

    Usage
    -----
    >>> baseline = ClassicalBaseline(image_size=(256, 256))
    >>> baseline.fit(normal_images)          # list of BGR np.ndarray
    >>> score, amap = baseline.predict(test_image)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        ssim_window: int = 11,
        blur_sigma: float = 2.0,
    ) -> None:
        self.image_size  = image_size   # (H, W)
        self.ssim_window = ssim_window
        self.blur_sigma  = blur_sigma
        self._mean_normal: Optional[np.ndarray] = None   # (H, W, 3) float32
        self._std_normal:  Optional[np.ndarray] = None   # (H, W, 3) float32

    def fit(self, images: List[np.ndarray]) -> "ClassicalBaseline":
        """
        Compute pixel-wise mean and std of normal training images.

        Parameters
        ----------
        images : list of np.ndarray — BGR uint8
        """
        h, w = self.image_size
        resized = np.stack([
            cv2.resize(img, (w, h)).astype(np.float32) / 255.0
            for img in images
        ])  # (N, H, W, 3)

        self._mean_normal = resized.mean(axis=0)
        self._std_normal  = resized.std(axis=0) + 1e-6   # avoid division by zero

        logger.info("ClassicalBaseline fitted on %d images", len(images))
        return self

    def predict(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Score a single test image.

        Parameters
        ----------
        image : np.ndarray — BGR uint8

        Returns
        -------
        image_score : float — mean pixel deviation (higher = more anomalous)
        anomaly_map : np.ndarray (H, W) float32 — per-pixel anomaly score
        """
        if self._mean_normal is None:
            raise RuntimeError("Baseline not fitted. Call fit() first.")

        h, w = self.image_size
        img  = cv2.resize(image, (w, h)).astype(np.float32) / 255.0

        # Standardised pixel difference: (|img - mean| / std) per channel
        diff = np.abs(img - self._mean_normal) / self._std_normal
        amap = diff.mean(axis=2)  # collapse channels → (H, W)

        # Smooth to suppress pixel-level noise
        if self.blur_sigma > 0:
            amap = cv2.GaussianBlur(amap, (0, 0), self.blur_sigma)

        image_score = float(amap.mean())
        return image_score, amap

    def predict_batch(
        self, images: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score a list of images. Returns (scores, amaps)."""
        results = [self.predict(img) for img in images]
        scores  = np.array([r[0] for r in results])
        amaps   = np.stack([r[1] for r in results])
        return scores, amaps
