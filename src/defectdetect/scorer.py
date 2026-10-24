"""
defectdetect.scorer
===================
Produces image-level anomaly scores and pixel-level anomaly maps from patch
scores, with GPU-accelerated upsampling for video-rate inference.

Given per-patch anomaly scores (from k-NN distance to the memory bank):
  1. Reshape flat scores to spatial feature map (H_f × W_f).
  2. Bilinear upsample to original image resolution — GPU via torch.nn.functional.
  3. Gaussian smoothing to reduce high-frequency noise.
  4. Return image-level score (max of map) and full anomaly map.

GPU path
--------
When a CUDA device is available, upsampling uses torch.nn.functional.interpolate
on GPU tensors — typically 5–10× faster than cv2.resize for batch processing.
The CPU fallback is always available for environments without a GPU.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """
    Convert raw k-NN patch distances into image scores and localisation maps.

    Parameters
    ----------
    image_size : (height, width)
        Target resolution for the upsampled anomaly map.
    gaussian_sigma : float
        Smoothing applied after upsampling. 4.0 is a reasonable default.
    device : str or None
        'cuda' for GPU upsampling, 'cpu' for CPU, None = auto.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        gaussian_sigma: float = 4.0,
        device: str | None = None,
    ) -> None:
        self.image_size     = image_size   # (H, W)
        self.gaussian_sigma = gaussian_sigma

        if device is None:
            self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self._device = torch.device(device)

        logger.info("AnomalyScorer — image_size=%s  device=%s", image_size, self._device)

    def compute(
        self,
        patch_scores: np.ndarray,
        spatial_shape: Tuple[int, int],
        n_images: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert flat patch scores to image scores and anomaly maps.

        Parameters
        ----------
        patch_scores  : np.ndarray, shape (N_images * H_f * W_f,)
        spatial_shape : (H_f, W_f)
        n_images      : int

        Returns
        -------
        image_scores : np.ndarray, shape (N_images,)
        anomaly_maps : np.ndarray, shape (N_images, H, W)
        """
        H_f, W_f     = spatial_shape
        H_out, W_out = self.image_size

        # Reshape to (N, 1, H_f, W_f) for torch interpolation
        score_maps = patch_scores.reshape(n_images, 1, H_f, W_f).astype(np.float32)

        if self._device.type == "cuda" and torch.cuda.is_available():
            anomaly_maps = self._upsample_gpu(score_maps, H_out, W_out)
        else:
            anomaly_maps = self._upsample_cpu(score_maps, H_out, W_out)

        image_scores = anomaly_maps.max(axis=(1, 2))
        return image_scores, anomaly_maps

    def _upsample_gpu(self, score_maps: np.ndarray, H: int, W: int) -> np.ndarray:
        """GPU bilinear upsample + Gaussian smooth via torch."""
        t = torch.from_numpy(score_maps).to(self._device)       # (N,1,H_f,W_f)
        t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        t = t.squeeze(1).cpu().numpy()                           # (N,H,W)
        if self.gaussian_sigma > 0:
            for i in range(len(t)):
                t[i] = gaussian_filter(t[i], sigma=self.gaussian_sigma)
        return t

    def _upsample_cpu(self, score_maps: np.ndarray, H: int, W: int) -> np.ndarray:
        """CPU fallback: cv2.resize + scipy gaussian_filter."""
        n = len(score_maps)
        out = np.zeros((n, H, W), dtype=np.float32)
        for i, smap in enumerate(score_maps[:, 0]):
            up = cv2.resize(smap, (W, H), interpolation=cv2.INTER_LINEAR)
            if self.gaussian_sigma > 0:
                up = gaussian_filter(up, sigma=self.gaussian_sigma)
            out[i] = up
        return out

    @staticmethod
    def normalise(maps: np.ndarray) -> np.ndarray:
        """
        Normalise anomaly maps to [0, 1] across the full dataset.
        Required before thresholding or visualisation.
        """
        mn, mx = maps.min(), maps.max()
        if mx - mn < 1e-8:
            return np.zeros_like(maps)
        return (maps - mn) / (mx - mn)

    @staticmethod
    def overlay(image_bgr: np.ndarray, anomaly_map: np.ndarray,
                alpha: float = 0.5) -> np.ndarray:
        """
        Blend a normalised anomaly map (range [0,1]) as a JET heatmap onto the image.

        Parameters
        ----------
        image_bgr   : np.ndarray (H, W, 3) uint8 BGR
        anomaly_map : np.ndarray (H, W) float32 in [0, 1]
        alpha       : float — heatmap opacity

        Returns
        -------
        np.ndarray (H, W, 3) uint8
        """
        heat     = (anomaly_map * 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        h, w = image_bgr.shape[:2]
        if heat_bgr.shape[:2] != (h, w):
            heat_bgr = cv2.resize(heat_bgr, (w, h))
        return cv2.addWeighted(image_bgr, 1 - alpha, heat_bgr, alpha, 0)
