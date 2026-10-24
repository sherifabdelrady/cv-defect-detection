"""
defectdetect.extractor
======================
Deep feature extraction from a pretrained WideResNet-50 backbone with full
CUDA/MPS/CPU device support and automatic mixed precision (AMP) inference.

PatchCore uses intermediate CNN features (layer2 + layer3) because they carry
both local texture information and higher-level structural context — unlike
final classifier features which are too abstract for anomaly localisation.

Design decisions
----------------
- WideResNet-50-2 (torchvision) pretrained on ImageNet — frozen, no fine-tuning.
- Extract from layer2 (stride 8) and layer3 (stride 16) then bilinearly upsample
  both to the same spatial resolution and concatenate.
- Average-pool each spatial position across channels → patch-level embedding.
- All inference runs in torch.no_grad() with AMP for maximum GPU throughput.
- Automatic device detection: CUDA > MPS (Apple Silicon) > CPU.

GPU acceleration
----------------
On an NVIDIA GPU, feature extraction is 10–30× faster than CPU.
Pass device='cuda' or let the constructor auto-detect:
    extractor = FeatureExtractor()  # auto-selects best device

Reference: Roth et al. CVPR 2022 — Section 3.1
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import Wide_ResNet50_2_Weights

logger = logging.getLogger(__name__)

# ImageNet normalisation — required because WideResNet was trained on it
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _best_device() -> torch.device:
    """Auto-select the fastest available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU detected: %s (%s, %.1f GB VRAM)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).name,
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple MPS device selected")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found — using CPU")
    return device


class FeatureExtractor(nn.Module):
    """
    Extract multi-scale patch features from a frozen WideResNet-50-2 backbone.

    Parameters
    ----------
    layers : list of str
        Names of intermediate layers to hook. Default: ['layer2', 'layer3'].
    input_size : int
        Images are resized to (input_size × input_size) before extraction.
        Use 256 or 320 for higher-resolution feature maps (better localisation).
    device : str or torch.device or None
        'cuda', 'mps', 'cpu', or None (auto-select best available).
    amp : bool
        Use Automatic Mixed Precision (float16 on CUDA) for ~2× speedup.
        Has no effect on CPU/MPS.

    Usage
    -----
    >>> extractor = FeatureExtractor()          # auto GPU
    >>> extractor = FeatureExtractor('cuda')    # explicit GPU
    >>> batch = extractor.preprocess([bgr_img]) # (1, 3, 224, 224)
    >>> patches, spatial = extractor.extract_patches(batch)  # (N, 1536)
    """

    def __init__(
        self,
        layers: List[str] = None,
        input_size: int = 224,
        device: str | torch.device | None = None,
        amp: bool = True,
    ) -> None:
        super().__init__()
        self.layers     = layers or ["layer2", "layer3"]
        self.input_size = input_size
        self.device     = torch.device(device) if device else _best_device()
        self.amp        = amp and (self.device.type == "cuda")

        # Load backbone — weights=DEFAULT uses latest ImageNet-1k checkpoint
        backbone = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

        # Move to device — use channels_last memory format for CUDA speedup
        if self.device.type == "cuda":
            backbone = backbone.to(memory_format=torch.channels_last)
        backbone.to(self.device)
        self._backbone = backbone

        # Forward hooks capture intermediate feature maps
        self._features: dict = {}
        self._hooks: List = []
        for name in self.layers:
            layer = dict(backbone.named_children())[name]
            hook  = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

        # Input preprocessing pipeline (runs on CPU, then tensor moves to GPU)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size),
                               interpolation=transforms.InterpolationMode.BICUBIC,
                               antialias=True),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        logger.info(
            "FeatureExtractor ready — WideResNet-50-2, layers: %s, device: %s, AMP: %s",
            self.layers, self.device, self.amp,
        )

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = output.detach()
        return hook

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract and concatenate multi-scale patch features.

        Parameters
        ----------
        images : torch.Tensor, shape (N, 3, H, W), float32, normalised

        Returns
        -------
        torch.Tensor, shape (N, C_total, H_out, W_out)
        """
        images = images.to(self.device)

        # Channels-last for CUDA efficiency
        if self.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        self._features.clear()

        # AMP context: float16 matmuls on GPU → ~2× throughput
        if self.amp:
            with torch.cuda.amp.autocast():
                _ = self._backbone(images)
        else:
            _ = self._backbone(images)

        feature_maps = [self._features[name] for name in self.layers]

        # Upsample all maps to the largest spatial resolution
        target_h, target_w = feature_maps[0].shape[2], feature_maps[0].shape[3]
        aligned = [
            F.interpolate(f.float(), size=(target_h, target_w),
                          mode="bilinear", align_corners=False)
            for f in feature_maps
        ]
        return torch.cat(aligned, dim=1)   # (N, C1+C2, H, W)

    @torch.no_grad()
    def extract_patches(self, images: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract patch embeddings from images and return as a 2-D numpy array.

        Parameters
        ----------
        images : torch.Tensor, shape (N, 3, H, W)

        Returns
        -------
        patches : np.ndarray, shape (N * H_f * W_f, C_total)
        spatial_shape : (H_f, W_f)
        """
        feat = self.extract(images)           # (N, C, H_f, W_f)
        N, C, H_f, W_f = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        return patches.astype(np.float32), (H_f, W_f)

    def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Convert a list of BGR uint8 OpenCV images to a normalised torch tensor.

        Parameters
        ----------
        images : list of np.ndarray — BGR uint8

        Returns
        -------
        torch.Tensor, shape (N, 3, input_size, input_size)
        """
        tensors = []
        for img in images:
            rgb = img[:, :, ::-1].copy()
            t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t   = self.transform(t)
            tensors.append(t)
        return torch.stack(tensors)

    def warmup(self, n_runs: int = 3) -> None:
        """
        Run dummy forward passes to warm up CUDA kernels.
        Call once before timing benchmarks or starting a video loop.
        """
        dummy = torch.zeros(1, 3, self.input_size, self.input_size)
        for _ in range(n_runs):
            self.extract(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        logger.info("GPU warmup complete (%d runs)", n_runs)

    def __del__(self):
        for h in self._hooks:
            h.remove()
