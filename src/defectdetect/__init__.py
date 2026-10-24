"""
defectdetect — Industrial Visual Anomaly Detection
===================================================
PatchCore-inspired anomaly detection pipeline for industrial surface inspection.

Architecture
------------
1. Feature extraction  — WideResNet-50 intermediate layers (no fine-tuning)
2. Memory bank         — Coreset-sampled patch embeddings from normal training images
3. Anomaly scoring     — k-NN distance from test patch to nearest memory bank entry
4. Anomaly map         — Spatially upsampled score map for pixel-level localisation
5. Threshold selection — F1-maximising threshold on a validation split

GPU Acceleration (auto-detected)
---------------------------------
  - FeatureExtractor : CUDA forward pass + AMP float16 + channels_last layout
  - MemoryBank.score : torch einsum k-NN on GPU tensor (50-200× faster)
  - AnomalyScorer    : torch.nn.functional.interpolate on GPU
  - All paths fall back to CPU automatically

Reference
---------
Roth et al. (2022). Towards Total Recall in Industrial Anomaly Detection. CVPR 2022.
https://arxiv.org/abs/2106.08265

This is an original re-implementation — not a copy of any public codebase.
"""

from defectdetect.extractor import FeatureExtractor, _best_device
from defectdetect.memory_bank import MemoryBank
from defectdetect.scorer import AnomalyScorer
from defectdetect.baseline import ClassicalBaseline
from defectdetect.evaluator import AnomalyEvaluator

__version__ = "0.1.0"
__all__ = [
    "FeatureExtractor",
    "MemoryBank",
    "AnomalyScorer",
    "ClassicalBaseline",
    "AnomalyEvaluator",
    "_best_device",
]
