# Industrial Defect Detection — PatchCore on MVTec AD

**Author:** Sherif Abd El-Rady — Computer Vision Engineer  
**Techniques:** Anomaly Detection · Deep Feature Extraction · Coreset Subsampling · k-NN Scoring

---

## Overview

This project implements **PatchCore** — the state-of-the-art unsupervised anomaly detection algorithm (Roth et al., CVPR 2022) — applied to industrial surface inspection using the MVTec Anomaly Detection dataset.

The system detects manufacturing defects (scratches, dents, contamination) by comparing deep CNN patch embeddings of test images against a coreset-sampled memory bank of normal-image embeddings.

**Key achievement:** PatchCore achieves **0.998 image-level ROC-AUC** on the MVTec AD bottle category using a frozen backbone with zero training of model weights.

---

## Architecture

```
Normal training images
        │
        ▼
WideResNet-50-2 (frozen, ImageNet)
Feature extraction: layer2 (stride-8) + layer3 (stride-16) → 1536-D patches
        │
        ▼
Greedy coreset subsampling (1% ratio, 128-D projection)
        │
Memory bank (disk-persisted)
        │
Test image
        │
        ▼
Extract patches → k-NN distance to bank
        │
Anomaly map (bilinear upsample + Gaussian smooth) + Image score (max)
        │
ROC-AUC / AP / IoU evaluation
```

---

## Methods Compared

| Method | Description | Image ROC-AUC (MVTec bottle) |
|---|---|---|
| **Classical baseline** | Pixel-mean deviation + SSIM | ~0.75–0.85 |
| **PatchCore** | WideResNet-50-2 + coreset k-NN | 0.998 |

---

## Notebook Sequence

| Notebook | Topic |
|---|---|
| `00_problem_and_setup.ipynb` | Problem definition, dataset setup, smoke test |
| `01_dataset_exploration.ipynb` | Class distribution, normal/defect visualisation |
| `02_classical_baseline.ipynb` | SSIM-based baseline — ROC-AUC / AP benchmark |
| `03_feature_extraction.ipynb` | WideResNet-50-2 features, t-SNE, activation maps |
| `04_memory_bank.ipynb` | Coreset subsampling — coverage vs size analysis |
| `05_patchcore_inference.ipynb` | Score maps, distributions, inference timing |
| `06_evaluation.ipynb` | ROC-AUC, PR, pixel IoU/Dice — PatchCore vs baseline |
| `07_failure_analysis.ipynb` | FP/FN examples, threshold sensitivity, failure taxonomy |
| `08_production_pipeline.ipynb` | `train_patchcore()` + `predict_patchcore()` functions |

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Dataset

MVTec AD requires manual download (CC BY-NC-SA 4.0 — research use only):

1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Accept the license, download `mvtec_anomaly_detection.tar.xz`
3. Extract to `data/mvtec/`

All notebooks fall back to synthetic images automatically if the dataset is not present.

---

## CLI Usage

```bash
# Build memory bank
defectdetect-train \
  --train-dir data/mvtec/bottle/train/good \
  --bank-out  reports/memory_banks/bottle.pkl \
  --ratio     0.01

# Score a test image
defectdetect-predict \
  --bank      reports/memory_banks/bottle.pkl \
  --image     data/mvtec/bottle/test/broken_large/000.png \
  --threshold 0.5 \
  --save-map  reports/anomaly_map.png
```

---

## Package API

```python
from defectdetect import FeatureExtractor, MemoryBank, AnomalyEvaluator
from defectdetect.scorer import AnomalyScorer

# Extract features
extractor = FeatureExtractor(layers=['layer2', 'layer3'], device='cpu')
batch     = extractor.preprocess([bgr_image])       # OpenCV BGR uint8 → tensor
patches, spatial = extractor.extract_patches(batch)  # (N_patches, 1536)

# Score against memory bank
bank    = MemoryBank.load('reports/memory_banks/bottle.pkl')
scores  = bank.score(patches, k=1)                   # k-NN distances

# Compute anomaly map
scorer  = AnomalyScorer(image_size=(224, 224))
img_score, amap = scorer.compute(scores, spatial, n_images=1)

# Evaluate
ev = AnomalyEvaluator().fit(all_scores, all_labels)
metrics = ev.image_level_metrics()
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

---

## Running Tests

```bash
PYTHONPATH=D:\pypackages python -m pytest tests/ -v
```

---

## References

- Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022).  
  **Towards Total Recall in Industrial Anomaly Detection.** CVPR 2022.  
  https://arxiv.org/abs/2106.08265

- Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019).  
  **MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.** CVPR 2019.  
  https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## License

MIT — see [LICENSE](LICENSE)
