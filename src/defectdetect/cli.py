"""
defectdetect.cli
================
Command-line interface for the defectdetect package.

Entry points (defined in pyproject.toml):
  defectdetect-train    — build a memory bank from a folder of normal images
  defectdetect-predict  — score one or more images against a memory bank

Usage examples
--------------
# Training (one-time per product category):
  defectdetect-train \\
    --train-dir data/mvtec/bottle/train/good \\
    --bank-out  reports/memory_banks/bottle.pkl \\
    --ratio     0.01

# Inference:
  defectdetect-predict \\
    --bank       reports/memory_banks/bottle.pkl \\
    --image      path/to/test.png \\
    --threshold  0.5 \\
    --save-map   path/to/anomaly_map.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def train() -> None:
    """CLI entry point: build a memory bank from normal training images."""
    parser = argparse.ArgumentParser(
        prog="defectdetect-train",
        description="Build a PatchCore memory bank from normal training images.",
    )
    parser.add_argument("--train-dir",  type=Path, required=True,
                        help="Directory of normal training images.")
    parser.add_argument("--bank-out",   type=Path, required=True,
                        help="Output path for the memory bank (.pkl).")
    parser.add_argument("--ratio",      type=float, default=0.01,
                        help="Coreset sampling ratio (default: 0.01 = 1%%)")
    parser.add_argument("--layers",     nargs="+", default=["layer2", "layer3"],
                        help="WideResNet layer names to extract from.")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Image input resolution (default: 224).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Images per batch during feature extraction.")
    parser.add_argument("--device",     default="cpu",
                        help="'cpu' or 'cuda' (default: cpu)")
    args = parser.parse_args()

    # Lazy imports — avoids loading torch just for --help
    import cv2
    import numpy as np
    from tqdm import tqdm
    from defectdetect import FeatureExtractor, MemoryBank

    if not args.train_dir.exists():
        print(f"ERROR: train-dir not found: {args.train_dir}", file=sys.stderr)
        sys.exit(1)

    extractor = FeatureExtractor(layers=args.layers, input_size=args.input_size,
                                  device=args.device)
    paths = sorted(args.train_dir.glob("*.png")) + sorted(args.train_dir.glob("*.jpg"))
    if not paths:
        print(f"ERROR: No images found in {args.train_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Training on {len(paths)} images from: {args.train_dir}")
    all_patches = []
    for i in tqdm(range(0, len(paths), args.batch_size), desc="Extracting"):
        imgs = [cv2.imread(str(p)) for p in paths[i:i + args.batch_size]]
        imgs = [img for img in imgs if img is not None]
        if not imgs:
            continue
        batch = extractor.preprocess(imgs)
        pat, _ = extractor.extract_patches(batch)
        all_patches.append(pat)

    all_patches = np.vstack(all_patches)
    bank = MemoryBank(sampling_ratio=args.ratio, projection_dim=128)
    bank.fit(all_patches)
    bank.save(args.bank_out)

    print(f"\n✓ Memory bank saved → {args.bank_out}")
    print(f"  Total patches : {len(all_patches):,}")
    print(f"  Bank size     : {bank.size:,}")
    print(f"  Sampling ratio: {args.ratio}")


def predict() -> None:
    """CLI entry point: score one or more test images against a memory bank."""
    parser = argparse.ArgumentParser(
        prog="defectdetect-predict",
        description="Score test images for anomalies using a fitted memory bank.",
    )
    parser.add_argument("--bank",      type=Path, required=True,
                        help="Path to the memory bank .pkl file.")
    parser.add_argument("--image",     type=Path, nargs="+", required=True,
                        help="One or more image paths to score.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Anomaly score threshold (optional).")
    parser.add_argument("--save-map",  type=Path, default=None,
                        help="Save the anomaly map heatmap (only for single image).")
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from defectdetect import FeatureExtractor, MemoryBank
    from defectdetect.scorer import AnomalyScorer

    if not args.bank.exists():
        print(f"ERROR: bank not found: {args.bank}", file=sys.stderr)
        sys.exit(1)

    bank      = MemoryBank.load(args.bank)
    extractor = FeatureExtractor(layers=["layer2", "layer3"], input_size=224,
                                  device=args.device)
    scorer    = AnomalyScorer(image_size=(224, 224), gaussian_sigma=4.0)

    print(f"{'Image':<40} {'Score':>8}  {'Verdict'}")
    print("-" * 65)

    for img_path in args.image:
        if not img_path.exists():
            print(f"{str(img_path):<40}  ERROR: not found")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{str(img_path):<40}  ERROR: could not read")
            continue

        batch = extractor.preprocess([img])
        patches, sp = extractor.extract_patches(batch)
        ps = bank.score(patches, k=1)
        img_score, amap = scorer.compute(ps, sp, n_images=1)
        score = float(img_score[0])

        verdict = ""
        if args.threshold is not None:
            verdict = "⚠  ANOMALY" if score >= args.threshold else "✓  NORMAL"

        print(f"{str(img_path):<40} {score:>8.4f}  {verdict}")

        if args.save_map and len(args.image) == 1:
            amap_norm = amap[0]
            amap_norm = (amap_norm - amap_norm.min()) / (amap_norm.max() - amap_norm.min() + 1e-8)
            heat = cv2.applyColorMap((amap_norm * 255).astype("uint8"), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.5, heat, 0.5, 0)
            cv2.imwrite(str(args.save_map), overlay)
            print(f"\nAnomaly map saved → {args.save_map}")
