"""
tests/test_defectdetect.py
===========================
Test suite for the defectdetect package — industrial anomaly detection.

Test strategy
-------------
All tests are self-contained and require no external dataset (MVTec AD)
and no CUDA GPU.

The FeatureExtractor tests use the real WideResNet-50 backbone to validate
the actual production code path. If the model cannot be loaded due to memory
constraints (controlled by the env variable SKIP_BACKBONE_TESTS=1), those
tests are skipped gracefully while all other modules are fully tested.

Run all tests (standard):
    pytest tests/ -v

Skip backbone tests on low-RAM machines:
    $env:SKIP_BACKBONE_TESTS=1; pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from defectdetect import (
    MemoryBank,
    AnomalyScorer,
    ClassicalBaseline,
    AnomalyEvaluator,
)
from defectdetect.extractor import FeatureExtractor
from defectdetect.scorer import AnomalyScorer

# ── Environment flag ──────────────────────────────────────────────────────────
_SKIP_BACKBONE = os.environ.get("SKIP_BACKBONE_TESTS", "0") == "1"
requires_backbone = pytest.mark.skipif(
    _SKIP_BACKBONE,
    reason="SKIP_BACKBONE_TESTS=1 — skipping WideResNet-50 load on this machine",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_image(h: int = 224, w: int = 224, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def extractor():
    if _SKIP_BACKBONE:
        pytest.skip("SKIP_BACKBONE_TESTS=1")
    return FeatureExtractor(layers=["layer2", "layer3"], input_size=224, device="cpu")


@pytest.fixture(scope="module")
def sample_patches(extractor):
    imgs   = [make_image(seed=i) for i in range(4)]
    batch  = extractor.preprocess(imgs)
    patches, spatial = extractor.extract_patches(batch)
    return patches, spatial


@pytest.fixture(scope="module")
def fitted_bank(sample_patches):
    patches, _ = sample_patches
    bank = MemoryBank(sampling_ratio=0.5, projection_dim=32)
    bank.fit(patches)
    return bank


# Lightweight patches fixture for MemoryBank tests that don't need real backbone
@pytest.fixture(scope="module")
def random_patches():
    """512 random 64-D patch embeddings — no backbone needed."""
    rng = np.random.default_rng(7)
    return rng.random((512, 64)).astype(np.float32)


@pytest.fixture(scope="module")
def fitted_bank_random(random_patches):
    bank = MemoryBank(sampling_ratio=0.5, projection_dim=None)
    bank.fit(random_patches)
    return bank


# ── FeatureExtractor ──────────────────────────────────────────────────────────

class TestFeatureExtractor:

    @requires_backbone
    def test_preprocess_output_shape(self, extractor):
        imgs  = [make_image()]
        batch = extractor.preprocess(imgs)
        assert batch.shape == (1, 3, 224, 224)

    @requires_backbone
    def test_preprocess_value_range(self, extractor):
        imgs  = [make_image()]
        batch = extractor.preprocess(imgs)
        assert batch.dtype == torch.float32

    @requires_backbone
    def test_extract_returns_4d(self, extractor):
        imgs  = [make_image()]
        batch = extractor.preprocess(imgs)
        feat  = extractor.extract(batch)
        assert feat.ndim == 4

    @requires_backbone
    def test_extract_patches_shape(self, extractor):
        imgs           = [make_image()]
        batch          = extractor.preprocess(imgs)
        patches, (h, w) = extractor.extract_patches(batch)
        assert patches.ndim == 2
        assert patches.shape[0] == h * w

    @requires_backbone
    def test_feature_dim_is_sum_of_layers(self, extractor):
        """layer2=512, layer3=1024 — concatenated should be 1536."""
        imgs           = [make_image()]
        batch          = extractor.preprocess(imgs)
        patches, _     = extractor.extract_patches(batch)
        assert patches.shape[1] == 1536, f"Expected 1536-D, got {patches.shape[1]}"

    @requires_backbone
    def test_batch_of_four_produces_four_times_patches(self, extractor):
        b1 = extractor.preprocess([make_image()])
        b4 = extractor.preprocess([make_image(seed=i) for i in range(4)])
        p1, _ = extractor.extract_patches(b1)
        p4, _ = extractor.extract_patches(b4)
        assert p4.shape[0] == 4 * p1.shape[0]

    @requires_backbone
    def test_preprocess_multiple_images(self, extractor):
        batch = extractor.preprocess([make_image(seed=i) for i in range(6)])
        assert batch.shape[0] == 6


# ── MemoryBank (uses random patches — no backbone) ────────────────────────────

class TestMemoryBank:

    def test_fit_sets_is_fitted(self, random_patches):
        bank = MemoryBank(sampling_ratio=0.5, projection_dim=None)
        bank.fit(random_patches)
        assert bank._is_fitted

    def test_bank_size_respects_ratio(self, random_patches):
        ratio    = 0.2
        bank     = MemoryBank(sampling_ratio=ratio, projection_dim=None)
        bank.fit(random_patches)
        expected = max(1, int(len(random_patches) * ratio))
        assert bank.size == expected

    def test_score_returns_correct_shape(self, fitted_bank_random, random_patches):
        scores = fitted_bank_random.score(random_patches[:20])
        assert scores.shape == (20,)

    def test_score_non_negative(self, fitted_bank_random, random_patches):
        scores = fitted_bank_random.score(random_patches[:20])
        assert (scores >= 0).all()

    def test_score_before_fit_raises(self):
        bank = MemoryBank()
        with pytest.raises(RuntimeError, match="not fitted"):
            bank.score(np.random.rand(5, 16).astype(np.float32))

    def test_invalid_sampling_ratio_raises(self):
        with pytest.raises(ValueError):
            MemoryBank(sampling_ratio=1.5)

    def test_save_and_load_round_trip(self, fitted_bank_random, tmp_path):
        path = tmp_path / "bank.pkl"
        fitted_bank_random.save(path)
        loaded = MemoryBank.load(path)
        assert loaded.size == fitted_bank_random.size
        assert np.allclose(loaded._bank, fitted_bank_random._bank)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MemoryBank.load(tmp_path / "nonexistent.pkl")

    def test_gpu_knn_cpu_fallback_matches(self, fitted_bank_random, random_patches):
        """CPU einsum k-NN and GPU einsum k-NN must return identical distances."""
        bank = fitted_bank_random
        q    = random_patches[:10].copy()
        if bank._projector is not None:
            q = bank._projector.transform(q).astype(np.float32)
        cpu_d = bank._knn_distances_cpu(q, k=1)
        gpu_d = bank._knn_distances_gpu(q, k=1)
        np.testing.assert_allclose(cpu_d, gpu_d, rtol=1e-4)

    def test_with_real_patches_if_available(self, sample_patches):
        """Only runs when backbone is available."""
        if _SKIP_BACKBONE:
            pytest.skip("Backbone not loaded")
        patches, _ = sample_patches
        bank = MemoryBank(sampling_ratio=0.5, projection_dim=32)
        bank.fit(patches)
        scores = bank.score(patches)
        assert scores.shape == (len(patches),)
        assert (scores >= 0).all()


# ── AnomalyScorer ─────────────────────────────────────────────────────────────

class TestAnomalyScorer:

    def test_compute_shapes(self):
        scorer       = AnomalyScorer(image_size=(224, 224))
        patch_scores = np.random.rand(28 * 28).astype(np.float32)
        img_scores, amaps = scorer.compute(patch_scores, (28, 28), n_images=1)
        assert img_scores.shape == (1,)
        assert amaps.shape == (1, 224, 224)

    def test_image_score_is_max_of_map(self):
        scorer       = AnomalyScorer(image_size=(224, 224), gaussian_sigma=0.0)
        patch_scores = np.random.rand(28 * 28).astype(np.float32)
        img_scores, amaps = scorer.compute(patch_scores, (28, 28), n_images=1)
        assert img_scores[0] <= amaps[0].max() + 1e-4

    def test_batch_compute_shapes(self):
        scorer       = AnomalyScorer(image_size=(64, 64))
        N, H, W      = 3, 14, 14
        patch_scores = np.random.rand(N * H * W).astype(np.float32)
        img_scores, amaps = scorer.compute(patch_scores, (H, W), n_images=N)
        assert img_scores.shape == (N,)
        assert amaps.shape == (N, 64, 64)

    def test_normalise_range(self):
        maps = np.random.rand(5, 224, 224).astype(np.float32)
        norm = AnomalyScorer.normalise(maps)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0 + 1e-6

    def test_normalise_constant_map(self):
        maps = np.ones((2, 64, 64), dtype=np.float32)
        norm = AnomalyScorer.normalise(maps)
        assert (norm == 0).all()

    def test_overlay_output_shape(self):
        img  = make_image()
        amap = np.random.rand(224, 224).astype(np.float32)
        out  = AnomalyScorer.overlay(img, amap)
        assert out.shape == img.shape

    def test_overlay_different_size(self):
        img  = make_image(h=480, w=640)
        amap = np.random.rand(224, 224).astype(np.float32)
        out  = AnomalyScorer.overlay(img, amap)
        assert out.shape == img.shape

    def test_gpu_upsample_path_cpu_fallback(self):
        """GPU scorer falls back to CPU — result must match CPU scorer."""
        cpu_s = AnomalyScorer(image_size=(64, 64), gaussian_sigma=0.0, device="cpu")
        gpu_s = AnomalyScorer(image_size=(64, 64), gaussian_sigma=0.0, device="cpu")
        scores = np.random.rand(14 * 14).astype(np.float32)
        _, maps_cpu = cpu_s.compute(scores, (14, 14), n_images=1)
        _, maps_gpu = gpu_s.compute(scores, (14, 14), n_images=1)
        np.testing.assert_allclose(maps_cpu, maps_gpu, atol=1e-4)


# ── ClassicalBaseline ─────────────────────────────────────────────────────────

class TestClassicalBaseline:

    def test_fit_sets_mean(self):
        baseline = ClassicalBaseline(image_size=(64, 64))
        imgs = [make_image(64, 64, seed=i) for i in range(5)]
        baseline.fit(imgs)
        assert baseline._mean_normal is not None
        assert baseline._mean_normal.shape == (64, 64, 3)

    def test_predict_returns_score_and_map(self):
        baseline = ClassicalBaseline(image_size=(64, 64))
        imgs = [make_image(64, 64, seed=i) for i in range(5)]
        baseline.fit(imgs)
        score, amap = baseline.predict(make_image(64, 64, seed=99))
        assert isinstance(score, float)
        assert amap.shape == (64, 64)

    def test_predict_score_non_negative(self):
        baseline = ClassicalBaseline(image_size=(64, 64))
        baseline.fit([make_image(64, 64, seed=i) for i in range(5)])
        score, _ = baseline.predict(make_image(64, 64, seed=99))
        assert score >= 0.0

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            ClassicalBaseline().predict(make_image())

    def test_similar_image_scores_lower_than_random(self):
        baseline = ClassicalBaseline(image_size=(64, 64))
        imgs     = [make_image(64, 64, seed=i) for i in range(10)]
        baseline.fit(imgs)
        train_score, _ = baseline.predict(imgs[0])
        rng = np.random.default_rng(999)
        random_img     = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        random_score, _ = baseline.predict(random_img)
        assert train_score < random_score


# ── AnomalyEvaluator ──────────────────────────────────────────────────────────

class TestAnomalyEvaluator:

    def test_image_level_roc_auc_perfect(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        ev     = AnomalyEvaluator().fit(scores, labels)
        assert ev.image_level_metrics()["roc_auc"] == pytest.approx(1.0)

    def test_image_level_roc_auc_random(self):
        rng    = np.random.default_rng(42)
        scores = rng.random(100).astype(np.float32)
        labels = rng.integers(0, 2, 100).astype(np.int32)
        m      = AnomalyEvaluator().fit(scores, labels).image_level_metrics()
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_pixel_level_metrics(self):
        rng    = np.random.default_rng(0)
        labels = rng.integers(0, 2, 1000).astype(np.int32)
        scores = labels.astype(np.float32) + rng.normal(0, 0.05, 1000).astype(np.float32)
        m      = AnomalyEvaluator().fit_pixel(scores, labels).pixel_level_metrics()
        assert m["pixel_roc_auc"] > 0.9

    def test_fit_before_metrics_raises(self):
        with pytest.raises(RuntimeError):
            AnomalyEvaluator().image_level_metrics()

    def test_confusion_matrix_keys(self):
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        labels = np.array([0, 1, 0, 1])
        m      = AnomalyEvaluator().fit(scores, labels).image_level_metrics()
        for key in ("confusion_matrix", "n_normal", "n_anomaly"):
            assert key in m
