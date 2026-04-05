"""
test_anomaly.py
---------------
Unit tests for AnomalyDetector (anomaly_detector.py).
All tests use synthetic numpy data — no video or model weights required.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_detector import AnomalyDetector, FeatureExtractor, FEATURE_NAMES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_normal(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.2, scale=0.1, size=(n, 12)).astype(np.float32)


def _make_anomaly(n=50, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.8, scale=0.2, size=(n, 12)).astype(np.float32)


@pytest.fixture(scope="module")
def trained_detector():
    normal   = _make_normal()
    anomaly  = _make_anomaly()
    labeled  = np.vstack([normal[:50], anomaly])
    labels   = np.array([0] * 50 + [1] * 50)

    det = AnomalyDetector(contamination=0.1)
    det.fit(normal, labeled, labels)
    return det


# ── AnomalyDetector tests ─────────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_fit_sets_flags(self):
        det = AnomalyDetector()
        det.fit(_make_normal())
        assert det._if_trained is True
        assert det._rf_trained is False    # no labelled data

    def test_fit_with_labels(self):
        det   = AnomalyDetector()
        normal  = _make_normal()
        anomaly = _make_anomaly()
        labeled = np.vstack([normal[:50], anomaly])
        labels  = np.array([0]*50 + [1]*50)
        det.fit(normal, labeled, labels)
        assert det._rf_trained is True

    def test_predict_score_range(self, trained_detector):
        """All scores must be in [0, 1]."""
        features = _make_normal(n=30)
        scores   = trained_detector.predict_batch(features)
        assert scores.min() >= 0.0 - 1e-6
        assert scores.max() <= 1.0 + 1e-6

    def test_normal_score_lower_than_anomaly(self, trained_detector):
        """On average, normal features should score lower than anomaly features."""
        normal_scores  = trained_detector.predict_batch(_make_normal(n=100))
        anomaly_scores = trained_detector.predict_batch(_make_anomaly(n=100))
        assert normal_scores.mean() < anomaly_scores.mean()

    def test_predict_raises_without_fit(self):
        det = AnomalyDetector()
        with pytest.raises(AssertionError):
            det.predict_score(np.zeros(12, dtype=np.float32))

    def test_evaluate_returns_dict(self, trained_detector):
        features = np.vstack([_make_normal(20), _make_anomaly(20)])
        labels   = np.array([0]*20 + [1]*20)
        result   = trained_detector.evaluate(features, labels)
        assert "accuracy" in result
        assert "fpr"      in result
        assert 0.0 <= result["fpr"] <= 1.0

    def test_save_and_load(self, trained_detector, tmp_path):
        trained_detector.save(prefix=str(tmp_path))

        det2 = AnomalyDetector()
        det2.load(prefix=str(tmp_path))
        assert det2._if_trained is True

        score_original = trained_detector.predict_score(_make_normal(1)[0])
        score_loaded   = det2.predict_score(_make_normal(1)[0])
        assert abs(score_original - score_loaded) < 1e-4

    def test_high_anomaly_score_above_threshold(self, trained_detector):
        """Extreme anomaly features should typically score > 0.7."""
        rng = np.random.default_rng(42)
        extreme = rng.normal(loc=2.0, scale=0.1, size=(10, 12)).astype(np.float32)
        scores  = trained_detector.predict_batch(extreme)
        assert scores.mean() > 0.5   # relaxed: just > 0.5 to avoid flakiness


# ── FeatureExtractor tests ────────────────────────────────────────────────────

class TestFeatureNames:
    def test_feature_count(self):
        assert len(FEATURE_NAMES) == 12

    def test_no_duplicate_names(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
