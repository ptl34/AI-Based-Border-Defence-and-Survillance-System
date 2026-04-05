"""
anomaly_detector.py
-------------------
Ensemble anomaly detection using:
  • Isolation Forest  (unsupervised — no labelled data required)
  • Random Forest     (supervised  — uses labelled normal/anomaly features)

Final score = 0.6 * IsolationForest + 0.4 * RandomForest  (weighted vote)
Score range: 0.0 (normal)  →  1.0 (high anomaly)

Usage:
    from src.anomaly_detector import AnomalyDetector, FeatureExtractor
    detector = AnomalyDetector()
    detector.fit(normal_features, labeled_features, labels)
    score = detector.predict_score(feature_vector)
    detector.save()
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score)
from ultralytics import YOLO
import cv2


# ──────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION  (12-dimensional vector per frame)
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "num_persons",       # 0
    "num_vehicles",      # 1
    "num_weapons",       # 2
    "total_objects",     # 3
    "avg_confidence",    # 4
    "max_confidence",    # 5
    "brightness_mean",   # 6
    "brightness_var",    # 7
    "mean_flow",         # 8  (optical flow magnitude mean)
    "std_flow",          # 9  (optical flow magnitude std)
    "weapon_flag",       # 10 (1 if any weapon detected)
    "crowd_flag",        # 11 (1 if >5 persons in frame)
]


class FeatureExtractor:
    """Extracts a fixed 12-dim feature vector from a frame + YOLO results."""

    def __init__(self, yolo_model: YOLO):
        self.model   = yolo_model
        self._prev   = None          # previous grayscale frame for optical flow

    def reset(self):
        """Call between videos to clear the optical-flow buffer."""
        self._prev = None

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR uint8 array (640×640 recommended).

        Returns:
            float32 numpy array of shape (12,)
        """
        results = self.model(frame, verbose=False)[0]
        boxes   = results.boxes

        # ── Detection features ───────────────────────────────────────────────
        num_persons  = int(sum(1 for c in boxes.cls if int(c) == 0))
        num_vehicles = int(sum(1 for c in boxes.cls if int(c) == 1))
        num_weapons  = int(sum(1 for c in boxes.cls if int(c) == 2))
        total_objs   = len(boxes)
        avg_conf     = float(boxes.conf.mean()) if total_objs > 0 else 0.0
        max_conf     = float(boxes.conf.max())  if total_objs > 0 else 0.0

        # ── Image statistics ─────────────────────────────────────────────────
        gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        brightness_m  = float(gray.mean())
        brightness_v  = float(gray.var())

        # ── Optical flow ─────────────────────────────────────────────────────
        if self._prev is not None:
            flow      = cv2.calcOpticalFlowFarneback(
                            self._prev,
                            gray.astype(np.uint8),
                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _    = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_flow = float(mag.mean())
            std_flow  = float(mag.std())
        else:
            mean_flow = 0.0
            std_flow  = 0.0

        self._prev = gray.astype(np.uint8)

        # ── Boolean flags ────────────────────────────────────────────────────
        weapon_flag = float(num_weapons > 0)
        crowd_flag  = float(num_persons > 5)

        return np.array([
            num_persons, num_vehicles, num_weapons, total_objs,
            avg_conf, max_conf, brightness_m, brightness_v,
            mean_flow, std_flow, weapon_flag, crowd_flag,
        ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  ANOMALY DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Ensemble of Isolation Forest + Random Forest.
    The Isolation Forest is always trained (unsupervised).
    The Random Forest is optional — only trained when labelled data is provided.
    """

    MODEL_PATHS = {
        "isolation_forest": "models/isolation_forest.pkl",
        "random_forest":    "models/random_forest.pkl",
        "scaler":           "models/scaler.pkl",
    }

    def __init__(self, contamination: float = 0.1):
        """
        Args:
            contamination: Expected fraction of anomalies in training data.
                           IsolationForest hyperparameter. Default 0.10.
        """
        self.iso_forest    = IsolationForest(
            contamination  = contamination,
            n_estimators   = 200,
            random_state   = 42,
            n_jobs         = -1,
        )
        self.rf_classifier = RandomForestClassifier(
            n_estimators   = 200,
            max_depth      = 8,
            random_state   = 42,
            n_jobs         = -1,
        )
        self.scaler        = StandardScaler()
        self._rf_trained   = False
        self._if_trained   = False

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self,
            normal_features:  np.ndarray,
            labeled_features: np.ndarray = None,
            labels:           np.ndarray = None) -> None:
        """
        Train the ensemble.

        Args:
            normal_features:  Shape (N, 12) — normal-only frames for IF.
            labeled_features: Shape (M, 12) — labelled frames for RF (optional).
            labels:           Shape (M,)    — 0=normal, 1=anomaly (optional).
        """
        print("[AnomalyDetector] Fitting scaler on normal features …")
        self.scaler.fit(normal_features)
        X_normal = self.scaler.transform(normal_features)

        print("[AnomalyDetector] Training Isolation Forest …")
        self.iso_forest.fit(X_normal)
        self._if_trained = True
        print(f"  Isolation Forest trained on {len(normal_features)} samples.")

        if labeled_features is not None and labels is not None:
            X_lab = self.scaler.transform(labeled_features)
            print("[AnomalyDetector] Training Random Forest …")
            self.rf_classifier.fit(X_lab, labels)
            self._rf_trained = True
            print(f"  Random Forest trained on {len(labeled_features)} samples.")
        else:
            print("  [INFO] No labelled data — Random Forest skipped. "
                  "Ensemble will use Isolation Forest only.")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_score(self, features: np.ndarray) -> float:
        """
        Compute an anomaly score for a single feature vector.

        Returns:
            Float in [0.0, 1.0].  > 0.7 = HIGH,  0.4–0.7 = MEDIUM,  < 0.4 = LOW
        """
        assert self._if_trained, "Call fit() before predict_score()."
        X = self.scaler.transform(features.reshape(1, -1))

        # Isolation Forest: decision_function returns negative scores for anomalies
        # typical range ~ [-0.5, 0.5]; we flip and normalise to [0, 1]
        raw_if  = float(self.iso_forest.decision_function(X)[0])
        iso_score = 1.0 - (raw_if + 0.5)        # higher = more anomalous
        iso_score = float(np.clip(iso_score, 0.0, 1.0))

        if self._rf_trained:
            rf_proba   = float(self.rf_classifier.predict_proba(X)[0][1])
            final      = 0.6 * iso_score + 0.4 * rf_proba
        else:
            final      = iso_score

        return float(np.clip(final, 0.0, 1.0))

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for a batch of feature vectors."""
        return np.array([self.predict_score(f) for f in features])

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, features: np.ndarray,
                 labels: np.ndarray,
                 threshold: float = 0.5) -> dict:
        """
        Evaluate ensemble on labelled test data.

        Args:
            features:  Shape (N, 12)
            labels:    Shape (N,)  — 0=normal, 1=anomaly
            threshold: Score above this = predicted anomaly.

        Returns:
            Dict with accuracy, false_positive_rate, classification_report.
        """
        scores   = self.predict_batch(features)
        preds    = (scores >= threshold).astype(int)

        cm       = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr      = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

        print("\n── Anomaly Detection Evaluation ──────────────────")
        print(f"  Accuracy:           {accuracy:.3f}")
        print(f"  False Positive Rate:{fpr:.3f}  (target < 0.20)")
        try:
            auc = roc_auc_score(labels, scores)
            print(f"  ROC-AUC:            {auc:.3f}")
        except Exception:
            auc = None
        print(classification_report(labels, preds,
                                    target_names=["Normal", "Anomaly"]))
        return {"accuracy": accuracy, "fpr": fpr, "auc": auc}

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, prefix: str = "models") -> None:
        """Save all model artefacts to disk."""
        os.makedirs(prefix, exist_ok=True)
        joblib.dump(self.iso_forest,
                    os.path.join(prefix, "isolation_forest.pkl"))
        joblib.dump(self.rf_classifier,
                    os.path.join(prefix, "random_forest.pkl"))
        joblib.dump(self.scaler,
                    os.path.join(prefix, "scaler.pkl"))
        print(f"[AnomalyDetector] Models saved to {prefix}/")

    def load(self, prefix: str = "models") -> None:
        """Load persisted models from disk."""
        self.iso_forest    = joblib.load(os.path.join(prefix, "isolation_forest.pkl"))
        self.scaler        = joblib.load(os.path.join(prefix, "scaler.pkl"))
        self._if_trained   = True

        rf_path = os.path.join(prefix, "random_forest.pkl")
        if os.path.exists(rf_path):
            self.rf_classifier = joblib.load(rf_path)
            self._rf_trained   = True

        print(f"[AnomalyDetector] Loaded from {prefix}/  "
              f"(RF={'yes' if self._rf_trained else 'no'})")


# ──────────────────────────────────────────────────────────────────────────────
#  QUICK SMOKE-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smoke-testing AnomalyDetector with synthetic data …")
    rng = np.random.default_rng(0)

    normal_data  = rng.normal(loc=0.2, scale=0.1, size=(300, 12)).astype(np.float32)
    anomaly_data = rng.normal(loc=0.8, scale=0.2, size=(50, 12)).astype(np.float32)

    labeled  = np.vstack([normal_data[:50], anomaly_data])
    labels   = np.array([0] * 50 + [1] * 50)

    detector = AnomalyDetector(contamination=0.15)
    detector.fit(normal_data, labeled, labels)

    test_normal  = rng.normal(0.2, 0.1, (20, 12)).astype(np.float32)
    test_anomaly = rng.normal(0.8, 0.2, (20, 12)).astype(np.float32)
    test_X       = np.vstack([test_normal, test_anomaly])
    test_y       = np.array([0] * 20 + [1] * 20)

    detector.evaluate(test_X, test_y)
    detector.save()
    print("Done — models saved to models/")
