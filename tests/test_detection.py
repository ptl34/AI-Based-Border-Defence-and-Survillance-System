"""
test_detection.py
-----------------
Unit tests for ObjectDetector (detect_objects.py).
These tests run without a GPU and without the custom .pt weights
by using the standard YOLOv8n COCO model on a synthetic image.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detect_objects import ObjectDetector, Detection, CLASS_NAMES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector():
    """Shared ObjectDetector instance (downloads yolov8n.pt once)."""
    return ObjectDetector(weights_path="models/yolov8_border.pt",
                          confidence=0.25)


@pytest.fixture
def blank_frame():
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    return np.ones((640, 640, 3), dtype=np.uint8) * 255


# ── Detection class tests ────────────────────────────────────────────────────

class TestDetection:
    def test_bbox_property(self):
        d = Detection(0, "person", 0.9, 10, 20, 100, 200)
        assert d.bbox == (10, 20, 100, 200)

    def test_area(self):
        d = Detection(0, "person", 0.9, 0, 0, 100, 50)
        assert d.area == 5000

    def test_to_dict_keys(self):
        d = Detection(1, "vehicle", 0.75, 5, 5, 50, 50)
        keys = d.to_dict().keys()
        assert "class_id"   in keys
        assert "class_name" in keys
        assert "confidence" in keys
        assert "bbox"       in keys
        assert "area"       in keys

    def test_confidence_rounded(self):
        d = Detection(0, "person", 0.88888, 0, 0, 10, 10)
        assert d.to_dict()["confidence"] == round(0.88888, 4)


# ── ObjectDetector tests ─────────────────────────────────────────────────────

class TestObjectDetector:
    def test_instantiation(self, detector):
        assert detector is not None
        assert detector.model is not None

    def test_detect_returns_list(self, detector, blank_frame):
        result = detector.detect_frame(blank_frame)
        assert isinstance(result, list)

    def test_detect_no_objects_blank(self, detector, blank_frame):
        """A blank black frame should return zero or few detections."""
        result = detector.detect_frame(blank_frame)
        assert len(result) == 0   # no objects in empty frame

    def test_draw_detections_shape(self, detector, blank_frame):
        """draw_detections must return an array of same shape as input."""
        annotated = detector.draw_detections(blank_frame, [])
        assert annotated.shape == blank_frame.shape

    def test_count_classes_structure(self, detector):
        fake_dets = [
            Detection(0, "person",  0.9, 0,   0,  10, 10),
            Detection(0, "person",  0.8, 20,  0,  30, 10),
            Detection(1, "vehicle", 0.7, 50, 50, 100, 100),
        ]
        counts = detector._count_classes(fake_dets)
        assert counts["person"]  == 2
        assert counts["vehicle"] == 1
        assert counts["weapon"]  == 0
        assert counts["total"]   == 3

    def test_detect_frame_returns_detection_objects(self, detector, white_frame):
        dets = detector.detect_frame(white_frame)
        for d in dets:
            assert isinstance(d, Detection)
            assert 0.0 <= d.confidence <= 1.0
            assert d.x2 > d.x1
            assert d.y2 > d.y1
