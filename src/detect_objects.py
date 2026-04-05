"""
detect_objects.py
-----------------
YOLOv8 inference wrapper for border surveillance.
Detects: person (0), vehicle (1), weapon (2), suspicious_object (3).

Usage:
    from src.detect_objects import ObjectDetector
    detector = ObjectDetector("models/yolov8_border.pt")
    detections = detector.detect_frame(frame)
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_id:    int
    class_name:  str
    confidence:  float
    x1: int; y1: int; x2: int; y2: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_dict(self) -> dict:
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox":       [self.x1, self.y1, self.x2, self.y2],
            "area":       self.area,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  OBJECT DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "person",
    1: "vehicle",
    2: "weapon",
    3: "suspicious_object",
}

# Colours for bounding boxes (BGR)
BBOX_COLORS = {
    "person":            (0,  212, 255),   # cyan
    "vehicle":           (57, 255,  20),   # neon green
    "weapon":            (255, 49,  49),   # red
    "suspicious_object": (255, 165,   0),  # orange
}


class ObjectDetector:
    """
    Wraps a YOLOv8 model for border-surveillance inference.
    Falls back to the COCO-pretrained yolov8n.pt when no custom
    weights are found, so the code always runs out of the box.
    """

    def __init__(self, weights_path: str = "models/yolov8_border.pt",
                 confidence: float = 0.30):
        """
        Args:
            weights_path: Path to trained .pt weights file.
            confidence:   Minimum detection confidence (0–1).
        """
        self.confidence = confidence

        if Path(weights_path).exists():
            self.model = YOLO(weights_path)
            print(f"[ObjectDetector] Loaded custom weights: {weights_path}")
        else:
            print(f"[ObjectDetector] Custom weights not found — using yolov8n.pt")
            self.model = YOLO("yolov8n.pt")   # downloads ~6 MB if needed

    # ── Single frame ─────────────────────────────────────────────────────────

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.

        Returns:
            List of Detection objects (may be empty).
        """
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cls_name = CLASS_NAMES.get(cls_id, results.names.get(cls_id, f"cls_{cls_id}"))
            detections.append(Detection(cls_id, cls_name, conf, x1, y1, x2, y2))

        return detections

    # ── Draw bounding boxes ──────────────────────────────────────────────────

    def draw_detections(self, frame: np.ndarray,
                        detections: list[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a copy of the frame.

        Returns:
            Annotated frame (BGR uint8).
        """
        annotated = frame.copy()
        for det in detections:
            color = BBOX_COLORS.get(det.class_name, (200, 200, 200))
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated,
                          (det.x1, det.y1 - lh - 6),
                          (det.x1 + lw, det.y1), color, -1)
            cv2.putText(annotated, label,
                        (det.x1, det.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return annotated

    # ── Batch / video ────────────────────────────────────────────────────────

    def process_video(self, video_path: str,
                      output_path: str = None,
                      show: bool = False) -> list[dict]:
        """
        Run detection on every frame of a video.

        Args:
            video_path:  Input video path.
            output_path: If given, save annotated video here.
            show:        Display frames live (requires display).

        Returns:
            List of per-frame dicts: {frame_idx, detections, counts}
        """
        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_results = []
        frame_idx     = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets   = self.detect_frame(frame)
            counts = self._count_classes(dets)

            frame_results.append({
                "frame_idx":  frame_idx,
                "detections": [d.to_dict() for d in dets],
                "counts":     counts,
            })

            if writer or show:
                annotated = self.draw_detections(frame, dets)
                if writer:
                    writer.write(annotated)
                if show:
                    cv2.imshow("Border Surveillance", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        return frame_results

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _count_classes(detections: list[Detection]) -> dict:
        counts = {name: 0 for name in CLASS_NAMES.values()}
        for d in detections:
            if d.class_name in counts:
                counts[d.class_name] += 1
        counts["total"] = len(detections)
        return counts


# ──────────────────────────────────────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    detector = ObjectDetector()

    sample_frames = glob.glob("data/processed/**/*.jpg", recursive=True)[:5]
    if not sample_frames:
        print("[INFO] No sample frames found — download the dataset first.")
    else:
        for fp in sample_frames:
            frame = cv2.imread(fp)
            dets  = detector.detect_frame(frame)
            print(f"  {Path(fp).name}  →  {len(dets)} detections: "
                  f"{detector._count_classes(dets)}")
