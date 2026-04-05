"""
run_pipeline.py
---------------
Main end-to-end script: video → frames → YOLO detection → anomaly score →
alert generation → (optional) Azure upload.

Usage:
    python src/run_pipeline.py --video data/sample/test_video.mp4
    python src/run_pipeline.py --video data/sample/test_video.mp4 --azure
    python src/run_pipeline.py --video data/sample/test_video.mp4 --show
"""

import argparse
import os
import sys
import time
import csv
import yaml
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from detect_objects  import ObjectDetector
from anomaly_detector import AnomalyDetector, FeatureExtractor
from alert_manager   import AlertManager
from ultralytics     import YOLO


# ──────────────────────────────────────────────────────────────────────────────
#  PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        print(f"[WARN] {path} not found — using defaults.")
        return {
            "model":  {"yolo_weights": "yolov8n.pt",
                       "yolo_confidence": 0.30,
                       "fp_filter_threshold": 0.50,
                       "anomaly_threshold_high": 0.70,
                       "anomaly_threshold_med":  0.40},
            "video":  {"fps_extract": 1,
                       "processed_dir": "data/processed"},
            "alert":  {"db_path": "alerts.db"},
            "logging": {"log_file": "results/pipeline.log"},
        }
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(video_path:   str,
                 config_path:  str  = "config.yaml",
                 use_azure:    bool = False,
                 show:         bool = False) -> dict:
    """
    Run the full surveillance pipeline on a single video.

    Args:
        video_path:  Path to input video (.mp4 / .avi).
        config_path: Path to config.yaml.
        use_azure:   If True, upload frames + alerts to Azure.
        show:        If True, display annotated frames live.

    Returns:
        Dict of pipeline run statistics.
    """
    cfg     = load_config(config_path)
    mcfg    = cfg["model"]
    vcfg    = cfg["video"]
    acfg    = cfg["alert"]

    print("\n" + "═" * 55)
    print("  Border Surveillance AI — Processing Pipeline")
    print("═" * 55)
    print(f"  Video   : {video_path}")
    print(f"  Azure   : {'enabled' if use_azure else 'disabled'}")
    print("─" * 55)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading models …")
    yolo_model    = YOLO(mcfg.get("yolo_weights", "yolov8n.pt"))
    obj_detector  = ObjectDetector(mcfg.get("yolo_weights", "yolov8n.pt"),
                                   confidence=mcfg.get("yolo_confidence", 0.30))
    feat_extractor = FeatureExtractor(yolo_model)

    anomaly_det   = AnomalyDetector()
    try:
        anomaly_det.load()
    except Exception:
        print("  [WARN] Trained anomaly models not found — "
              "training on-the-fly with dummy normal data.")
        dummy_normal = np.random.normal(0.2, 0.1, (200, 12)).astype(np.float32)
        anomaly_det.fit(dummy_normal)
        anomaly_det.save()

    alert_mgr = AlertManager(
        db_path        = acfg.get("db_path", "alerts.db"),
        conf_threshold = mcfg.get("fp_filter_threshold", 0.50),
    )

    # ── Optional Azure ────────────────────────────────────────────────────────
    uploader = None
    if use_azure:
        from azure_uploader import AzureUploader
        az = cfg.get("azure", {})
        try:
            uploader = AzureUploader(
                az["blob_connection_string"],
                az["cosmos_endpoint"],
                az["cosmos_key"],
            )
        except Exception as exc:
            print(f"  [WARN] Azure init failed: {exc}  — continuing without cloud upload.")

    # ── Open video ────────────────────────────────────────────────────────────
    print("\n[2/5] Opening video …")
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    interval     = max(1, int(video_fps / vcfg.get("fps_extract", 1)))

    print(f"  Resolution : {int(cap.get(3))}×{int(cap.get(4))}")
    print(f"  FPS        : {video_fps:.1f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Processing every {interval} frames")

    # ── Output directories ────────────────────────────────────────────────────
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_dir = os.path.join(vcfg.get("processed_dir", "data/processed"),
                             "pipeline_runs", run_id)
    os.makedirs(frame_dir,        exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    # ── Processing loop ───────────────────────────────────────────────────────
    print("\n[3/5] Processing frames …")
    stats = {
        "run_id":      run_id,
        "video":       video_path,
        "frames_processed": 0,
        "objects_detected": 0,
        "alerts_high":  0,
        "alerts_medium":0,
        "alerts_low":   0,
        "start_time":  time.time(),
    }

    csv_rows = []
    feat_extractor.reset()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            frame_resized = cv2.resize(frame, (640, 640))

            # ── YOLO detection ────────────────────────────────────────────
            detections = obj_detector.detect_frame(frame_resized)
            counts     = obj_detector._count_classes(detections)
            stats["objects_detected"] += counts["total"]

            # ── Feature extraction ────────────────────────────────────────
            features      = feat_extractor.extract(frame_resized)
            anomaly_score = anomaly_det.predict_score(features)

            # ── Confidence: use avg of YOLO detection confs ────────────────
            if detections:
                avg_conf = float(np.mean([d.confidence for d in detections]))
            else:
                avg_conf = 0.3   # fallback when no objects detected

            # ── Alert ─────────────────────────────────────────────────────
            frame_filename = f"frame_{stats['frames_processed']:05d}.jpg"
            frame_path     = os.path.join(frame_dir, frame_filename)
            cv2.imwrite(frame_path, frame_resized)

            obj_names = [d.class_name for d in detections]
            alert = alert_mgr.process(
                confidence       = avg_conf,
                anomaly_score    = anomaly_score,
                alert_type       = _infer_alert_type(counts),
                frame_path       = frame_path,
                location         = "sector_01",
                objects_detected = obj_names,
            )

            if alert:
                stats[f"alerts_{alert.priority.lower()}"] += 1
                # ── Azure upload (HIGH alerts only) ───────────────────────
                if uploader and alert.priority == "HIGH":
                    blob_url = uploader.upload_frame(frame_path, frame_filename)
                    d        = alert.to_dict()
                    d["frame_url"] = blob_url
                    uploader.save_alert(d)

            # ── CSV row ───────────────────────────────────────────────────
            csv_rows.append({
                "frame_idx":     frame_idx,
                "timestamp":     datetime.utcnow().isoformat(),
                "anomaly_score": round(anomaly_score, 4),
                "avg_confidence":round(avg_conf, 4),
                "priority":      alert.priority if alert else "FILTERED",
                "objects":       ",".join(obj_names),
                "num_persons":   counts["person"],
                "num_vehicles":  counts["vehicle"],
                "num_weapons":   counts["weapon"],
            })

            stats["frames_processed"] += 1

            # ── Live display ──────────────────────────────────────────────
            if show:
                annotated = obj_detector.draw_detections(frame_resized, detections)
                label = f"Score: {anomaly_score:.2f}  |  {alert.priority if alert else 'OK'}"
                color = (0,212,255) if not alert else {"HIGH":(0,0,255),"MEDIUM":(0,200,200),"LOW":(0,255,0)}.get(alert.priority,(200,200,200))
                cv2.putText(annotated, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("Border Surveillance", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    print("\n[4/5] Saving metrics …")
    csv_path = f"results/metrics/run_{run_id}.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  Metrics saved → {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - stats["start_time"]
    eff_fps = stats["frames_processed"] / elapsed if elapsed > 0 else 0

    print("\n[5/5] ─── Pipeline Complete ─────────────────────────────")
    print(f"  ✅  Frames processed  : {stats['frames_processed']}")
    print(f"  ✅  Objects detected  : {stats['objects_detected']}")
    print(f"  🔴  HIGH alerts       : {stats['alerts_high']}")
    print(f"  🟡  MEDIUM alerts     : {stats['alerts_medium']}")
    print(f"  🟢  LOW alerts        : {stats['alerts_low']}")
    print(f"  ⏱️   Elapsed time      : {elapsed:.1f}s")
    print(f"  ⚡  Effective FPS     : {eff_fps:.1f}")
    print(f"  📄  Metrics CSV       : {csv_path}")
    print("─" * 55)

    stats["elapsed_sec"] = round(elapsed, 2)
    stats["effective_fps"] = round(eff_fps, 2)
    return stats


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _infer_alert_type(counts: dict) -> str:
    if counts.get("weapon", 0) > 0:
        return "weapon_detected"
    if counts.get("person", 0) > 5:
        return "crowd_spike"
    if counts.get("vehicle", 0) > 3:
        return "vehicle_surge"
    if counts.get("total", 0) > 0:
        return "motion_anomaly"
    return "no_detection"


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Border Surveillance AI — End-to-End Pipeline")
    parser.add_argument("--video",   required=True,
                        help="Path to input video file (.mp4 / .avi)")
    parser.add_argument("--config",  default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--azure",   action="store_true",
                        help="Upload frames + alerts to Azure")
    parser.add_argument("--show",    action="store_true",
                        help="Display annotated frames live")
    args = parser.parse_args()

    run_pipeline(
        video_path  = args.video,
        config_path = args.config,
        use_azure   = args.azure,
        show        = args.show,
    )
