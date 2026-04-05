# AI-Based Border Defence and Surveillance System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://docs.ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Real-time border surveillance system using YOLOv8 object detection and ensemble anomaly detection to automatically identify threats and generate prioritized alerts.

---

## Overview

An end-to-end AI-powered surveillance pipeline for border security. It detects persons, vehicles, weapons and suspicious objects in real-time using a fine-tuned YOLOv8n model, scores each frame for anomalous activity using an ensemble of Isolation Forest and Random Forest, and generates prioritized alerts logged to a local SQLite database. A Streamlit dashboard provides live monitoring and visualization.

---

## Results

| Metric | Value |
|--------|-------|
| YOLOv8 mAP@50 | 0.290+ |
| Suspicious Object mAP | 0.545 |
| Person mAP | 0.254 |
| Anomaly Detection Accuracy | 0.993 |
| ROC-AUC | 1.00 |
| Processing Speed (CPU) | 1.5 FPS |
| Training Images | 126 |
| Custom Annotated Frames | 48 |

---

## Repository Structure

```
Border-Surveillance-Project/
│
├── 📁 data/
│   ├── processed/                      # Cleaned & preprocessed frames
│   ├── annotations/                    # CVAT annotation exports
│   ├── to_annotate/                    # Frames selected for annotation
│   └── raw/                            # Raw datasets (ucf_crime, dota, xview)
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb          # Data preprocessing pipeline
│   ├── 03_Object_Detection.ipynb       # YOLOv8 training & evaluation
│   ├── 04_Anomaly_Detection.ipynb      # Ensemble model training
│   └── 05_End_to_End_Demo.ipynb        # Full pipeline demo
│
├── 📁 src/
│   ├── preprocessing.py                # Frame extraction & normalization
│   ├── detect_objects.py               # YOLOv8 inference wrapper
│   ├── anomaly_detector.py             # Ensemble anomaly scoring
│   ├── alert_manager.py                # Alert prioritization & logging
│   ├── azure_uploader.py               # Azure Blob/CosmosDB integration
│   ├── run_pipeline.py                 # Main end-to-end script
│   ├── pick_frames.py                  # Frame selection for annotation
│   ├── merge_annotations.py            # Merge CVAT exports
│   ├── fix_empty_labels.py             # Remove empty label files
│   └── prepare_annotated_dataset.py    # Build final YOLO dataset
│
├── 📁 dashboard/
│   └── streamlit_app.py                # Streamlit web dashboard
│
├── 📁 models/
│   ├── yolov8_border.pt                # Trained YOLOv8 weights
│   ├── anomaly_ensemble.pkl            # Anomaly detection model
│   ├── isolation_forest.pkl            # Isolation Forest model
│   ├── random_forest.pkl               # Random Forest model
│   └── scaler.pkl                      # Feature scaler
│
├── 📁 results/
│   ├── metrics/                        # Evaluation CSVs
│   ├── screenshots/                    # Demo screenshots
│   └── charts/                         # Performance visualizations
│
├── 📁 tests/
│   ├── test_detection.py
│   ├── test_anomaly.py
│   └── test_azure_upload.py
│
├── 📁 docs/
│   ├── project_report.pdf
│   └── presentation.pptx
│
├── .github/workflows/ci_cd.yml         # GitHub Actions pipeline
├── config.example.yaml                 # Template config
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container setup
├── docker-compose.yml                  # Multi-service setup
├── .gitignore
└── README.md
```

---

## Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| UCF-Crime | [Kaggle](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) | Surveillance anomaly videos — 111,308 frames |
| DOTA | [captain-whu.github.io](https://captain-whu.github.io/DOTA/dataset.html) | Aerial object detection — 87 chips |
| xView | [xviewdataset.org](https://xviewdataset.org) | Satellite imagery |
| Custom | CVAT annotated | 48 border surveillance frames, 4 classes |

**Detection Classes:** `person` · `vehicle` · `weapon` · `suspicious_object`

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/jainilgupta02/Border-Surveillance-Project
cd Border-Surveillance-Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Download Dataset
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d odins0n/ucf-crime-dataset
unzip ucf-crime-dataset.zip -d data/raw/ucf_crime/
```

### 2. Preprocess
```bash
python src/preprocessing.py
```

### 3. Annotation Workflow
```bash
# Pick frames for annotation
python src/pick_frames.py

# Annotate in CVAT → export as YOLO 1.1 → place in data/to_annotate/

# Merge all CVAT exports
python src/merge_annotations.py

# Remove empty labels
python src/fix_empty_labels.py

# Build final dataset
python src/prepare_annotated_dataset.py
```

### 4. Train YOLOv8
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data     = 'data/border_data.yaml',
    epochs   = 30,
    imgsz    = 640,
    batch    = 8,
    device   = 'cpu',    # use 0 for GPU
    workers  = 0,
)
```

### 5. Run Full Pipeline
```bash
python src/run_pipeline.py --video path/to/video.mp4 --show
```

### 6. Launch Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

### 7. Run with Docker
```bash
docker-compose up
```

---

## Pipeline Architecture

```
Raw Video / Frames
        ↓
Preprocessing (OpenCV)
640x640 resize + normalize
        ↓
YOLOv8n Detection
person / vehicle / weapon / suspicious_object
        ↓
Feature Extraction
920 feature vectors per frame
        ↓
Anomaly Ensemble
Isolation Forest + Random Forest → score 0 to 1
        ↓
Alert Manager
HIGH (>0.7) / MEDIUM (0.4-0.7) / LOW (<0.4)
        ↓
SQLite Database + Streamlit Dashboard
```

---

## Run Notebooks in Order

```
01_EDA.ipynb               → Dataset exploration and visualization
02_Preprocessing.ipynb     → Frame extraction and dataset preparation
03_Object_Detection.ipynb  → YOLOv8 training and evaluation
04_Anomaly_Detection.ipynb → Ensemble anomaly model training
05_End_to_End_Demo.ipynb   → Full pipeline demonstration
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Object Detection | Ultralytics YOLOv8n |
| Anomaly Detection | Scikit-learn (Isolation Forest + Random Forest) |
| Computer Vision | OpenCV |
| Dashboard | Streamlit |
| Database | SQLite |
| Cloud (Designed) | Azure Blob Storage + Cosmos DB |
| Annotation | CVAT |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Language | Python 3.11, PyTorch 2.1 |

---

## Configuration

Copy `config.example.yaml` to `config.yaml`:

```yaml
model:
  weights: models/yolov8_border.pt
  confidence: 0.25
  iou: 0.45

anomaly:
  threshold: 0.5
  contamination: 0.10

alerts:
  db_path: alerts.db

azure:
  connection_string: YOUR_CONNECTION_STRING
  container_name: surveillance-frames
  cosmos_endpoint: YOUR_COSMOS_ENDPOINT
```

> Azure integration is fully coded in `src/azure_uploader.py`. Local file system is used by default when credentials are not configured.

---

## References

| # | Reference |
|---|-----------|
| 1 | UCF-Crime Dataset — https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset |
| 2 | DOTA Dataset — https://captain-whu.github.io/DOTA/dataset.html |
| 3 | xView Dataset — https://xviewdataset.org |
| 4 | Ultralytics YOLOv8 — https://docs.ultralytics.com |
| 5 | Scikit-learn — https://scikit-learn.org |
| 6 | OpenCV — https://opencv.org |
| 7 | Streamlit — https://docs.streamlit.io |
| 8 | CVAT — https://cvat.ai |
| 9 | Docker — https://docs.docker.com |
| 10 | GitHub Actions — https://docs.github.com/actions |

---

## License

MIT License — see [LICENSE](LICENSE) for details.