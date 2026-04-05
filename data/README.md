# 📦 Dataset Download Instructions

Large datasets are NOT committed to this repo. Follow the steps below to download each dataset and place it in the correct folder.

---

## 1. UCF-Crime Dataset  *(primary — anomaly detection)*

**Size:** ~128 hours of CCTV footage · 1,900 videos · 13 anomaly categories

### Method A — Kaggle (fastest, recommended)
```bash
pip install kaggle

# Place kaggle.json in ~/.kaggle/
# Download from: https://www.kaggle.com/settings -> API -> Create New Token
kaggle datasets download -d odins0n/ucf-crime-dataset
unzip ucf-crime-dataset.zip -d data/raw/ucf_crime/
```

### Method B — Official Request Form
1. Go to https://crcv.ucf.edu/projects/real-world/
2. Scroll to **"Download"** section
3. Fill Google Form: name, institution, purpose ("GTU Internship 2026 - Border Surveillance AI")
4. You'll receive a Google Drive download link by email (1–3 days)

### Method C — Google Colab (if using cloud training)
```python
from google.colab import drive
drive.mount('/content/drive')
# Upload dataset directly to your Drive and reference from there
```

**Expected folder structure after download:**
```
data/raw/ucf_crime/
├── Anomaly/
│   ├── Assault/          (~50 .mp4 videos)
│   ├── Fighting/
│   ├── Robbery/
│   ├── Explosion/
│   ├── Stealing/
│   └── ...               (8 more anomaly categories)
└── Normal_Videos/         (~950 normal surveillance videos)
```

**For this project — start with 50 videos:**
```bash
# Run the organizer script to copy a small working subset:
python src/dataset_download.py
```

---

## 2. xView Satellite Dataset  *(aerial vehicle detection)*

**Size:** ~7.5 GB · 1M+ objects · 0.3m resolution

```
1. Go to: http://xviewdataset.org
2. Click "Download Dataset"
3. Register with your email -> confirm -> log in
4. Download val_images.tgz (~500 MB) — enough for this project
5. Download train_labels.zip (~15 MB)
```

```bash
tar -xzf val_images.tgz  -C data/raw/xview/
unzip train_labels.zip   -C data/raw/xview/
```

> **Note:** If xView download is slow, skip it initially. YOLOv8 with COCO pre-training already detects vehicles. xView is only needed for satellite/aerial fine-tuning.

---

## 3. DOTA v2  *(aerial object detection)*

**Size:** ~280K annotated objects in aerial images

```
1. Go to: https://captain-whu.github.io/DOTA/index.html
2. Click "Download" -> register for access
3. Download "DOTA-v2.0 val set" (~2 GB) for testing
```

```bash
unzip DOTA_v2_val.zip -C data/raw/dota/
```

---

## 4. COCO (pre-training — already included with YOLOv8)

YOLOv8 uses COCO weights by default (`yolov8n.pt`). No manual download needed.

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')   # Downloads COCO-pretrained weights automatically
```

---

CVAT us used for dataannotations
# Saves to: data/annotations/border-surveillance-1/
```

---

## Storage Summary

| Dataset      | Raw Size | Folder                  | Required? |
|--------------|----------|-------------------------|-----------|
| UCF-Crime    | ~60 GB   | `data/raw/ucf_crime/`   | ✅ Yes    |
| xView        | ~7.5 GB  | `data/raw/xview/`       | Optional  |
| DOTA v2      | ~20 GB   | `data/raw/dota/`        | Optional  |
| COCO weights | ~6 MB    | Auto-downloaded by YOLO | ✅ Auto   |
| Roboflow     | ~50 MB   | `data/annotations/`     | ✅ Yes    |

> ⚠️ All `data/raw/` folders are in `.gitignore` — do NOT commit raw datasets to GitHub.
