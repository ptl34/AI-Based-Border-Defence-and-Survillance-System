# 🤖 Model Card — Border Surveillance AI

---

## 1. YOLOv8 Object Detection Model

| Field             | Detail                                          |
|-------------------|-------------------------------------------------|
| **Model Name**    | yolov8_border.pt                                |
| **Base Model**    | YOLOv8n (Ultralytics, COCO pre-trained)         |
| **Fine-tuned on** | Roboflow custom border surveillance dataset     |
| **Input Size**    | 640 × 640 px, BGR uint8                         |
| **Output**        | Bounding boxes + class IDs + confidence scores  |
| **Classes**       | person (0), vehicle (1), weapon (2), suspicious_object (3) |
| **Epochs**        | 50                                              |
| **Batch Size**    | 16                                              |
| **Optimizer**     | SGD (default Ultralytics)                       |
| **Hardware**      | Google Colab T4 GPU                             |

### Performance Metrics

| Metric        | Value  | Target |
|---------------|--------|--------|
| mAP\@50       | _TBD_  | ≥ 88 % |
| mAP\@50–95    | _TBD_  | ≥ 60 % |
| Precision     | _TBD_  | ≥ 85 % |
| Recall        | _TBD_  | ≥ 85 % |
| Inference FPS | _TBD_  | ≥ 20   |

> Fill in _TBD_ values after training in `notebooks/03_Object_Detection.ipynb`.

### Training Data

- **Source:** 100 manually annotated frames from UCF-Crime and custom border footage
- **Annotation tool:** Roboflow
- **Augmentations applied:** horizontal flip, rotation ±15°, brightness ±30%, mosaic

### Intended Use

Detection of persons, vehicles, weapons, and suspicious objects in CCTV / surveillance video feeds at border checkpoints. **Not** intended for facial recognition or identification of individuals.

### Limitations

- Performance may degrade significantly in low-light or night-vision footage without IR pre-processing
- Small objects (<32 × 32 px) at long range may be missed
- Model has not been evaluated on satellite or aerial imagery (use xView-trained variant for that)

---

## 2. Anomaly Detection Ensemble

| Field             | Detail                                             |
|-------------------|----------------------------------------------------|
| **Model Name**    | isolation_forest.pkl + random_forest.pkl           |
| **Architecture**  | Ensemble: Isolation Forest (60%) + Random Forest (40%) |
| **Feature dim**   | 12 (see `anomaly_detector.py` — FEATURE_NAMES)     |
| **Training data** | Normal surveillance frames (unsupervised IF) + labelled UCF-Crime frames (supervised RF) |
| **Contamination** | 0.10 (Isolation Forest)                            |
| **n_estimators**  | 200 (both models)                                  |

### Feature Vector (12 dimensions)

| Index | Feature Name     | Description                               |
|-------|-----------------|-------------------------------------------|
| 0     | num_persons     | Count of persons detected by YOLO         |
| 1     | num_vehicles    | Count of vehicles detected                |
| 2     | num_weapons     | Count of weapons detected                 |
| 3     | total_objects   | Total detected objects                    |
| 4     | avg_confidence  | Mean YOLO confidence score                |
| 5     | max_confidence  | Max YOLO confidence score                 |
| 6     | brightness_mean | Mean pixel brightness (grayscale)         |
| 7     | brightness_var  | Variance of pixel brightness              |
| 8     | mean_flow       | Mean optical flow magnitude               |
| 9     | std_flow        | Std deviation of optical flow magnitude   |
| 10    | weapon_flag     | 1 if any weapon detected, else 0          |
| 11    | crowd_flag      | 1 if >5 persons in frame, else 0          |

### Performance Metrics

| Metric              | Value  | Target  |
|---------------------|--------|---------|
| Detection Accuracy  | _TBD_  | ≥ 85 %  |
| False Positive Rate | _TBD_  | ≤ 20 %  |
| ROC-AUC             | _TBD_  | ≥ 0.85  |

> Fill in _TBD_ values after training in `notebooks/04_Anomaly_Detection.ipynb`.

### Alert Priority Thresholds

| Threshold | Priority | Action                      |
|-----------|----------|-----------------------------|
| score > 0.70 | 🔴 HIGH   | Email + SMS + Cosmos DB log |
| score > 0.40 | 🟡 MEDIUM | Cosmos DB log only          |
| score ≤ 0.40 | 🟢 LOW    | Filtered out (not logged)   |

---

## 3. Ethical Considerations

- This system is designed as a **decision-support tool** — all HIGH alerts should be reviewed by a human operator before any action is taken
- The system does not perform facial recognition or biometric identification
- Detected individuals are not stored in any identity database
- False positive and false negative rates must be monitored continuously in deployment

---

## 4. Files

```
models/
├── yolov8_border.pt         # Fine-tuned YOLOv8 weights (Git LFS)
├── isolation_forest.pkl     # Trained Isolation Forest
├── random_forest.pkl        # Trained Random Forest classifier
├── scaler.pkl               # StandardScaler fitted on training data
└── model_card.md            # This file
```
