"""
prepare_annotated_dataset.py
-----------------------------
After annotating frames with LabelImg, this script:
  1. Finds all annotated PNG/JPG + .txt label pairs
  2. Validates every label file
  3. Splits into train / val / test  (70 / 15 / 15)
  4. Copies everything into data/processed/yolo_annotated/
  5. Creates border_data.yaml for YOLOv8 training

Run:
    python src/prepare_annotated_dataset.py
"""

import os
import glob
import shutil
import random
import yaml
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
TO_ANNOTATE_DIR = "data/to_annotate"          # your annotated frames + .txt files
OUTPUT_DIR      = "data/processed/yolo_annotated"
DATA_YAML_PATH  = "data/border_data.yaml"

CLASSES = ["person", "vehicle", "weapon", "suspicious_object"]
SPLIT   = (0.70, 0.15, 0.15)    # train / val / test


# ── Create output folders ─────────────────────────────────────────────────────
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)


# ── Find all annotated image + label pairs ────────────────────────────────────
# LabelImg saves .txt next to each image regardless of image extension
# Kaggle UCF-Crime = PNG,  preprocessed frames may be PNG or JPG

png_files = glob.glob(f"{TO_ANNOTATE_DIR}/*.png")
jpg_files = glob.glob(f"{TO_ANNOTATE_DIR}/*.jpg")
all_image_files = sorted(png_files + jpg_files)

print(f"Images found in {TO_ANNOTATE_DIR}/:")
print(f"  PNG: {len(png_files)}")
print(f"  JPG: {len(jpg_files)}")
print(f"  Total: {len(all_image_files)}")

if not all_image_files:
    print("\n[ERROR] No images found.")
    print("  1. Run pick_frames.py first")
    print("  2. Then annotate in LabelImg")
    exit(1)

pairs          = []
missing_labels = []

for img_path in all_image_files:
    lbl_path = str(Path(img_path).with_suffix(".txt"))
    if os.path.exists(lbl_path):
        pairs.append((img_path, lbl_path))
    else:
        missing_labels.append(img_path)

print(f"\nAnnotated pairs (image + label): {len(pairs)}")
if missing_labels:
    print(f"\n[WARN] {len(missing_labels)} images have NO .txt label file yet:")
    for f in missing_labels[:5]:
        print(f"  {os.path.basename(f)}")
    if len(missing_labels) > 5:
        print(f"  ... and {len(missing_labels)-5} more")
    print("\n  Open LabelImg and annotate these before continuing.")


# ── Validate every label file ─────────────────────────────────────────────────
print("\nValidating label files...")

valid_pairs   = []
invalid_pairs = []

for img_path, lbl_path in pairs:
    valid = True
    errors = []

    with open(lbl_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Empty .txt = image with no objects (background) — perfectly valid
    if not lines:
        valid_pairs.append((img_path, lbl_path))
        continue

    for line_num, line in enumerate(lines, 1):
        parts = line.split()

        # Must have exactly 5 values: class cx cy w h
        if len(parts) != 5:
            errors.append(f"  Line {line_num}: needs 5 values, got {len(parts)}: '{line}'")
            valid = False
            continue

        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Class ID must be 0, 1, 2 or 3
            if cls_id < 0 or cls_id >= len(CLASSES):
                errors.append(f"  Line {line_num}: class_id={cls_id} invalid (valid: 0-{len(CLASSES)-1})")
                valid = False

            # All values must be between 0 and 1
            for name, val in [("cx",cx),("cy",cy),("w",w),("h",h)]:
                if not (0.0 <= val <= 1.0):
                    errors.append(f"  Line {line_num}: {name}={val} out of range [0,1]")
                    valid = False

        except ValueError as e:
            errors.append(f"  Line {line_num}: cannot parse numbers — {e}")
            valid = False

    if valid:
        valid_pairs.append((img_path, lbl_path))
    else:
        invalid_pairs.append((img_path, lbl_path))
        print(f"\n  ❌ {os.path.basename(lbl_path)}")
        for err in errors:
            print(err)

print(f"\n  ✅ Valid   : {len(valid_pairs)}")
print(f"  ❌ Invalid : {len(invalid_pairs)}  ← fix these in LabelImg")

if len(valid_pairs) < 10:
    print("\n[WARN] Less than 10 valid annotations — annotate more frames first.")
    exit(1)


# ── Split into train / val / test ─────────────────────────────────────────────
random.seed(42)
random.shuffle(valid_pairs)

n       = len(valid_pairs)
n_train = int(SPLIT[0] * n)
n_val   = int(SPLIT[1] * n)

splits = {
    "train": valid_pairs[:n_train],
    "val":   valid_pairs[n_train: n_train + n_val],
    "test":  valid_pairs[n_train + n_val:],
}

print(f"\nSplitting {n} valid pairs:")


# ── Copy files — keep original extension (PNG or JPG) ─────────────────────────
for split_name, split_pairs in splits.items():
    for img_path, lbl_path in split_pairs:
        stem = Path(img_path).stem
        ext  = Path(img_path).suffix          # .png or .jpg — preserve original

        dst_img = f"{OUTPUT_DIR}/{split_name}/images/{stem}{ext}"
        dst_lbl = f"{OUTPUT_DIR}/{split_name}/labels/{stem}.txt"

        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)

    print(f"  {split_name:<8}: {len(split_pairs):>3} images")


# ── Count total objects per class across all annotations ──────────────────────
class_counts = {name: 0 for name in CLASSES}
for _, lbl_path in valid_pairs:
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                if 0 <= cls_id < len(CLASSES):
                    class_counts[CLASSES[cls_id]] += 1

print("\nAnnotated object counts:")
for cls_name, count in class_counts.items():
    bar = "█" * min(count, 40)
    print(f"  {cls_name:<20}: {count:>4}  {bar}")


# ── Create border_data.yaml ───────────────────────────────────────────────────
yolo_config = {
    "path":  os.path.abspath(OUTPUT_DIR),
    "train": "train/images",
    "val":   "val/images",
    "test":  "test/images",
    "nc":    len(CLASSES),
    "names": {i: name for i, name in enumerate(CLASSES)},
}

with open(DATA_YAML_PATH, "w") as f:
    yaml.dump(yolo_config, f, default_flow_style=False, sort_keys=False)

print(f"\n✅ border_data.yaml saved → {DATA_YAML_PATH}")
print(f"✅ Dataset ready at       → {OUTPUT_DIR}/")
print(f"\nFinal structure:")
for split in ["train", "val", "test"]:
    imgs = len(glob.glob(f"{OUTPUT_DIR}/{split}/images/*"))
    lbls = len(glob.glob(f"{OUTPUT_DIR}/{split}/labels/*.txt"))
    print(f"  {split}/images/  {imgs} files")
    print(f"  {split}/labels/  {lbls} files")

print("\n✅ Done! Now open 03_Object_Detection.ipynb and run training.")
