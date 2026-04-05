

import os
import glob
import shutil
import random
import yaml
from pathlib import Path

print("Scanning for empty label files...")

TO_ANNOTATE  = "data/to_annotate"
OUTPUT_DIR   = "data/processed/yolo_annotated"
DATA_YAML    = "data/border_data.yaml"
CLASSES      = ["person", "vehicle", "weapon", "suspicious_object"]

# ── Find all pairs where label is NOT empty ───────────────────────────────────
all_imgs = (glob.glob(f"{TO_ANNOTATE}/*.jpg") +
            glob.glob(f"{TO_ANNOTATE}/*.png"))

good_pairs  = []
empty_pairs = []

for img_path in all_imgs:
    lbl_path = str(Path(img_path).with_suffix(".txt"))
    if not os.path.exists(lbl_path):
        continue
    size = os.path.getsize(lbl_path)
    if size > 0:
        good_pairs.append((img_path, lbl_path))
    else:
        empty_pairs.append((img_path, lbl_path))

print(f"  With annotations (good) : {len(good_pairs)}")
print(f"  Empty labels (bad)      : {len(empty_pairs)}")

if len(good_pairs) < 10:
    print("\n[ERROR] Not enough annotated images.")
    print("  Go back to CVAT and annotate more frames.")
    exit(1)

# ── Rebuild output folders ────────────────────────────────────────────────────
for split in ["train", "val", "test"]:
    shutil.rmtree(f"{OUTPUT_DIR}/{split}", ignore_errors=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

# ── Split only good pairs ─────────────────────────────────────────────────────
random.seed(42)
random.shuffle(good_pairs)

n       = len(good_pairs)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

splits = {
    "train": good_pairs[:n_train],
    "val":   good_pairs[n_train: n_train + n_val],
    "test":  good_pairs[n_train + n_val:],
}

print(f"\nRebuilding dataset with ONLY annotated frames:")
total_instances = {"train": 0, "val": 0, "test": 0}

for split_name, pairs in splits.items():
    instances = 0
    for img_path, lbl_path in pairs:
        stem = Path(img_path).stem
        ext  = Path(img_path).suffix
        shutil.copy2(img_path, f"{OUTPUT_DIR}/{split_name}/images/{stem}{ext}")
        shutil.copy2(lbl_path, f"{OUTPUT_DIR}/{split_name}/labels/{stem}.txt")
        # Count instances
        with open(lbl_path) as f:
            instances += len([l for l in f.readlines() if l.strip()])
    total_instances[split_name] = instances
    print(f"  {split_name:<8}: {len(pairs):>3} images  |  {instances:>4} instances  "
          f"| avg {instances/len(pairs):.1f} objects/image")

# ── Update yaml ───────────────────────────────────────────────────────────────
config = {
    "path":  os.path.abspath(OUTPUT_DIR),
    "train": "train/images",
    "val":   "val/images",
    "test":  "test/images",
    "nc":    4,
    "names": {i: c for i, c in enumerate(CLASSES)},
}
with open(DATA_YAML, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"\n✅ Dataset rebuilt — only annotated frames kept")
print(f"✅ border_data.yaml updated")
print(f"\nNow restart training:")
print(f"  model.train(data='data/border_data.yaml', epochs=50, ...)")