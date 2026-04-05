"""
pick_frames.py
--------------
Pick 100 frames for annotation.
Searches MULTIPLE locations automatically — works whether you have
downloaded the dataset or run preprocessing or neither.

Run:
    python src/pick_frames.py
"""

import os
import glob
import shutil
import random
from pathlib import Path

# ── Output folder ─────────────────────────────────────────────────────────────
TO_ANNOTATE_DIR = "data/to_annotate"
os.makedirs(TO_ANNOTATE_DIR, exist_ok=True)

# ── Search in ALL possible locations — tries each one in order ────────────────
SEARCH_DIRS = [
    "data/processed/train",           # after running 02_Preprocessing.ipynb
    "data/processed",                 # alternate processed location
    "data/raw/ucf_crime/train",       # Kaggle UCF structure: ucf_crime/train/Category/*.png
    "data/raw/ucf_crime",             # flat Kaggle structure: ucf_crime/Category/*.png
    "data/raw",                       # anywhere inside raw folder
]

print("Searching for frames in all known locations...")
print("=" * 55)

all_frames = []
found_in   = None

for search_dir in SEARCH_DIRS:
    if not os.path.exists(search_dir):
        print(f"  ✗  {search_dir:<45} (folder does not exist)")
        continue

    pngs  = glob.glob(f"{search_dir}/**/*.png", recursive=True)
    jpgs  = glob.glob(f"{search_dir}/**/*.jpg", recursive=True)
    found = pngs + jpgs

    if found:
        print(f"  ✓  {search_dir:<45} → {len(found)} frames  (PNG={len(pngs)}  JPG={len(jpgs)})")
        all_frames = found
        found_in   = search_dir
        break
    else:
        print(f"  ✗  {search_dir:<45} (folder exists but has no PNG/JPG)")

print("=" * 55)

# ── If still nothing found — show exactly what IS in data/ ───────────────────
if not all_frames:
    print("\n[ERROR] No PNG or JPG frames found anywhere.\n")
    print("Your data/ folder currently looks like this:")
    print("-" * 40)
    if os.path.exists("data"):
        for root, dirs, files in os.walk("data"):
            level = root.replace("data", "").count(os.sep)
            if level > 4:
                continue
            indent     = "  " * level
            sub_indent = "  " * (level + 1)
            print(f"{indent}{os.path.basename(root)}/")
            img_files = [f for f in files if f.endswith((".png",".jpg",".jpeg",".mp4"))]
            for f in img_files[:3]:
                print(f"{sub_indent}{f}")
            if len(img_files) > 3:
                print(f"{sub_indent}... and {len(img_files)-3} more")
    else:
        print("  data/ folder does not exist at all!")
    print("-" * 40)
    print("\nYou need to download UCF-Crime dataset first.")
    print("\nFASTEST WAY (Kaggle):")
    print("  pip install kaggle")
    print("  # Get kaggle.json from kaggle.com → Settings → API → Create Token")
    print("  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
    print("  kaggle datasets download -d odins0n/ucf-crime-dataset")
    print("  unzip ucf-crime-dataset.zip -d data/raw/ucf_crime/")
    print()
    print("If your dataset is somewhere else, open src/pick_frames.py")
    print("and add your folder path to the SEARCH_DIRS list at the top.")
    exit(1)

# ── Detect which extension is more common ────────────────────────────────────
png_count = len([f for f in all_frames if f.lower().endswith(".png")])
jpg_count = len([f for f in all_frames if f.lower().endswith(".jpg")])
EXT       = ".png" if png_count >= jpg_count else ".jpg"

# Filter to just the dominant extension
all_frames = [f for f in all_frames if f.lower().endswith(EXT)]

print(f"\nFormat detected : {EXT}  (PNG={png_count}  JPG={jpg_count})")
print(f"Found in        : {found_in}")
print(f"Total frames    : {len(all_frames)}")

# ── Group frames by their category folder ────────────────────────────────────
by_category = {}
for fp in all_frames:
    cat = os.path.basename(os.path.dirname(fp))
    by_category.setdefault(cat, []).append(fp)

print(f"\nCategories found: {len(by_category)}")
print("-" * 55)

selected = []
for cat, frames in sorted(by_category.items()):
    random.seed(42)
    n_pick = min(8, len(frames))
    picked = random.sample(frames, n_pick)
    selected.extend(picked)
    print(f"  {cat:<30} : {n_pick:>3} selected  ({len(frames)} available)")

# Limit to 100 total
selected = selected[:100]
print("-" * 55)
print(f"  Total selected  : {len(selected)}")

# ── Copy to annotation folder ─────────────────────────────────────────────────
print(f"\nCopying {len(selected)} frames to {TO_ANNOTATE_DIR}/ ...")
for i, fp in enumerate(selected):
    dst = os.path.join(TO_ANNOTATE_DIR, f"frame_{i:04d}{EXT}")
    shutil.copy2(fp, dst)

copied = len(glob.glob(f"{TO_ANNOTATE_DIR}/*{EXT}"))
print(f"✅ {copied} frames ready in {TO_ANNOTATE_DIR}/")

# ── Create classes.txt for LabelImg ──────────────────────────────────────────
os.makedirs("data/annotations", exist_ok=True)
classes_txt = "data/annotations/classes.txt"
with open(classes_txt, "w") as f:
    f.write("person\nvehicle\nweapon\nsuspicious_object\n")
print(f"✅ classes.txt created → {classes_txt}")

# ── Next steps ────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  NEXT STEPS")
print("=" * 55)
print()
print("STEP 1 — Open LabelImg:")
print(f"   labelImg {TO_ANNOTATE_DIR} {classes_txt}")
print()
print("STEP 2 — Settings inside LabelImg (do this FIRST):")
print("   • Left sidebar: click 'PascalVOC' → switch to 'YOLO'")
print("   • View menu → tick 'Auto Save Mode'")
print()
print("STEP 3 — Annotate each frame:")
print("   W key  → draw bounding box")
print("   D key  → next image")
print("   A key  → previous image")
print("   Del    → delete selected box")
print()
print("STEP 4 — After annotating, run:")
print("   python src/prepare_annotated_dataset.py")
print()