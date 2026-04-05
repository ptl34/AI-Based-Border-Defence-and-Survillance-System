"""
merge_annotations.py
--------------------
Merges all CVAT export label files from subfolders (n, nm, oop, w, p, etc.)
into data/to_annotate/ root, matching them to the existing images by filename stem.

Handles nested paths like:
  p/obj_Train_data/to_annotate/frame_0000.txt
  n/obj_Train_data/frame_0000.txt
  nm/frame_0000.txt

Run from project root:
    python src/merge_annotations.py
"""

import os
import glob
import shutil
from pathlib import Path

TO_ANN = "data/to_annotate"

print("=" * 60)
print("  MERGE ANNOTATIONS")
print("=" * 60)

# ── Step 1: Find ALL root images (frame_XXXX.jpg or .png) ─────────────────────
root_imgs = (
    glob.glob(f"{TO_ANN}/*.jpg") +
    glob.glob(f"{TO_ANN}/*.png")
)
root_imgs = [f for f in root_imgs if "classes" not in f.lower()]

# Build lookup: stem (e.g. "frame_0000") -> full image path
img_lookup = {}
for img in root_imgs:
    stem = Path(img).stem          # e.g. "frame_0000"
    img_lookup[stem] = img

print(f"\nRoot images found : {len(img_lookup)}")

# ── Step 2: Find ALL label .txt files anywhere inside subfolders ───────────────
# Exclude obj.names, obj.data, Train.txt, classes.txt
EXCLUDE_NAMES = {"obj.names", "obj.data", "train.txt", "classes.txt",
                 "obj.names\n", "obj_train.txt"}

all_subfolder_labels = []
subfolders = [
    d for d in os.listdir(TO_ANN)
    if os.path.isdir(os.path.join(TO_ANN, d))
]

print(f"Subfolders found  : {subfolders}")
print()

for sf in sorted(subfolders):
    sf_path = os.path.join(TO_ANN, sf)
    txts = glob.glob(f"{sf_path}/**/*.txt", recursive=True)
    valid = [
        t for t in txts
        if Path(t).name.lower() not in EXCLUDE_NAMES
        and os.path.getsize(t) > 0
    ]
    print(f"  {sf:<10} : {len(valid)} non-empty label files")
    all_subfolder_labels.extend(valid)

print(f"\nTotal subfolder labels (non-empty): {len(all_subfolder_labels)}")

# ── Step 3: Normalize stems for matching ──────────────────────────────────────
# CVAT sometimes exports with 5-digit padding (frame_00000) vs 4-digit (frame_0000)
# We normalize both to the base number for matching

def normalize_stem(stem):
    """
    frame_00000 -> frame_0000  (5-digit to 4-digit)
    frame_0000  -> frame_0000  (unchanged)
    frame_00013 -> frame_0013  (5-digit to 4-digit)
    """
    if stem.startswith("frame_"):
        num_part = stem[6:]          # everything after "frame_"
        try:
            num = int(num_part)
            return f"frame_{num:04d}"   # normalize to 4-digit
        except ValueError:
            pass
    return stem

# Build normalized lookup for root images
img_lookup_norm = {}
for stem, path in img_lookup.items():
    img_lookup_norm[normalize_stem(stem)] = path

print(f"\nMatching labels to root images...")
print("-" * 60)

fixed   = 0
skipped = 0
no_match = []

for lbl_path in all_subfolder_labels:
    raw_stem  = Path(lbl_path).stem
    norm_stem = normalize_stem(raw_stem)

    # Try exact match first, then normalized
    target_img = img_lookup.get(raw_stem) or img_lookup_norm.get(norm_stem)

    if target_img is None:
        no_match.append(raw_stem)
        skipped += 1
        continue

    # Destination label path = same folder as image, same stem, .txt extension
    dst_lbl = str(Path(target_img).with_suffix(".txt"))

    # Only overwrite if destination is empty or doesn't exist
    if not os.path.exists(dst_lbl) or os.path.getsize(dst_lbl) == 0:
        shutil.copy2(lbl_path, dst_lbl)
        print(f"  FIXED : {Path(target_img).name}  ← {Path(lbl_path).parent.parent.name}/{Path(lbl_path).name}")
        fixed += 1
    else:
        # Destination already has content — merge by appending unique lines
        with open(dst_lbl) as f:
            existing_lines = set(l.strip() for l in f if l.strip())
        with open(lbl_path) as f:
            new_lines = [l.strip() for l in f if l.strip()]
        added = 0
        with open(dst_lbl, "a") as f:
            for line in new_lines:
                if line not in existing_lines:
                    f.write(line + "\n")
                    added += 1
        if added:
            print(f"  MERGE : {Path(target_img).name}  +{added} new annotations")
            fixed += 1

# ── Step 4: Final summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  Fixed / merged : {fixed}")
print(f"  No image match : {skipped}")
if no_match:
    print(f"  Unmatched stems: {no_match[:5]}{'...' if len(no_match)>5 else ''}")

# Count results
all_root_txts = [
    f for f in glob.glob(f"{TO_ANN}/*.txt")
    if Path(f).name.lower() not in EXCLUDE_NAMES
]
non_empty = [t for t in all_root_txts if os.path.getsize(t) > 0]
empty     = [t for t in all_root_txts if os.path.getsize(t) == 0]

print(f"\nFinal state in {TO_ANN}/")
print(f"  Images          : {len(img_lookup)}")
print(f"  Non-empty labels: {len(non_empty)}  <- annotated")
print(f"  Empty labels    : {len(empty)}    <- no annotations")

print("""
Next steps:
  python src/fix_empty_labels.py
  python src/prepare_annotated_dataset.py
""")
