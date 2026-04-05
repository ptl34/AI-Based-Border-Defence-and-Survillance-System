"""
merge_annotations.py
--------------------
Merges ALL CVAT export subfolders into one clean data/to_annotate/ folder.

Handles this exact structure:
    data/to_annotate/
    ├── n/obj_train_data/   ← labels only (no images)
    ├── nm/obj_train_data/  ← labels only (no images)
    ├── oop/obj_train_data/ ← labels only (no images)
    ├── w/obj_train_data/   ← weapon labels only (no images)
    ├── frame_0000.jpg      ← root images
    ├── frame_0000.txt      ← root labels
    ...

Run:
    python src/merge_annotations.py
"""

import os
import glob
import shutil
from pathlib import Path

ROOT = "data/to_annotate"
TEMP = "data/to_annotate_merged_temp"
os.makedirs(TEMP, exist_ok=True)

SKIP_FILES = {"obj.names", "obj.data", "Train.txt", "train.txt",
              "classes.txt", "notes.json"}

print("=" * 60)
print("  Merging ALL annotation subfolders")
print("=" * 60)

# ── Step 1: Get all ROOT-level images (frame_XXXX.jpg) ───────────────────────
root_images = (glob.glob(f"{ROOT}/*.jpg") +
               glob.glob(f"{ROOT}/*.png"))
print(f"\nRoot images found : {len(root_images)}")

# Build lookup: stem → image path
root_img_lookup = {Path(img).stem: img for img in root_images}

# ── Step 2: Get all ROOT-level labels ────────────────────────────────────────
root_labels = [f for f in glob.glob(f"{ROOT}/*.txt")
               if Path(f).name not in SKIP_FILES]
root_lbl_lookup = {Path(lbl).stem: lbl for lbl in root_labels}
print(f"Root labels found : {len(root_labels)}")

# ── Step 3: Get all subfolder labels (n/nm/oop/w) ────────────────────────────
subfolders = ['n', 'nm', 'oop', 'w']
subfolder_labels = {}  # stem → best label path

for sf in subfolders:
    sf_path = os.path.join(ROOT, sf)
    if not os.path.exists(sf_path):
        continue
    txts = [f for f in glob.glob(f"{sf_path}/**/*.txt", recursive=True)
            if Path(f).name not in SKIP_FILES]
    non_empty = [t for t in txts if os.path.getsize(t) > 0]
    print(f"  {sf}/  → {len(non_empty)} non-empty labels")
    for lp in txts:
        stem = Path(lp).stem
        size = os.path.getsize(lp)
        # Keep non-empty label, prefer largest
        if stem not in subfolder_labels:
            subfolder_labels[stem] = lp
        else:
            if size > os.path.getsize(subfolder_labels[stem]):
                subfolder_labels[stem] = lp

print(f"\nSubfolder unique label stems: {len(subfolder_labels)}")

# ── Step 4: Build final best label for each image stem ───────────────────────
# Priority: subfolder non-empty > root non-empty > root empty > nothing
best_labels = {}

# Start with root labels
for stem, lbl in root_lbl_lookup.items():
    best_labels[stem] = lbl

# Override with subfolder labels if they are non-empty
for stem, lbl in subfolder_labels.items():
    if os.path.getsize(lbl) > 0:
        best_labels[stem] = lbl  # subfolder non-empty wins

print(f"Total stems with labels: {len(best_labels)}")

# ── Step 5: Build matched pairs ───────────────────────────────────────────────
matched = []
no_label = []

for stem, img_path in root_img_lookup.items():
    if stem in best_labels:
        matched.append((img_path, best_labels[stem]))
    else:
        no_label.append(img_path)

print(f"\nMatched pairs   : {len(matched)}")
print(f"No label found  : {len(no_label)}")

# Show how many are non-empty
non_empty_pairs = [(i, l) for i, l in matched if os.path.getsize(l) > 0]
empty_pairs     = [(i, l) for i, l in matched if os.path.getsize(l) == 0]
print(f"  Non-empty (annotated) : {len(non_empty_pairs)}")
print(f"  Empty (no objects)    : {len(empty_pairs)}")

# ── Step 6: Copy to temp ──────────────────────────────────────────────────────
print(f"\nCopying {len(matched)} pairs to temp...")
for i, (img_path, lbl_path) in enumerate(matched):
    ext     = Path(img_path).suffix.lower()
    name    = f"frame_{i:04d}"
    shutil.copy2(img_path, os.path.join(TEMP, f"{name}{ext}"))
    shutil.copy2(lbl_path, os.path.join(TEMP, f"{name}.txt"))

# ── Step 7: Copy classes.txt ──────────────────────────────────────────────────
obj_names = glob.glob(f"{ROOT}/**/obj.names", recursive=True)
os.makedirs("data/annotations", exist_ok=True)
if obj_names:
    shutil.copy2(obj_names[0], "data/annotations/classes.txt")
    print(f"\n  classes.txt from: {obj_names[0]}")
    with open("data/annotations/classes.txt") as f:
        for i, line in enumerate(f.readlines()):
            print(f"    {i}: {line.strip()}")
else:
    with open("data/annotations/classes.txt", "w") as f:
        f.write("person\nvehicle\nweapon\nsuspicious_object\n")
    print("\n  Default classes.txt created")

# ── Step 8: Replace ROOT ──────────────────────────────────────────────────────
print(f"\nReplacing {ROOT}/...")
shutil.rmtree(ROOT)
os.makedirs(ROOT)
for f in os.listdir(TEMP):
    shutil.move(os.path.join(TEMP, f), os.path.join(ROOT, f))
shutil.rmtree(TEMP)

# ── Final report ──────────────────────────────────────────────────────────────
final_imgs  = glob.glob(f"{ROOT}/*.jpg") + glob.glob(f"{ROOT}/*.png")
final_txts  = [f for f in glob.glob(f"{ROOT}/*.txt")
               if Path(f).name != "classes.txt"]
non_empty_f = [t for t in final_txts if os.path.getsize(t) > 0]
empty_f     = [t for t in final_txts if os.path.getsize(t) == 0]

print(f"\n{'=' * 60}")
print(f"  MERGE COMPLETE")
print(f"{'=' * 60}")
print(f"  Total images          : {len(final_imgs)}")
print(f"  ✅ Non-empty labels   : {len(non_empty_f)}  ← annotated frames")
print(f"  ⚠️  Empty labels       : {len(empty_f)}   ← no annotations")
print(f"\n  First 8 files:")
for f in sorted(os.listdir(ROOT))[:8]:
    size = os.path.getsize(os.path.join(ROOT, f))
    status = "✅" if size > 0 else "⚠️ "
    print(f"    {status} {f:<30} ({size} bytes)")
print(f"\n  Next steps:")
print(f"  python src/fix_empty_labels.py")
print(f"  python src/prepare_annotated_dataset.py")
print(f"{'=' * 60}")