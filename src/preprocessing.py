"""
preprocessing.py
----------------
Frame extraction, normalization, optical flow computation,
data augmentation, and train/val/test splitting.

Supports TWO dataset formats:
  1. Kaggle UCF-Crime  → pre-extracted PNG frames  (use prepare_kaggle_frames)
  2. Raw MP4 videos    → extract frames yourself    (use split_dataset)

Usage:
    python src/preprocessing.py
    # or import and call functions individually from notebooks
"""

import cv2
import os
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import shutil


# ──────────────────────────────────────────────────────────────────────────────
#  KAGGLE UCF-CRIME  (PNG frames — primary method)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_kaggle_frames(kaggle_dir: str,
                          processed_dir: str,
                          max_per_category: int = 200,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15) -> dict:
    """
    Kaggle UCF-Crime dataset gives pre-extracted .png frames.
    This function:
      1. Reads PNG frames from kaggle_dir/train/ and kaggle_dir/test/
      2. Resizes every frame to 640×640
      3. Saves as .jpg into processed_dir/train/ val/ test/

    Folder structure expected from Kaggle:
        kaggle_dir/
        ├── train/
        │   ├── Abuse/       (*.png frames)
        │   ├── Assault/
        │   ├── Fighting/
        │   ├── Normal/
        │   └── ...
        └── test/
            ├── Abuse/
            └── ...

    Args:
        kaggle_dir:        Root of the unzipped Kaggle dataset.
        processed_dir:     Where to write resized frames.
        max_per_category:  Max frames to take per category (keeps storage small).
        val_ratio:         Fraction of train frames to use as validation.
        test_ratio:        Fraction of train frames to use as test.

    Returns:
        Dict with frame counts per split.
    """
    counts = {"train": 0, "val": 0, "test": 0}

    # ── Process Kaggle 'train' folder ────────────────────────────────────────
    train_path = os.path.join(kaggle_dir, "train")
    if not os.path.exists(train_path):
        # Some Kaggle versions have no subfolder — frames are directly inside
        train_path = kaggle_dir
        print(f"[INFO] No 'train/' subfolder found — reading directly from {kaggle_dir}")

    categories = sorted([
        d for d in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, d))
    ])

    if not categories:
        print(f"[WARN] No category subfolders found in {train_path}")
        return counts

    print(f"\nFound {len(categories)} categories: {categories}\n")

    for cat in categories:
        src_cat = os.path.join(train_path, cat)

        # Accept both .png and .jpg source frames
        frames = sorted(
            glob.glob(f"{src_cat}/*.png") +
            glob.glob(f"{src_cat}/*.jpg")
        )

        if not frames:
            print(f"  [SKIP] {cat} — no .png/.jpg frames found")
            continue

        # Limit frames per category to save disk space
        frames = frames[:max_per_category]
        random.seed(42)
        random.shuffle(frames)

        # Split: 70 / 15 / 15
        n       = len(frames)
        n_val   = max(1, int(val_ratio  * n))
        n_test  = max(1, int(test_ratio * n))
        n_train = n - n_val - n_test

        split_map = {
            "train": frames[:n_train],
            "val":   frames[n_train: n_train + n_val],
            "test":  frames[n_train + n_val:],
        }

        for split_name, split_frames in split_map.items():
            dst_dir = os.path.join(processed_dir, split_name, cat)
            os.makedirs(dst_dir, exist_ok=True)

            saved = 0
            for i, fp in enumerate(tqdm(split_frames,
                                        desc=f"  {cat}/{split_name}",
                                        leave=False)):
                frame = cv2.imread(fp)
                if frame is None:
                    continue
                frame_resized = cv2.resize(frame, (640, 640),
                                           interpolation=cv2.INTER_LINEAR)
                out_path = os.path.join(dst_dir, f"frame_{i:05d}.jpg")
                cv2.imwrite(out_path, frame_resized)
                saved += 1

            counts[split_name] += saved

        print(f"  ✓ {cat:<25}  "
              f"train={len(split_map['train'])}  "
              f"val={len(split_map['val'])}  "
              f"test={len(split_map['test'])}")

    # ── Process Kaggle 'test' folder (add to our test split) ─────────────────
    kaggle_test_path = os.path.join(kaggle_dir, "test")
    if os.path.exists(kaggle_test_path) and kaggle_test_path != train_path:
        print("\nProcessing Kaggle test/ folder → adding to our test split...")
        for cat in sorted(os.listdir(kaggle_test_path)):
            src_cat = os.path.join(kaggle_test_path, cat)
            if not os.path.isdir(src_cat):
                continue
            frames = sorted(
                glob.glob(f"{src_cat}/*.png") +
                glob.glob(f"{src_cat}/*.jpg")
            )[:50]   # 50 test frames per category

            dst_dir = os.path.join(processed_dir, "test", cat)
            os.makedirs(dst_dir, exist_ok=True)

            for i, fp in enumerate(frames):
                frame = cv2.imread(fp)
                if frame is None:
                    continue
                frame_resized = cv2.resize(frame, (640, 640))
                cv2.imwrite(os.path.join(dst_dir, f"ktest_{i:05d}.jpg"), frame_resized)
                counts["test"] += 1

    return counts


# ──────────────────────────────────────────────────────────────────────────────
#  FRAME EXTRACTION FROM MP4  (fallback — if you have raw videos)
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str, fps: int = 1) -> int:
    """
    Extract frames from a video at the given FPS rate.
    Each frame is resized to 640×640 and normalized to 0–255 uint8.

    Args:
        video_path:  Path to input .mp4 / .avi file.
        output_dir:  Folder where extracted frames will be saved.
        fps:         How many frames to extract per second of video.

    Returns:
        Number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval   = max(1, int(video_fps / fps))
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    saved     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frame_resized = cv2.resize(frame, (640, 640),
                                       interpolation=cv2.INTER_LINEAR)
            frame_norm = (frame_resized / 255.0 * 255).astype(np.uint8)
            out_path   = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(out_path, frame_norm)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def split_dataset(raw_dir: str,
                  processed_dir: str,
                  split: tuple = (0.70, 0.15, 0.15),
                  fps: int = 1) -> dict:
    """
    Walk raw_dir for .mp4 files, extract frames, split into train/val/test.
    Use this ONLY if you have raw .mp4 videos (not the Kaggle PNG version).
    """
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    video_files = sorted(glob.glob(f"{raw_dir}/**/*.mp4", recursive=True))
    if not video_files:
        print(f"[WARN] No .mp4 files found in {raw_dir}")
        print("       If you downloaded from Kaggle (PNG frames), use prepare_kaggle_frames() instead.")
        return {}

    random.seed(42)
    random.shuffle(video_files)

    n       = len(video_files)
    n_train = int(split[0] * n)
    n_val   = int(split[1] * n)

    splits = {
        "train": video_files[:n_train],
        "val":   video_files[n_train: n_train + n_val],
        "test":  video_files[n_train + n_val:],
    }

    counts = {}
    for split_name, videos in splits.items():
        total_frames = 0
        print(f"\n[{split_name.upper()}]  {len(videos)} videos")
        for vp in tqdm(videos, desc=f"  Extracting {split_name}"):
            category = Path(vp).parent.name
            out_dir  = os.path.join(processed_dir, split_name, category,
                                    Path(vp).stem)
            n_saved  = extract_frames(vp, out_dir, fps=fps)
            total_frames += n_saved
        counts[split_name] = total_frames
        print(f"  → {total_frames} frames saved")

    return counts


# ──────────────────────────────────────────────────────────────────────────────
#  OPTICAL FLOW
# ──────────────────────────────────────────────────────────────────────────────

def compute_optical_flow(frame1: np.ndarray,
                         frame2: np.ndarray) -> tuple:
    """
    Compute dense Farneback optical flow between two consecutive frames.

    Returns:
        (magnitude, angle) arrays — both shape (H, W), float32.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        flow       = None,
        pyr_scale  = 0.5,
        levels     = 3,
        winsize    = 15,
        iterations = 3,
        poly_n     = 5,
        poly_sigma = 1.2,
        flags      = 0,
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude, angle


def extract_flow_features(frame_dir: str) -> dict:
    """
    Compute optical-flow statistics over a sorted sequence of frames.
    Works with both .jpg and .png frames.

    Returns:
        Dict with keys: mean_flow, max_flow, std_flow, motion_ratio
    """
    frame_paths = sorted(
        glob.glob(os.path.join(frame_dir, "*.jpg")) +
        glob.glob(os.path.join(frame_dir, "*.png"))
    )

    if len(frame_paths) < 2:
        return {"mean_flow": 0.0, "max_flow": 0.0,
                "std_flow": 0.0, "motion_ratio": 0.0}

    mean_flows, max_flows, std_flows = [], [], []

    for i in range(len(frame_paths) - 1):
        f1 = cv2.imread(frame_paths[i])
        f2 = cv2.imread(frame_paths[i + 1])
        if f1 is None or f2 is None:
            continue
        mag, _ = compute_optical_flow(f1, f2)
        mean_flows.append(float(mag.mean()))
        max_flows.append(float(mag.max()))
        std_flows.append(float(mag.std()))

    if not mean_flows:
        return {"mean_flow": 0.0, "max_flow": 0.0,
                "std_flow": 0.0, "motion_ratio": 0.0}

    mean_f = float(np.mean(mean_flows))
    return {
        "mean_flow":    mean_f,
        "max_flow":     float(np.mean(max_flows)),
        "std_flow":     float(np.mean(std_flows)),
        "motion_ratio": float(np.sum(np.array(mean_flows) > 2.0) / len(mean_flows)),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  DATA AUGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

def augment_frame(frame: np.ndarray) -> list:
    """
    Apply common augmentations to a single frame.
    Returns a list of augmented variants (including the original).
    """
    augmented = [frame]

    augmented.append(cv2.flip(frame, 1))                                      # horizontal flip
    augmented.append(cv2.convertScaleAbs(frame, alpha=1.0, beta=30))          # brighter
    augmented.append(cv2.convertScaleAbs(frame, alpha=1.0, beta=-30))         # darker

    h, w   = frame.shape[:2]
    center = (w // 2, h // 2)
    for angle in (10, -10):                                                   # rotation
        M       = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))
        augmented.append(rotated)

    return augmented


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN  — auto-detects PNG (Kaggle) vs MP4 (raw videos)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Border Surveillance — Preprocessing Pipeline")
    print("=" * 55)

    RAW_DIR       = "data/raw/ucf_crime"
    PROCESSED_DIR = "data/processed"

    # ── Auto-detect dataset format ────────────────────────────────────────────
    png_files = glob.glob(f"{RAW_DIR}/**/*.png", recursive=True)
    mp4_files = glob.glob(f"{RAW_DIR}/**/*.mp4", recursive=True)

    if png_files:
        print(f"\n[AUTO-DETECT] Found {len(png_files)} PNG frames")
        print("  → Using Kaggle mode (prepare_kaggle_frames)\n")
        counts = prepare_kaggle_frames(
            kaggle_dir        = RAW_DIR,
            processed_dir     = PROCESSED_DIR,
            max_per_category  = 200,   # increase if you have storage
        )

    elif mp4_files:
        print(f"\n[AUTO-DETECT] Found {len(mp4_files)} MP4 videos")
        print("  → Using video extraction mode (split_dataset)\n")
        counts = split_dataset(
            raw_dir       = RAW_DIR,
            processed_dir = PROCESSED_DIR,
            fps           = 1,
        )

    else:
        print(f"\n[ERROR] No .png or .mp4 files found in {RAW_DIR}")
        print("  Please download the dataset first.")
        print("  See data/README.md for instructions.")
        counts = {}

    # ── Summary ───────────────────────────────────────────────────────────────
    if counts:
        print("\n── Summary ──────────────────────────────────────")
        for split_name, n in counts.items():
            print(f"  {split_name:<8}: {n:>6} frames")
        print(f"  {'TOTAL':<8}: {sum(counts.values()):>6} frames")
        print("─" * 50)
        print("  ✅ Preprocessing complete!")
        print(f"  Output → {PROCESSED_DIR}/train | val | test")
