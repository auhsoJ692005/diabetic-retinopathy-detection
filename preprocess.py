"""
preprocess_full.py

Applies the winning Optuna preprocessing (green channel + Gaussian blur=5)
to every cropped image at native resolution.

Reads:  datasets/cropped/cropped_index.csv
Writes: datasets/preprocessed/preprocessed_index.csv
        datasets/preprocessed/<image_id>.png

Preprocessing:
1. Load cropped image (BGR, native resolution)
2. Convert BGR -> G channel -> RGB (replicate to 3 channels)
3. Gaussian blur with kernel size 5, sigma auto-derived (sigmaX=0)

Why replicate green to 3 channels:
- DenseNet-121 expects 3-channel input
- Replicating preserves the ImageNet-pretrained input interface
- The value distribution of green is close enough to RGB brightness
  that pretrained features remain useful

Params are read from CONFIG below. To change them, edit here and re-run.
The chosen values come from Optuna trial #16 (green, blur=5, val_loss=1.2612).
"""

import sys
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


CONFIG = {
    "input_index":   "datasets/cropped/cropped_index.csv",
    "output_dir":    "datasets/preprocessed",
    "output_index":  "datasets/preprocessed/preprocessed_index.csv",
    "color_space":   "green",   # from Optuna best trial
    "blur_kernel":   5,         # from Optuna best trial
    "save_format":   ".png",    # PNG to avoid JPEG recompression
    "overwrite":     False,     # skip already-processed files
}


def apply_color_space(img_bgr, mode):
    """Return 3-channel uint8 image for the given color-space mode."""
    if mode == "rgb":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if mode == "green":
        g = img_bgr[:, :, 1]
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    if mode == "hsv":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if mode == "lab":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    raise ValueError(f"Unknown color_space: {mode}")


def apply_blur(img, kernel_size):
    k = int(kernel_size)
    if k <= 1:
        return img
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)


def preprocess_image(img_bgr, params):
    img = apply_color_space(img_bgr, params["color_space"])
    img = apply_blur(img, params["blur_kernel"])
    return img


def main():
    in_idx  = Path(CONFIG["input_index"])
    out_dir = Path(CONFIG["output_dir"])
    out_idx = Path(CONFIG["output_index"])

    if not in_idx.exists():
        sys.exit(f"Missing input index: {in_idx}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_idx)
    print(f"Loaded {len(df)} rows from {in_idx}")
    print(f"Preprocessing: color_space={CONFIG['color_space']}  "
          f"blur_kernel={CONFIG['blur_kernel']}")

    params = {
        "color_space": CONFIG["color_space"],
        "blur_kernel": CONFIG["blur_kernel"],
    }

    new_paths = []
    skipped = 0
    failed = []

    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        src = Path(row["image_path"])
        out_path = out_dir / f"{row['image_id']}{CONFIG['save_format']}"

        # Skip already-processed
        if out_path.exists() and not CONFIG["overwrite"]:
            new_paths.append(str(out_path.resolve()))
            skipped += 1
            continue

        if not src.exists():
            failed.append((row["image_id"], "missing_input"))
            new_paths.append(None)
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            failed.append((row["image_id"], "unreadable"))
            new_paths.append(None)
            continue

        processed = preprocess_image(img, params)
        cv2.imwrite(str(out_path), processed)
        new_paths.append(str(out_path.resolve()))

    elapsed = time.time() - t0

    df["image_path"] = new_paths
    before = len(df)
    df = df[df["image_path"].notna()].reset_index(drop=True)
    dropped = before - len(df)

    print(f"\n{'=' * 60}")
    print("Preprocessing complete")
    print(f"{'=' * 60}")
    print(f"Total:    {len(df)}")
    print(f"Skipped:  {skipped} (already existed)")
    print(f"Dropped:  {dropped}")
    print(f"Time:     {elapsed:.1f}s  "
          f"({elapsed / max(1, len(df) - skipped):.3f}s per image)")

    if failed:
        print("\nFirst failures:")
        for img_id, reason in failed[:10]:
            print(f"  {img_id}: {reason}")

    # Sanity check: verify one image actually has the expected properties
    print(f"\n{'=' * 60}")
    print("Sanity check on a sample")
    print(f"{'=' * 60}")
    sample = df.sample(min(3, len(df)), random_state=42)
    for _, row in sample.iterrows():
        img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            print(f"  {row['image_id']}: FAILED to reload")
            continue
        # Green-channel replicate -> R=G=B per pixel
        b, g, r = cv2.split(img)
        same = np.array_equal(r, g) and np.array_equal(g, b)
        print(f"  {row['image_id']}: shape={img.shape}  "
              f"mean={img.mean():.1f}  "
              f"all_channels_equal={same}  "
              f"label={row['label']}  dataset={row['dataset']}")

    df.to_csv(out_idx, index=False)
    print(f"\nSaved preprocessed index to {out_idx}")


if __name__ == "__main__":
    main()