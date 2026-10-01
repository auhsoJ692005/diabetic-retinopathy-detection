"""
crop_images.py

Circular crop + black-border removal for APTOS and Messidor-2.

Scale-aware: works on both small (614x819) and large (1958x2588) images.
Uses contour area + centrality scoring instead of just largest contour.
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


CONFIG = {
    "input_index":     "datasets/unified/unified_index.csv",
    "output_dir":      "datasets/cropped",
    "output_index":    "datasets/cropped/cropped_index.csv",
    "target_size":     None,
    "apply_mask":      True,
    "min_retina_frac": 0.02,
}


def detect_retina_bbox(img_bgr):
    """
    Return (x, y, w, h) bounding box of the retinal region, or None.

    Scale-aware: blur and morphology kernels scale with image size.
    Picks the contour that is both large and centered, not just the largest.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    min_dim = min(H, W)

    # Scale blur kernel to ~1% of min dimension, forced odd, min 5
    k = max(5, int(min_dim * 0.01) | 1)
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Scale morphology kernel to ~2% of min dimension, forced odd, min 5
    mk = max(5, int(min_dim * 0.02) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cx_img, cy_img = W / 2, H / 2

    best, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        # Ignore tiny contours
        if area < 0.03 * H * W:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        # Normalized distance from image center
        dist = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5
        dist_norm = dist / (0.5 * (H + W))
        # Score: area fraction minus penalty for off-center
        score = (area / (H * W)) - 0.5 * dist_norm
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        # Fallback: largest contour regardless of size
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)
    return best


def circular_mask(shape, cx, cy, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, thickness=-1)
    return mask


def crop_retina(img_bgr, apply_mask, target_size):
    bbox = detect_retina_bbox(img_bgr)
    if bbox is None:
        return None

    x, y, w, h = bbox
    if (w * h) / (img_bgr.shape[0] * img_bgr.shape[1]) < CONFIG["min_retina_frac"]:
        return None

    # Make bbox square, centered on the retina's center
    side = max(w, h)
    cx = x + w // 2
    cy = y + h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(img_bgr.shape[1], x0 + side)
    y1 = min(img_bgr.shape[0], y0 + side)

    cropped = img_bgr[y0:y1, x0:x1].copy()

    if apply_mask:
        h_c, w_c = cropped.shape[:2]
        m = circular_mask(cropped.shape, w_c / 2, h_c / 2, min(w_c, h_c) / 2)
        cropped[m == 0] = 0

    if target_size is not None:
        cropped = cv2.resize(cropped, (target_size, target_size),
                             interpolation=cv2.INTER_AREA)

    return cropped


def main():
    in_idx  = Path(CONFIG["input_index"])
    out_dir = Path(CONFIG["output_dir"])
    out_idx = Path(CONFIG["output_index"])
    if not in_idx.exists():
        sys.exit(f"Missing input index: {in_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_idx)
    print(f"Loaded {len(df)} rows from {in_idx}")

    new_paths, failures = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping"):
        src = Path(row["image_path"])
        if not src.exists():
            failures.append((row["image_id"], "missing_file"))
            new_paths.append(None)
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            failures.append((row["image_id"], "unreadable"))
            new_paths.append(None)
            continue
        cropped = crop_retina(img, CONFIG["apply_mask"], CONFIG["target_size"])
        if cropped is None:
            failures.append((row["image_id"], "detection_failed"))
            new_paths.append(None)
            continue
        out_path = out_dir / f"{row['image_id']}.png"
        cv2.imwrite(str(out_path), cropped)
        new_paths.append(str(out_path.resolve()))

    df["image_path"] = new_paths
    before = len(df)
    df = df[df["image_path"].notna()].reset_index(drop=True)
    print(f"\nProcessed: {len(df)}  Dropped: {before - len(df)}")
    if failures:
        print("First failures:")
        for img_id, reason in failures[:10]:
            print(f"  {img_id}: {reason}")
    df.to_csv(out_idx, index=False)
    print(f"Saved cropped index to {out_idx}")


if __name__ == "__main__":
    main()