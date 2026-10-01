"""
slice_bags.py

Slices each preprocessed retinal image into 224x224 non-overlapping
patches and builds a bag index for MIL training.

Reads:  datasets/preprocessed/preprocessed_index.csv
Writes: datasets/bags/<split>/<image_id>/patch_<row>_<col>.png
        datasets/bags/bag_index.csv

Patches are skipped if fewer than 20% of pixels have intensity > 10,
which removes low-information edge patches around the circular mask.
Raising this from 0.05 (previous run) trims edge patches and tightens
the bag-size distribution.

Bags are grouped by image_id: one retinal image = one bag.

IMPORTANT: If you previously ran this with a lower min_informative_frac,
delete datasets/bags/ entirely before re-running. Otherwise stale
patches from the old run will remain on disk and the bag index will
not match the filesystem.
"""

import sys
import time
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


CONFIG = {
    "input_index":   "datasets/preprocessed/preprocessed_index.csv",
    "output_root":   "datasets/bags",
    "output_index":  "datasets/bags/bag_index.csv",
    "patch_size":    224,
    "stride":        224,
    "min_informative_frac": 0.20,  # raised from 0.05
    "intensity_threshold":  10,
    "save_format":   ".png",
    "overwrite":     True,          # force rewrite; assumes bags dir was cleared
    "clean_output":  True,          # delete output_root before writing
    "log_every":     250,
}


def slice_image(img, patch_size, stride):
    """Yield (row, col, patch) tuples for a full grid over the image."""
    h, w = img.shape[:2]
    for r, y in enumerate(range(0, h - patch_size + 1, stride)):
        for c, x in enumerate(range(0, w - patch_size + 1, stride)):
            patch = img[y:y + patch_size, x:x + patch_size]
            yield r, c, patch


def is_informative(patch, threshold, min_frac):
    """Return True if the patch has at least min_frac pixels above threshold."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    frac = float((gray > threshold).mean())
    return frac >= min_frac


def main():
    in_idx = Path(CONFIG["input_index"])
    out_root = Path(CONFIG["output_root"])
    out_idx = Path(CONFIG["output_index"])

    if not in_idx.exists():
        sys.exit(f"Missing input index: {in_idx}")

    # Clean output directory if requested
    if CONFIG["clean_output"] and out_root.exists():
        print(f"Removing existing output at {out_root} ...")
        shutil.rmtree(out_root)

    df = pd.read_csv(in_idx)
    n_total = len(df)

    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Input index:          {in_idx}")
    print(f"Output root:          {out_root}")
    print(f"Patch size:           {CONFIG['patch_size']}x{CONFIG['patch_size']}")
    print(f"Stride:               {CONFIG['stride']} "
          f"({'no overlap' if CONFIG['stride'] == CONFIG['patch_size'] else 'overlap'})")
    print(f"Min informative frac: {CONFIG['min_informative_frac']}")
    print(f"Intensity threshold:  {CONFIG['intensity_threshold']}")
    print(f"Clean output:         {CONFIG['clean_output']}")
    print(f"Images to process:    {n_total}")

    print("\n" + "=" * 60)
    print("Input distribution")
    print("=" * 60)
    print(df.groupby(["split", "dataset"]).size().unstack(fill_value=0))

    bag_rows = []
    failed = []
    total_patches = 0
    total_skipped = 0
    total_written = 0
    images_with_zero_patches = []
    per_bag_sizes = []

    t0 = time.time()
    pbar = tqdm(df.iterrows(), total=n_total, desc="Slicing")

    for i, (_, row) in enumerate(pbar):
        src = Path(row["image_path"])
        if not src.exists():
            failed.append((row["image_id"], "missing_input"))
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            failed.append((row["image_id"], "unreadable"))
            continue

        h, w = img.shape[:2]
        if h < CONFIG["patch_size"] or w < CONFIG["patch_size"]:
            failed.append((row["image_id"], f"too_small_{h}x{w}"))
            continue

        img_dir = out_root / row["split"] / row["image_id"]
        img_dir.mkdir(parents=True, exist_ok=True)

        image_patch_count = 0
        image_skipped = 0
        image_written = 0

        for r, c, patch in slice_image(img, CONFIG["patch_size"], CONFIG["stride"]):
            if not is_informative(patch,
                                  CONFIG["intensity_threshold"],
                                  CONFIG["min_informative_frac"]):
                image_skipped += 1
                continue

            patch_name = f"patch_{r:02d}_{c:02d}{CONFIG['save_format']}"
            patch_path = img_dir / patch_name

            if not patch_path.exists() or CONFIG["overwrite"]:
                ok = cv2.imwrite(str(patch_path), patch)
                if not ok:
                    failed.append((row["image_id"], f"write_failed_{patch_name}"))
                    continue
                image_written += 1

            bag_rows.append({
                "bag_id":           f"{row['image_id']}_r{r:02d}_c{c:02d}",
                "parent_image_id":  row["image_id"],
                "dataset":          row["dataset"],
                "split":            row["split"],
                "label":            int(row["label"]),
                "patch_path":       str(patch_path.resolve()),
                "patch_row":        r,
                "patch_col":        c,
            })
            image_patch_count += 1

        total_patches += image_patch_count
        total_skipped += image_skipped
        total_written += image_written
        per_bag_sizes.append(image_patch_count)

        if image_patch_count == 0:
            images_with_zero_patches.append(row["image_id"])
            failed.append((row["image_id"], "no_informative_patches"))

        pbar.set_postfix({
            "patches": total_patches,
            "skipped": total_skipped,
            "avg/img": f"{total_patches / (i + 1):.1f}",
        })

        if (i + 1) % CONFIG["log_every"] == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_total - i - 1) / max(0.01, rate)
            tqdm.write(
                f"[{i + 1}/{n_total}]  "
                f"patches={total_patches}  "
                f"skipped={total_skipped}  "
                f"avg/img={total_patches / (i + 1):.1f}  "
                f"elapsed={elapsed:.0f}s  "
                f"eta={remaining:.0f}s"
            )

    pbar.close()
    elapsed = time.time() - t0

    if not bag_rows:
        sys.exit("No patches produced — check inputs.")

    bags_df = pd.DataFrame(bag_rows)
    bags_df.to_csv(out_idx, index=False)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Slicing complete")
    print("=" * 60)
    print(f"Images processed:  {bags_df['parent_image_id'].nunique()} / {n_total}")
    print(f"Total patches:     {total_patches}")
    print(f"Patches skipped:   {total_skipped}")
    print(f"Patches written:   {total_written}")
    print(f"Avg patches/img:   {total_patches / max(1, bags_df['parent_image_id'].nunique()):.1f}")
    print(f"Time:              {elapsed:.1f}s "
          f"({elapsed / max(1, n_total):.3f}s per image)")

    # Skip rate
    if total_patches + total_skipped > 0:
        skip_rate = total_skipped / (total_patches + total_skipped)
        print(f"Skip rate:         {skip_rate:.1%}")

    if images_with_zero_patches:
        print(f"\nImages with zero informative patches: "
              f"{len(images_with_zero_patches)}")
        for img_id in images_with_zero_patches[:10]:
            print(f"  {img_id}")

    if failed:
        print(f"\nFailed images (total {len(failed)}):")
        for img_id, reason in failed[:15]:
            print(f"  {img_id}: {reason}")

    print("\n" + "=" * 60)
    print("Patch distribution by split and dataset")
    print("=" * 60)
    print(bags_df.groupby(["split", "dataset"]).size().unstack(fill_value=0))

    print("\nPatches per class:")
    print(bags_df["label"].value_counts().sort_index())

    print("\nBag size statistics (patches per image):")
    sizes = pd.Series(per_bag_sizes)
    print(f"  min:    {sizes.min()}")
    print(f"  25%:    {int(sizes.quantile(0.25))}")
    print(f"  median: {int(sizes.median())}")
    print(f"  75%:    {int(sizes.quantile(0.75))}")
    print(f"  90%:    {int(sizes.quantile(0.90))}")
    print(f"  95%:    {int(sizes.quantile(0.95))}")
    print(f"  max:    {sizes.max()}")
    print(f"  mean:   {sizes.mean():.1f}")

    # Per-split bag size stats
    print("\nTrain bag sizes:")
    train_df = bags_df[bags_df["split"] == "train"]
    train_sizes = train_df.groupby("parent_image_id").size()
    print(train_sizes.describe())

    print("\nEval bag sizes:")
    eval_df = bags_df[bags_df["split"] == "eval"]
    eval_sizes = eval_df.groupby("parent_image_id").size()
    print(eval_sizes.describe())

    print(f"\nSaved bag index to {out_idx}")


if __name__ == "__main__":
    main()