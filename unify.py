"""
unify_index.py

Builds a unified index across APTOS 2019 and Messidor-2,
verifies integrity, and produces an 80/20 stratified split.

Expected schemas:
- APTOS:      train.csv with columns ['id_code', 'diagnosis'] and images in train_images/
- Messidor-2: messidor_data.csv with columns
              ['image_id', 'adjudicated_dr_grade', 'adjudicated_dme', 'adjudicated_gradable']
              and images in images/

Adjust CONFIG paths below to match your local setup.
"""

import sys
import hashlib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# -----------------------------
# CONFIG
# -----------------------------
CONFIG = {
    "aptos_csv":       "datasets/aptos/train.csv",
    "aptos_images":    "datasets/aptos/train_images",
    "aptos_id_col":    "id_code",
    "aptos_label_col": "diagnosis",

    "messidor_csv":        "datasets/MESSIDOR-2/messidor_data.csv",
    "messidor_images":     "datasets/MESSIDOR-2/images",
    "messidor_id_col":     "image_id",
    "messidor_label_col":  "adjudicated_dr_grade",
    "messidor_filter_col": "adjudicated_gradable",

    "output_dir":  "data/unified",
    "split_ratio": 0.80,
    "seed":        42,
    "image_exts":  [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
}

APTOS_NAME    = "aptos"
MESSIDOR_NAME = "messidor2"


# -----------------------------
# HELPERS
# -----------------------------
def find_image(image_dir: Path, id_value, exts):
    """Return the first matching image path for a given id, or None."""
    id_str = str(id_value)

    # Direct match with each extension
    for ext in exts:
        candidate = image_dir / f"{id_str}{ext}"
        if candidate.exists():
            return candidate

    # If the id already carries an extension, try it as-is
    direct = image_dir / id_str
    if direct.is_file():
        return direct

    # Fallback: case-insensitive stem match
    stem_lower = Path(id_str).stem.lower()
    for p in image_dir.iterdir():
        if p.is_file() and p.stem.lower() == stem_lower and p.suffix.lower() in exts:
            return p
    return None


def build_dataset_index(csv_path, image_dir, dataset_name, exts,
                        id_col="id_code", label_col="diagnosis",
                        filter_col=None):
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"{csv_path}: missing label column '{label_col}'")
    if id_col not in df.columns:
        raise ValueError(f"{csv_path}: missing id column '{id_col}'")

    if filter_col is not None:
        if filter_col not in df.columns:
            raise ValueError(f"{csv_path}: missing filter column '{filter_col}'")
        before = len(df)
        df = df[df[filter_col] == 1].reset_index(drop=True)
        print(f"[{dataset_name}] dropped {before - len(df)} rows via {filter_col}==1")

    records, missing = [], []
    for _, row in df.iterrows():
        img_path = find_image(image_dir, row[id_col], exts)
        if img_path is None:
            missing.append(row[id_col])
            continue
        records.append({
            "image_id":   f"{dataset_name}_{row[id_col]}",
            "source_id":  row[id_col],
            "image_path": str(img_path.resolve()),
            "label":      int(row[label_col]),
            "dataset":    dataset_name,
        })

    if missing:
        print(f"[{dataset_name}] WARNING: {len(missing)} images not found on disk. "
              f"First few: {missing[:5]}")

    out = pd.DataFrame(records)
    print(f"[{dataset_name}] resolved {len(out)} / {len(df)} rows")
    return out


def check_label_range(df: pd.DataFrame) -> None:
    bad = df[~df["label"].between(0, 4)]
    if len(bad):
        raise ValueError(f"Labels outside 0-4 found:\n{bad.head()}")
    print(f"[check] all labels are within 0-4")


def check_duplicates(df: pd.DataFrame) -> None:
    dup_ids = df[df.duplicated("image_id", keep=False)]
    if len(dup_ids):
        raise ValueError(f"Duplicate image_id values:\n{dup_ids.head()}")
    print(f"[check] no duplicate image_id values")


def check_files_exist(df: pd.DataFrame) -> None:
    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image paths do not exist. "
                                f"First few: {missing[:5]}")
    print(f"[check] all {len(df)} image paths exist on disk")


def hash_split(df: pd.DataFrame) -> str:
    """Reproducible hash of split assignments for provenance."""
    h = hashlib.sha256()
    for _, row in df.sort_values("image_id").iterrows():
        h.update(f"{row['image_id']}:{row['split']}".encode())
    return h.hexdigest()[:12]


# -----------------------------
# MAIN
# -----------------------------
def main():
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    aptos_csv     = Path(CONFIG["aptos_csv"])
    aptos_imgs    = Path(CONFIG["aptos_images"])
    messidor_csv  = Path(CONFIG["messidor_csv"])
    messidor_imgs = Path(CONFIG["messidor_images"])

    for p in (aptos_csv, messidor_csv):
        if not p.exists():
            sys.exit(f"Missing required file: {p}")
    for p in (aptos_imgs, messidor_imgs):
        if not p.is_dir():
            sys.exit(f"Missing required directory: {p}")

    print("=" * 60)
    print("Building dataset indexes")
    print("=" * 60)

    aptos_df = build_dataset_index(
        aptos_csv, aptos_imgs, APTOS_NAME, CONFIG["image_exts"],
        id_col=CONFIG["aptos_id_col"],
        label_col=CONFIG["aptos_label_col"],
    )
    messidor_df = build_dataset_index(
        messidor_csv, messidor_imgs, MESSIDOR_NAME, CONFIG["image_exts"],
        id_col=CONFIG["messidor_id_col"],
        label_col=CONFIG["messidor_label_col"],
        filter_col=CONFIG["messidor_filter_col"],
    )

    if len(aptos_df) == 0:
        sys.exit("APTOS index is empty — check paths and CSV column names.")
    if len(messidor_df) == 0:
        sys.exit("Messidor-2 index is empty — check paths and CSV column names.")

    unified = pd.concat([aptos_df, messidor_df], ignore_index=True)

    print("\n" + "=" * 60)
    print("Integrity checks")
    print("=" * 60)
    check_label_range(unified)
    check_duplicates(unified)
    check_files_exist(unified)

    print("\nOverall label distribution:")
    print(unified["label"].value_counts().sort_index())
    print("\nPer-dataset label distribution:")
    print(unified.groupby(["dataset", "label"]).size().unstack(fill_value=0))

    print("\n" + "=" * 60)
    print(f"Stratified split ({int(CONFIG['split_ratio']*100)}/"
          f"{int((1 - CONFIG['split_ratio'])*100)})")
    print("=" * 60)

    train_parts, eval_parts = [], []
    for name, subset in unified.groupby("dataset"):
        tr, ev = train_test_split(
            subset,
            train_size=CONFIG["split_ratio"],
            stratify=subset["label"],
            random_state=CONFIG["seed"],
        )
        train_parts.append(tr)
        eval_parts.append(ev)
        print(f"[{name}] train={len(tr)}  eval={len(ev)}")

    train_df = pd.concat(train_parts, ignore_index=True).assign(split="train")
    eval_df  = pd.concat(eval_parts,  ignore_index=True).assign(split="eval")

    final = pd.concat([train_df, eval_df], ignore_index=True)
    final = final.sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts().sort_index())
    print("\nEval label distribution:")
    print(eval_df["label"].value_counts().sort_index())

    overlap = set(train_df["image_id"]) & set(eval_df["image_id"])
    if overlap:
        raise ValueError(f"Overlap between train and eval: {list(overlap)[:5]}")
    print("\n[check] no overlap between train and eval")

    unified_path = out_dir / "unified_index.csv"
    train_path   = out_dir / "train_index.csv"
    eval_path    = out_dir / "eval_index.csv"
    final.to_csv(unified_path, index=False)
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path,  index=False)

    print("\n" + "=" * 60)
    print("Saved")
    print("=" * 60)
    print(f"  {unified_path}  ({len(final)} rows)")
    print(f"  {train_path}  ({len(train_df)} rows)")
    print(f"  {eval_path}   ({len(eval_df)} rows)")
    print(f"\nSplit hash (for provenance): {hash_split(final)}")


if __name__ == "__main__":
    main()