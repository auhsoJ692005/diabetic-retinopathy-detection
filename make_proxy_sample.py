"""
make_proxy_sample.py

Stratified proxy sample from the cropped training index, weighted
toward minority classes.
"""

import sys
import pandas as pd
from pathlib import Path


CONFIG = {
    "train_index":   "datasets/cropped/cropped_index.csv",
    "output_csv":    "datasets/proxy/proxy_sample.csv",
    "total_size":    400,
    "seed":          42,
    "class_weights": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
}


def main():
    idx_path = Path(CONFIG["train_index"])
    if not idx_path.exists():
        sys.exit(f"Missing {idx_path}")
    df = pd.read_csv(idx_path)
    df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"Training rows available: {len(df)}")

    weights = CONFIG["class_weights"]
    total_w = sum(weights[l] for l in df["label"].unique() if l in weights)
    if total_w == 0:
        sys.exit("No classes matched the weighting map.")

    parts = []
    for label, sub in df.groupby("label"):
        w = weights.get(label, 1.0)
        n = max(1, int(round(CONFIG["total_size"] * w / total_w)))
        n = min(n, len(sub))
        parts.append(sub.sample(n=n, random_state=CONFIG["seed"]))
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=CONFIG["seed"]).reset_index(drop=True)

    out = Path(CONFIG["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)

    print(f"\nProxy sample size: {len(sample)}")
    print("Class distribution:")
    print(sample["label"].value_counts().sort_index())
    print("Dataset distribution:")
    print(sample["dataset"].value_counts())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()