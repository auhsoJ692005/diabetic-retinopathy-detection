import pandas as pd
import cv2
from pathlib import Path

failed_ids = [
    "aptos_4158c340fa49", "aptos_8846b09384a4",
]

df = pd.read_csv("datasets/unified/unified_index.csv")
sub = df[df["image_id"].isin(failed_ids)]
for _, row in sub.iterrows():
    img = cv2.imread(row["image_path"])
    if img is None:
        print(f"{row['image_id']}: UNREADABLE")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"{row['image_id']}: shape={img.shape} "
          f"gray_min={gray.min()} gray_max={gray.max()} "
          f"gray_mean={gray.mean():.1f} "
          f"label={row['label']} dataset={row['dataset']}")