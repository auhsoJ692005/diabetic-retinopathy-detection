import os
import pandas as pd

excel_dir = r"./datasets/messidor"
output_csv = r"./datasets/messidor/messidor_unified_labels.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

excel_files = [
    f for f in os.listdir(excel_dir)
    if f.lower().endswith(('.xls', '.xlsx')) and not f.startswith('~$')
]
print(f"Found {len(excel_files)} Excel files.")

all_dfs = []

for file in excel_files:
    file_path = os.path.join(excel_dir, file)
    print(f"\nReading: {file}")

    df = pd.read_excel(file_path)

    # Normalize headers: strip whitespace, lowercase, collapse internal spaces
    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
        .str.replace(r'\s+', ' ', regex=True)
    )

    df = df.rename(columns={
        'image name': 'image_id',
        'retinopathy grade': 'dr_stage',
        'risk of macular edema': 'macular_edema',
        # 'ophthalmologic department' is dropped below
    })

    required = {'image_id', 'dr_stage'}
    missing = required - set(df.columns)
    if missing:
        print(f"  ⚠️ Missing {missing} in {file}. Found: {list(df.columns)}")
        continue

    keep = ['image_id', 'dr_stage']
    if 'macular_edema' in df.columns:
        keep.append('macular_edema')

    df = df[keep].copy()

    # Clean image_id: string, strip, remove any accidental extension
    df['image_id'] = (
        df['image_id'].astype(str).str.strip()
        .str.replace(r'\.(tif|tiff|png|jpg|jpeg|ppm)$', '', regex=True, case=False)
    )

    # Coerce labels to numeric, drop invalid rows
    df['dr_stage'] = pd.to_numeric(df['dr_stage'], errors='coerce')
    if 'macular_edema' in df.columns:
        df['macular_edema'] = pd.to_numeric(df['macular_edema'], errors='coerce')

    before = len(df)
    df = df.dropna(subset=['image_id', 'dr_stage'])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with bad image_id/dr_stage.")

    df['dr_stage'] = df['dr_stage'].astype(int)
    if 'macular_edema' in df.columns:
        df['macular_edema'] = df['macular_edema'].astype('Int64')  # nullable int

    df['source_file'] = file
    all_dfs.append(df)
    print(f"  ✓ {len(df)} rows")

if not all_dfs:
    raise SystemExit("❌ No valid files loaded. Check path and headers.")

unified = pd.concat(all_dfs, ignore_index=True)

dupes = unified['image_id'].duplicated().sum()
if dupes:
    print(f"\n⚠️ {dupes} duplicate image_ids — keeping first.")
    unified = unified.drop_duplicates(subset='image_id', keep='first')

print("\nDR stage distribution:")
print(unified['dr_stage'].value_counts().sort_index())

# Sanity check: Messidor should only have grades 0-3
unexpected = unified[~unified['dr_stage'].isin([0, 1, 2, 3])]
if len(unexpected):
    print(f"\n⚠️ {len(unexpected)} rows have unexpected DR grades "
          f"(Messidor-1 is 0–3): {unexpected['dr_stage'].unique().tolist()}")

unified.to_csv(output_csv, index=False)
print(f"\n🎉 Saved {len(unified)} rows to {output_csv}")