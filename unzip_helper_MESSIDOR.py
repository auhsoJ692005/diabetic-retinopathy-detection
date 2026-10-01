import os
import zipfile
from tqdm import tqdm

# 1. Point this to where your 12 zip folders are sitting
zip_dir = r"./datasets/zipped_datasets/MESSIDOR"
# 2. This is where all images will cleanly merge together
output_dir = r"./datasets/MESSIDOR"

os.makedirs(output_dir, exist_ok=True)

# Find all zip files in that directory (only actual .zip files)
zip_files = [
    f for f in os.listdir(zip_dir)
    if f.lower().endswith('.zip') and os.path.isfile(os.path.join(zip_dir, f))
]

# Exclude the output directory if it happens to be inside zip_dir
zip_files = [f for f in zip_files if not os.path.join(zip_dir, f).startswith(os.path.abspath(output_dir))]

print(f"Found {len(zip_files)} zipped files to extract.")

if not zip_files:
    print("No zip files found. Check your 'zip_dir' path.")
    raise SystemExit

# Track filename collisions
seen_names = set()
collisions = 0

# Loop through each zip file and extract it
for i, zip_file in enumerate(sorted(zip_files), 1):
    zip_path = os.path.join(zip_dir, zip_file)
    print(f"\n[{i}/{len(zip_files)}] Extracting: {zip_file}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            for file in tqdm(file_list, desc="Progress", unit="file"):
                # Skip directories
                if file.endswith('/') or file.endswith('\\'):
                    continue

                # Skip macOS metadata
                if "__MACOSX" in file or os.path.basename(file).startswith("._"):
                    continue

                filename = os.path.basename(file)
                if not filename:
                    continue

                # Handle duplicate filenames by renaming
                target_name = filename
                if target_name in seen_names:
                    collisions += 1
                    base, ext = os.path.splitext(filename)
                    target_name = f"{base}_dup{collisions}{ext}"
                
                seen_names.add(target_name)
                target_path = os.path.join(output_dir, target_name)

                # Skip if already extracted (resume-safe)
                if os.path.exists(target_path):
                    continue

                try:
                    with zip_ref.open(file) as source, open(target_path, "wb") as target:
                        # Stream in chunks to avoid loading huge files into RAM
                        while True:
                            chunk = source.read(1024 * 1024)  # 1 MB chunks
                            if not chunk:
                                break
                            target.write(chunk)
                except Exception as e:
                    print(f"\n  ⚠ Failed to extract {file}: {e}")
                    # Clean up partial file
                    if os.path.exists(target_path):
                        try:
                            os.remove(target_path)
                        except OSError:
                            pass

    except zipfile.BadZipFile:
        print(f"\n  ✗ Skipping corrupt/incomplete zip: {zip_file}")
        continue
    except Exception as e:
        print(f"\n  ✗ Error opening {zip_file}: {e}")
        continue

print(f"\nDone! All datasets unzipped and merged into: {os.path.abspath(output_dir)}")
print(f"Total unique files: {len(seen_names)}")
if collisions:
    print(f"Renamed {collisions} duplicate filenames.")