"""
GTZAN Dataset Preparation Script
Downloads from Kaggle, extracts, and creates train/val/test splits
"""

import os
import json
import zipfile
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
KAGGLE_DATASET = "andradaolteanu/gtzan-dataset-music-genre-classification"
BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("GTZAN Dataset Preparation")
print("=" * 60)

# Step 1: Check if Kaggle CLI is available
print("\n[1/4] Checking for dataset...")
try:
    # Try automatic download
    subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    
    print("  → Downloading from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR)],
        check=True,
        capture_output=True,
        text=True
    )
    print("  ✓ Download complete")
    
    # Extract
    print("\n[2/4] Extracting dataset...")
    zip_files = list(RAW_DIR.glob("*.zip"))
    if zip_files:
        with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        zip_files[0].unlink()
        print(f"  ✓ Extracted to {RAW_DIR}")
        
except (FileNotFoundError, subprocess.CalledProcessError):
    print("  → Kaggle CLI not available or download failed")
    print("\n" + "─" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("─" * 60)
    print("\nPlease download GTZAN dataset manually:")
    print(f"1. Visit: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    print(f"2. Download the dataset ZIP file")
    print(f"3. Place it in: {RAW_DIR}")
    print(f"4. Run this script again")
    print("\nOR extract the dataset and place 'genres_original' folder at:")
    print(f"   {RAW_DIR}/genres_original/")
    print("─" * 60)
    
    # Check if already extracted
    if not any(RAW_DIR.rglob("genres_original")):
        print("\n✗ Dataset not found. Exiting...")
        exit(1)
    print("\n✓ Found existing dataset!")

# Step 3: Find genres directory
print("\n[3/4] Locating audio files...")
genres_dir = None
for path in RAW_DIR.rglob("genres_original"):
    genres_dir = path
    break

if not genres_dir or not genres_dir.exists():
    # Try alternative structure
    genres_dir = RAW_DIR / "Data" / "genres_original"
    if not genres_dir.exists():
        print("✗ Could not find genres_original directory")
        exit(1)

print(f"✓ Found genres at: {genres_dir}")

# Step 4: Create stratified splits
print("\n[4/4] Creating train/val/test splits...")

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

all_files = []
labels = []

for genre in GENRES:
    genre_path = genres_dir / genre
    if not genre_path.exists():
        print(f"⚠ Warning: Genre '{genre}' not found")
        continue
    
    wav_files = sorted(genre_path.glob("*.wav"))
    for wav_file in wav_files:
        all_files.append(str(wav_file.relative_to(genres_dir)))
        labels.append(genre)

print(f"✓ Found {len(all_files)} audio files across {len(set(labels))} genres")

# Stratified split: 70% train, 15% val, 15% test
train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_files, labels, test_size=0.3, stratify=labels, random_state=42
)

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# Save splits
splits = {
    'train': (train_files, train_labels),
    'val': (val_files, val_labels),
    'test': (test_files, test_labels)
}

for split_name, (files, split_labels) in splits.items():
    split_file = SPLITS_DIR / f"{split_name}.txt"
    with open(split_file, 'w') as f:
        for file_path, label in zip(files, split_labels):
            f.write(f"{file_path}\t{label}\n")
    print(f"  ✓ {split_name}.txt: {len(files)} files")

# Save metadata
metadata = {
    'genres_dir': str(genres_dir),
    'num_genres': len(GENRES),
    'genres': GENRES,
    'splits': {
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files)
    }
}

with open(DATA_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("✓ Dataset preparation complete!")
print("=" * 60)
print(f"\nDataset location: {genres_dir}")
print(f"Splits saved to: {SPLITS_DIR}")
print(f"\nSplit distribution:")
print(f"  Train: {len(train_files)} files (70%)")
print(f"  Val:   {len(val_files)} files (15%)")
print(f"  Test:  {len(test_files)} files (15%)")
print("\nGenre distribution (per split):")
for genre in GENRES:
    train_count = train_labels.count(genre)
    val_count = val_labels.count(genre)
    test_count = test_labels.count(genre)
    print(f"  {genre:12s}: Train={train_count:2d}, Val={val_count:2d}, Test={test_count:2d}")