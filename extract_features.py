"""
Feature Extraction Pipeline for GTZAN Dataset
Extracts Mel-spectrograms (128x130) and MFCC (13x130) features
"""

import os
import json
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"

# Feature parameters
SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MELS = 128
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

# Create output directories
MELSPEC_DIR = PROCESSED_DIR / "melspectrogram"
MFCC_DIR = PROCESSED_DIR / "mfcc"
MELSPEC_DIR.mkdir(parents=True, exist_ok=True)
MFCC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("FEATURE EXTRACTION PIPELINE")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Sample Rate:  {SAMPLE_RATE} Hz")
print(f"  Duration:     {DURATION} seconds")
print(f"  Mel-spec:     {N_MELS} × ~130 (n_mels × time_steps)")
print(f"  MFCC:         {N_MFCC} × ~130 (n_coeffs × time_steps)")
print(f"  Hop Length:   {HOP_LENGTH}")
print(f"  FFT Size:     {N_FFT}")

# Load metadata
metadata_path = DATA_DIR / "metadata.json"
if not metadata_path.exists():
    print(f"\n✗ Error: metadata.json not found at {metadata_path}")
    print("Please run prepare_gtzan_data.py first!")
    exit(1)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

genres_dir = Path(metadata['genres_dir'])
genres = metadata['genres']
genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}

print(f"\nGenres: {', '.join(genres)}")
print(f"Dataset: {genres_dir}")


def extract_melspectrogram(audio, sr):
    """Extract Mel-spectrogram feature"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmax=sr // 2
    )
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_mfcc(audio, sr):
    """Extract MFCC feature"""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmax=sr // 2
    )
    return mfcc


def process_audio_file(file_path, genre):
    """Load audio and extract both features"""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Ensure exact 30 seconds
        target_length = SAMPLE_RATE * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Extract features
        melspec = extract_melspectrogram(audio, sr)
        mfcc = extract_mfcc(audio, sr)
        
        # Get genre index
        genre_idx = genre_to_idx[genre]
        
        return melspec, mfcc, genre_idx, True
        
    except Exception as e:
        print(f"\n✗ Error processing {file_path}: {e}")
        return None, None, None, False


def process_split(split_name):
    """Process all files in a split"""
    split_file = SPLITS_DIR / f"{split_name}.txt"
    
    if not split_file.exists():
        print(f"\n✗ Error: {split_file} not found")
        return
    
    # Read split file
    files = []
    labels = []
    with open(split_file, 'r') as f:
        for line in f:
            file_path, genre = line.strip().split('\t')
            files.append(file_path)
            labels.append(genre)
    
    print(f"\n{'=' * 70}")
    print(f"Processing {split_name.upper()} split ({len(files)} files)")
    print(f"{'=' * 70}")
    
    # Storage arrays
    melspec_features = []
    mfcc_features = []
    genre_labels = []
    file_names = []
    
    # Process each file with progress bar
    successful = 0
    failed = 0
    
    for file_path, genre in tqdm(zip(files, labels), total=len(files), desc=f"Extracting {split_name}"):
        full_path = genres_dir / file_path
        
        if not full_path.exists():
            print(f"\n⚠ Warning: File not found: {full_path}")
            failed += 1
            continue
        
        melspec, mfcc, genre_idx, success = process_audio_file(full_path, genre)
        
        if success:
            melspec_features.append(melspec)
            mfcc_features.append(mfcc)
            genre_labels.append(genre_idx)
            file_names.append(file_path)
            successful += 1
        else:
            failed += 1
    
    # Convert to numpy arrays
    melspec_features = np.array(melspec_features, dtype=np.float32)
    mfcc_features = np.array(mfcc_features, dtype=np.float32)
    genre_labels = np.array(genre_labels, dtype=np.int64)
    
    # Save features
    print(f"\nSaving features...")
    
    # Save Mel-spectrograms
    np.save(MELSPEC_DIR / f"{split_name}_features.npy", melspec_features)
    np.save(MELSPEC_DIR / f"{split_name}_labels.npy", genre_labels)
    
    # Save MFCC
    np.save(MFCC_DIR / f"{split_name}_features.npy", mfcc_features)
    np.save(MFCC_DIR / f"{split_name}_labels.npy", genre_labels)
    
    # Save file names for reference
    with open(PROCESSED_DIR / f"{split_name}_files.txt", 'w') as f:
        for fname in file_names:
            f.write(f"{fname}\n")
    
    print(f"✓ {split_name.upper()} split complete:")
    print(f"  Mel-spec shape: {melspec_features.shape}")
    print(f"  MFCC shape:     {mfcc_features.shape}")
    print(f"  Labels shape:   {genre_labels.shape}")
    print(f"  Successful:     {successful}")
    print(f"  Failed:         {failed}")
    
    return melspec_features.shape, mfcc_features.shape


# Process all splits
print("\n" + "=" * 70)
print("STARTING FEATURE EXTRACTION")
print("=" * 70)

splits = ['train', 'val', 'test']
results = {}

for split in splits:
    mel_shape, mfcc_shape = process_split(split)
    results[split] = {
        'melspec_shape': mel_shape,
        'mfcc_shape': mfcc_shape
    }

# Save extraction metadata
extraction_info = {
    'sample_rate': SAMPLE_RATE,
    'duration': DURATION,
    'n_mels': N_MELS,
    'n_mfcc': N_MFCC,
    'hop_length': HOP_LENGTH,
    'n_fft': N_FFT,
    'melspec_dir': str(MELSPEC_DIR),
    'mfcc_dir': str(MFCC_DIR),
    'genre_to_idx': genre_to_idx,
    'results': results
}

with open(PROCESSED_DIR / 'extraction_info.json', 'w') as f:
    json.dump(extraction_info, f, indent=2)

# Final summary
print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE!")
print("=" * 70)
print(f"\nOutput directories:")
print(f"  Mel-spectrograms: {MELSPEC_DIR}")
print(f"  MFCC:             {MFCC_DIR}")
print(f"\nFiles created per split:")
print(f"  <split>_features.npy  - Feature array")
print(f"  <split>_labels.npy    - Genre labels")
print(f"  <split>_files.txt     - File names reference")
print(f"\nExtraction info saved: {PROCESSED_DIR / 'extraction_info.json'}")
print("\n" + "=" * 70)
print("Ready for model training! 🚀")
print("=" * 70)