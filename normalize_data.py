"""
Data Diagnostic and Normalization Script
Checks data quality and creates normalized versions
"""

import numpy as np
from pathlib import Path
import json

BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed"
MELSPEC_DIR = DATA_DIR / "melspectrogram"
MFCC_DIR = DATA_DIR / "mfcc"

print("=" * 70)
print("DATA DIAGNOSTIC AND NORMALIZATION")
print("=" * 70)

# Load mel-spectrogram data
print("\n[1/3] Loading Mel-spectrogram data...")
train_mel = np.load(MELSPEC_DIR / "train_features.npy")
val_mel = np.load(MELSPEC_DIR / "val_features.npy")
test_mel = np.load(MELSPEC_DIR / "test_features.npy")

print(f"Train shape: {train_mel.shape}")
print(f"Val shape: {val_mel.shape}")
print(f"Test shape: {test_mel.shape}")

# Analyze mel-spectrogram statistics
print("\n[2/3] Analyzing Mel-spectrogram statistics...")
print(f"\nBEFORE NORMALIZATION:")
print(f"  Train - Min: {train_mel.min():.2f}, Max: {train_mel.max():.2f}, Mean: {train_mel.mean():.2f}, Std: {train_mel.std():.2f}")
print(f"  Val   - Min: {val_mel.min():.2f}, Max: {val_mel.max():.2f}, Mean: {val_mel.mean():.2f}, Std: {val_mel.std():.2f}")
print(f"  Test  - Min: {test_mel.min():.2f}, Max: {test_mel.max():.2f}, Mean: {test_mel.mean():.2f}, Std: {test_mel.std():.2f}")

# Check for NaN or Inf
print(f"\nData quality check:")
print(f"  Train NaN: {np.isnan(train_mel).sum()}, Inf: {np.isinf(train_mel).sum()}")
print(f"  Val NaN: {np.isnan(val_mel).sum()}, Inf: {np.isinf(val_mel).sum()}")
print(f"  Test NaN: {np.isnan(test_mel).sum()}, Inf: {np.isinf(test_mel).sum()}")

# Normalize using training set statistics
print("\n[3/3] Normalizing data...")

# Compute global mean and std from training set
train_mean = train_mel.mean()
train_std = train_mel.std()

print(f"\nNormalization parameters (from training set):")
print(f"  Mean: {train_mean:.4f}")
print(f"  Std: {train_std:.4f}")

# Apply normalization
train_mel_norm = (train_mel - train_mean) / (train_std + 1e-8)
val_mel_norm = (val_mel - train_mean) / (train_std + 1e-8)
test_mel_norm = (test_mel - train_mean) / (train_std + 1e-8)

print(f"\nAFTER NORMALIZATION:")
print(f"  Train - Min: {train_mel_norm.min():.2f}, Max: {train_mel_norm.max():.2f}, Mean: {train_mel_norm.mean():.2f}, Std: {train_mel_norm.std():.2f}")
print(f"  Val   - Min: {val_mel_norm.min():.2f}, Max: {val_mel_norm.max():.2f}, Mean: {val_mel_norm.mean():.2f}, Std: {val_mel_norm.std():.2f}")
print(f"  Test  - Min: {test_mel_norm.min():.2f}, Max: {test_mel_norm.max():.2f}, Mean: {test_mel_norm.mean():.2f}, Std: {test_mel_norm.std():.2f}")

# Save normalized data
MELSPEC_NORM_DIR = DATA_DIR / "melspectrogram_normalized"
MELSPEC_NORM_DIR.mkdir(exist_ok=True)

np.save(MELSPEC_NORM_DIR / "train_features.npy", train_mel_norm)
np.save(MELSPEC_NORM_DIR / "val_features.npy", val_mel_norm)
np.save(MELSPEC_NORM_DIR / "test_features.npy", test_mel_norm)

# Copy labels
train_labels = np.load(MELSPEC_DIR / "train_labels.npy")
val_labels = np.load(MELSPEC_DIR / "val_labels.npy")
test_labels = np.load(MELSPEC_DIR / "test_labels.npy")

np.save(MELSPEC_NORM_DIR / "train_labels.npy", train_labels)
np.save(MELSPEC_NORM_DIR / "val_labels.npy", val_labels)
np.save(MELSPEC_NORM_DIR / "test_labels.npy", test_labels)

# Save normalization parameters
norm_params = {
    'mean': float(train_mean),
    'std': float(train_std),
    'epsilon': 1e-8
}

with open(MELSPEC_NORM_DIR / "normalization_params.json", 'w') as f:
    json.dump(norm_params, f, indent=2)

print(f"\nNormalized data saved to: {MELSPEC_NORM_DIR}")

# Now do the same for MFCC
print("\n" + "=" * 70)
print("MFCC NORMALIZATION")
print("=" * 70)

train_mfcc = np.load(MFCC_DIR / "train_features.npy")
val_mfcc = np.load(MFCC_DIR / "val_features.npy")
test_mfcc = np.load(MFCC_DIR / "test_features.npy")

print(f"\nBEFORE NORMALIZATION:")
print(f"  Train - Min: {train_mfcc.min():.2f}, Max: {train_mfcc.max():.2f}, Mean: {train_mfcc.mean():.2f}, Std: {train_mfcc.std():.2f}")

train_mean_mfcc = train_mfcc.mean()
train_std_mfcc = train_mfcc.std()

train_mfcc_norm = (train_mfcc - train_mean_mfcc) / (train_std_mfcc + 1e-8)
val_mfcc_norm = (val_mfcc - train_mean_mfcc) / (train_std_mfcc + 1e-8)
test_mfcc_norm = (test_mfcc - train_mean_mfcc) / (train_std_mfcc + 1e-8)

print(f"\nAFTER NORMALIZATION:")
print(f"  Train - Min: {train_mfcc_norm.min():.2f}, Max: {train_mfcc_norm.max():.2f}, Mean: {train_mfcc_norm.mean():.2f}, Std: {train_mfcc_norm.std():.2f}")

MFCC_NORM_DIR = DATA_DIR / "mfcc_normalized"
MFCC_NORM_DIR.mkdir(exist_ok=True)

np.save(MFCC_NORM_DIR / "train_features.npy", train_mfcc_norm)
np.save(MFCC_NORM_DIR / "val_features.npy", val_mfcc_norm)
np.save(MFCC_NORM_DIR / "test_features.npy", test_mfcc_norm)

train_labels = np.load(MFCC_DIR / "train_labels.npy")
val_labels = np.load(MFCC_DIR / "val_labels.npy")
test_labels = np.load(MFCC_DIR / "test_labels.npy")

np.save(MFCC_NORM_DIR / "train_labels.npy", train_labels)
np.save(MFCC_NORM_DIR / "val_labels.npy", val_labels)
np.save(MFCC_NORM_DIR / "test_labels.npy", test_labels)

norm_params_mfcc = {
    'mean': float(train_mean_mfcc),
    'std': float(train_std_mfcc),
    'epsilon': 1e-8
}

with open(MFCC_NORM_DIR / "normalization_params.json", 'w') as f:
    json.dump(norm_params_mfcc, f, indent=2)

print(f"\nNormalized MFCC data saved to: {MFCC_NORM_DIR}")

print("\n" + "=" * 70)
print("NORMALIZATION COMPLETE")
print("=" * 70)
print("\nNow re-run training with normalized data!")
print("The training script will automatically use the normalized data.")