"""
Training Script for CNN-BiGRU Model (Model 1) - WITH WINDOWING
Implements sliding window segmentation and SpecAugment for better performance
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import time
import random

# Check for MPS (Apple Silicon) support
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Configuration
BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed" / "melspectrogram"
MODEL_DIR = BASE_DIR / "models" / "cnn_bigru"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Windowing parameters
WINDOW_SIZE = 130  # 3 seconds worth of frames
OVERLAP = 0.5  # 50% overlap
HOP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 65 frames

# Training hyperparameters
BATCH_SIZE = 64  # Increased since we have more samples now
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LR_PATIENCE = 7
LR_FACTOR = 0.5
GRAD_CLIP = 1.0
NUM_CLASSES = 10

# Data augmentation parameters
FREQ_MASK_PARAM = 15
TIME_MASK_PARAM = 25
USE_AUGMENTATION = True

print("=" * 70)
print("CNN-BiGRU MODEL TRAINING - WITH WINDOWING")
print("=" * 70)
print(f"Device: {device}")
print(f"Window Size: {WINDOW_SIZE} frames (3 seconds)")
print(f"Overlap: {OVERLAP*100:.0f}% ({HOP_SIZE} frame hop)")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Max Epochs: {NUM_EPOCHS}")
print(f"Data Augmentation: {USE_AUGMENTATION}")


class SpecAugment:
    """SpecAugment: Simple data augmentation for spectrograms"""
    def __init__(self, freq_mask_param=15, time_mask_param=25):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
    
    def __call__(self, spec):
        """Apply frequency and time masking"""
        spec = spec.clone()
        n_mels, n_steps = spec.shape[1], spec.shape[2]
        
        # Frequency masking
        f = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, n_mels - f)
        spec[:, f0:f0+f, :] = 0
        
        # Time masking
        t = random.randint(0, self.time_mask_param)
        t0 = random.randint(0, n_steps - t)
        spec[:, :, t0:t0+t] = 0
        
        return spec


class WindowedGTZANDataset(Dataset):
    """Dataset with sliding window segmentation"""
    def __init__(self, features, labels, window_size=130, hop_size=65, 
                 augment=False, freq_mask=15, time_mask=25):
        self.features = features  # (N, H, W) numpy array
        self.labels = labels      # (N,) numpy array
        self.window_size = window_size
        self.hop_size = hop_size
        self.augment = augment
        
        if augment:
            self.spec_augment = SpecAugment(freq_mask, time_mask)
        
        # Pre-compute window indices for each sample
        self.window_indices = []
        for idx in range(len(features)):
            width = features[idx].shape[1]
            num_windows = max(1, (width - window_size) // hop_size + 1)
            
            for w in range(num_windows):
                start = w * hop_size
                end = start + window_size
                
                # Skip if window exceeds bounds
                if end > width:
                    continue
                    
                self.window_indices.append((idx, start, end))
        
        print(f"  Created {len(self.window_indices)} windows from {len(features)} samples")
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        sample_idx, start, end = self.window_indices[idx]
        
        # Extract window
        window = self.features[sample_idx, :, start:end]
        label = self.labels[sample_idx]
        
        # Convert to tensor and add channel dimension
        window = torch.FloatTensor(window).unsqueeze(0)  # (1, H, W)
        label = torch.LongTensor([label])[0]
        
        # Apply augmentation during training
        if self.augment:
            window = self.spec_augment(window)
        
        return window, label


class CNNBiGRU(nn.Module):
    """CNN-BiGRU architecture for music genre classification"""
    def __init__(self, num_classes=10):
        super(CNNBiGRU, self).__init__()
        
        # CNN Component - 5 convolutional blocks
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.3),
            
            # Block 5
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # Pool frequency to 1
            nn.Dropout2d(0.3),
        )
        
        # BiGRU Component
        self.bigru1 = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(256, 128, batch_first=True, bidirectional=True)
        self.gru3 = nn.GRU(256, 64, batch_first=True, bidirectional=False)
        
        # Feature expansion layer (for fusion later)
        self.feature_expand = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, return_features=False):
        # CNN feature extraction
        x = self.conv_blocks(x)  # (batch, 128, 1, time)
        
        # Reshape for RNN: (batch, channels, 1, time) -> (batch, time, channels)
        batch_size = x.size(0)
        x = x.squeeze(2)  # Remove height dimension: (batch, 128, time)
        x = x.permute(0, 2, 1)  # (batch, time, 128)
        
        # BiGRU layers
        x, _ = self.bigru1(x)  # (batch, time, 256)
        x, _ = self.bigru2(x)  # (batch, time, 256)
        x, h = self.gru3(x)    # h: (1, batch, 64)
        
        # Use final hidden state
        features = h.squeeze(0)  # (batch, 64)
        
        # Expand features to 256-dim
        expanded_features = self.feature_expand(features)  # (batch, 256)
        
        if return_features:
            return expanded_features
        
        # Classification
        output = self.classifier(expanded_features)
        
        return output


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# Load data
print("\nLoading data...")
train_features = np.load(DATA_DIR / "train_features.npy")
train_labels = np.load(DATA_DIR / "train_labels.npy")
val_features = np.load(DATA_DIR / "val_features.npy")
val_labels = np.load(DATA_DIR / "val_labels.npy")

print(f"Original data shapes:")
print(f"  Train: {train_features.shape}, {train_labels.shape}")
print(f"  Val: {val_features.shape}, {val_labels.shape}")

# Create windowed datasets
print(f"\nCreating windowed datasets...")
print(f"  Training set (with augmentation):")
train_dataset = WindowedGTZANDataset(
    train_features, train_labels, 
    window_size=WINDOW_SIZE, 
    hop_size=HOP_SIZE,
    augment=USE_AUGMENTATION,
    freq_mask=FREQ_MASK_PARAM,
    time_mask=TIME_MASK_PARAM
)

print(f"  Validation set (no augmentation):")
val_dataset = WindowedGTZANDataset(
    val_features, val_labels, 
    window_size=WINDOW_SIZE, 
    hop_size=HOP_SIZE,
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Initialize model
model = CNNBiGRU(num_classes=NUM_CLASSES).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE
)

# Training loop
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

best_val_acc = 0.0
patience_counter = 0
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'lr': []
}

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] LR: {current_lr:.6f}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, MODEL_DIR / "cnn_bigru_best.pth")
        print(f">>> Best model saved! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break

# Save final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
    'val_loss': val_loss,
}, MODEL_DIR / "cnn_bigru_final.pth")

# Save training history
with open(MODEL_DIR / "training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

elapsed_time = time.time() - start_time
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Total time: {elapsed_time/60:.2f} minutes")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Models saved to: {MODEL_DIR}")
print(f"Expected performance: 75-85% (windowed)")
print("=" * 70)