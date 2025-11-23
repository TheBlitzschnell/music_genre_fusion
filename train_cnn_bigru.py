"""
Training Script for CNN-BiGRU Model (Model 1)
Processes Mel-spectrograms for music genre classification
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

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.5
GRAD_CLIP = 1.0
NUM_CLASSES = 10

print("=" * 70)
print("CNN-BiGRU MODEL TRAINING")
print("=" * 70)
print(f"Device: {device}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Max Epochs: {NUM_EPOCHS}")


# Dataset class
class GTZANDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Add channel dimension: (H, W) -> (1, H, W)
        feature = self.features[idx].unsqueeze(0)
        label = self.labels[idx]
        return feature, label


# CNN-BiGRU Architecture
class CNNBiGRU(nn.Module):
    def __init__(self, input_height=128, num_classes=10):
        super(CNNBiGRU, self).__init__()
        
        # CNN Component (5 layers) with asymmetric pooling
        # Pool in frequency dimension (height) more than time dimension (width)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d((2, 2))  # Reduce both dims
        self.drop1 = nn.Dropout2d(0.25)
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))  # Reduce both dims
        self.drop2 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))  # Reduce both dims
        self.drop3 = nn.Dropout2d(0.25)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d((4, 1))  # Pool only in frequency to preserve time
        self.drop4 = nn.Dropout2d(0.25)
        
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d((4, 1))  # Pool only in frequency to preserve time
        self.drop5 = nn.Dropout2d(0.25)
        
        self.relu = nn.ReLU()
        
        # After pooling: height will be 128//(2*2*2*4*4)=1, width will be 130//(2*2*2*1*1)=16
        # So features per timestep: 128 channels * 1 height = 128
        
        # BiGRU Component (3 layers)
        self.bigru1 = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(256, 128, batch_first=True, bidirectional=True)
        self.gru3 = nn.GRU(256, 64, batch_first=True, bidirectional=False)
        
        # Feature expansion layer (for fusion later)
        self.feature_expand = nn.Linear(64, 256)
        
        # Classification head
        self.fc1 = nn.Linear(256, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x, return_features=False):
        # CNN layers
        x = self.drop1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(self.relu(self.bn4(self.conv4(x)))))
        x = self.drop5(self.pool5(self.relu(self.bn5(self.conv5(x)))))
        
        # Reshape for RNN: (batch, channels, height, width) -> (batch, width, channels*height)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.contiguous().view(batch_size, width, channels * height)
        
        # BiGRU layers
        x, _ = self.bigru1(x)  # (batch, width, 256)
        x, _ = self.bigru2(x)  # (batch, width, 256)
        x, h = self.gru3(x)    # (batch, width, 64), h: (1, batch, 64)
        
        # Use final hidden state
        features = h.squeeze(0)  # (batch, 64)
        
        # Expand features to 256-dim for fusion
        expanded_features = self.relu(self.feature_expand(features))
        
        if return_features:
            return expanded_features
        
        # Classification
        x = self.dropout_fc(self.relu(self.fc1(expanded_features)))
        x = self.fc2(x)
        
        return x


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

print(f"Train set: {train_features.shape}, {train_labels.shape}")
print(f"Val set: {val_features.shape}, {val_labels.shape}")

# Create datasets and dataloaders
train_dataset = GTZANDataset(train_features, train_labels)
val_dataset = GTZANDataset(val_features, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Initialize model
input_height = train_features.shape[1]  # Should be 128
model = CNNBiGRU(input_height=input_height, num_classes=NUM_CLASSES).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
print(f"Expected performance: 85-89%")
print("=" * 70)