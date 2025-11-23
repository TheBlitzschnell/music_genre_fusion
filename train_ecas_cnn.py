"""
ECAS-CNN Training (Model 3)
Efficient Channel Attention CNN for MFCC features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed" / "mfcc_normalized"
MODEL_DIR = BASE_DIR / "models" / "ecas_cnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
PATIENCE = 25
NUM_CLASSES = 10

print("=" * 70)
print("ECAS-CNN TRAINING (MODEL 3)")
print("=" * 70)
print(f"Device: {device}")


class ECAModule(nn.Module):
    """Efficient Channel Attention Module"""
    def __init__(self, channels, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling
        y = self.avg_pool(x)
        
        # 1D convolution along channel dimension
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid activation
        y = self.sigmoid(y)
        
        # Multiply attention weights
        return x * y.expand_as(x)


class ECASCNN(nn.Module):
    """ECAS-CNN: CNN with Efficient Channel Attention for MFCC"""
    def __init__(self, num_classes=10):
        super(ECASCNN, self).__init__()
        
        # Block 1: 13×130 -> 6×65
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.eca1 = ECAModule(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        
        # Block 2: 6×65 -> 3×32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.eca2 = ECAModule(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        
        # Block 3: 3×32 -> 1×16
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.eca3 = ECAModule(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.3)
        
        # Block 4: 1×16 -> 1×8
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.eca4 = ECAModule(256)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.drop4 = nn.Dropout2d(0.3)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection for fusion (256 -> 128)
        self.feature_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, return_features=False):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.eca1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.eca2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.eca3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.eca4(x)
        x = self.pool4(x)
        x = self.drop4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Project to 128-dim for fusion
        features = self.feature_proj(x)
        
        if return_features:
            return features
        
        # Classification
        x = self.classifier(features)
        return x


class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 
                         'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 
                            'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(loader), 100. * correct / total


# Load data
print("\nLoading MFCC data...")
train_X = np.load(DATA_DIR / "train_features.npy")
train_y = np.load(DATA_DIR / "train_labels.npy")
val_X = np.load(DATA_DIR / "val_features.npy")
val_y = np.load(DATA_DIR / "val_labels.npy")

print(f"Train: {train_X.shape}")
print(f"Val: {val_X.shape}")

train_dataset = SimpleDataset(train_X, train_y)
val_dataset = SimpleDataset(val_X, val_y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = ECASCNN(NUM_CLASSES).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

# Training loop
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

best_val_acc = 0.0
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, MODEL_DIR / "ecas_cnn_best.pth")
        print(f">>> Best model saved! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

torch.save(model.state_dict(), MODEL_DIR / "ecas_cnn_final.pth")

with open(MODEL_DIR / "history.json", 'w') as f:
    json.dump(history, f, indent=2)

elapsed = (time.time() - start_time) / 60
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Time: {elapsed:.1f} minutes")
print(f"Best Val Acc: {best_val_acc:.2f}%")
print(f"Models saved to: {MODEL_DIR}")
print("\n" + "=" * 70)
print("ALL 3 BASE MODELS TRAINED!")
print("=" * 70)
print("Next step: Extract features and train fusion model")
print("=" * 70)