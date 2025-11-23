"""
DPA-CNN Training (Model 2)
Dual Parallel Attention CNN for music genre classification
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
DATA_DIR = BASE_DIR / "data" / "processed" / "melspectrogram_normalized"
MODEL_DIR = BASE_DIR / "models" / "dpa_cnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
PATIENCE = 25
NUM_CLASSES = 10

print("=" * 70)
print("DPA-CNN TRAINING (MODEL 2)")
print("=" * 70)
print(f"Device: {device}")


class DPAModule(nn.Module):
    """Dual Parallel Attention: Channel + Spatial Attention"""
    def __init__(self, channels, reduction=8):
        super(DPAModule, self).__init__()
        
        # Parallel Channel Attention
        self.theta = nn.Conv2d(channels, channels // reduction, 1)
        self.phi = nn.Conv2d(channels, channels // reduction, 1)
        self.psi = nn.Conv2d(channels, channels // reduction, 1)
        self.g = nn.Conv2d(channels // reduction, channels, 3, padding=1)
        
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Channel Attention
        theta = self.theta(x).view(batch_size, -1, H * W)
        phi = self.phi(x).view(batch_size, -1, H * W)
        psi = self.psi(x).view(batch_size, -1, H * W)
        
        attention = torch.bmm(theta.permute(0, 2, 1), phi)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(psi, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, H, W)
        out = self.g(out)
        channel_att = torch.sigmoid(out)
        
        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_input))
        
        # Combine attentions
        x = x * channel_att * spatial_att
        
        return x


class DPACNN(nn.Module):
    """DPA-CNN: CNN with Dual Parallel Attention modules"""
    def __init__(self, num_classes=10):
        super(DPACNN, self).__init__()
        
        # Layer 1: Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # Layer 2: Conv + DPA
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dpa2 = DPAModule(128)
        self.avgpool2 = nn.AvgPool2d(2, 2)
        
        # Layer 3: Conv + DPA
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dpa3 = DPAModule(256)
        self.avgpool3 = nn.AvgPool2d(2, 2)
        
        # Layer 4: Conv + DPA
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dpa4 = DPAModule(256)
        self.avgpool4 = nn.AvgPool2d(2, 2)
        
        # Layer 5: Conv + DPA + Global pooling
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.dpa5 = DPAModule(256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection for fusion (256 -> 512)
        self.feature_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, return_features=False):
        # Layer 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Layer 2 with DPA
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dpa2(x)
        x = self.avgpool2(x)
        
        # Layer 3 with DPA
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dpa3(x)
        x = self.avgpool3(x)
        
        # Layer 4 with DPA
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dpa4(x)
        x = self.avgpool4(x)
        
        # Layer 5 with DPA
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dpa5(x)
        x = self.global_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Project to 512-dim for fusion
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
print("\nLoading data...")
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
model = DPACNN(NUM_CLASSES).to(device)
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
        }, MODEL_DIR / "dpa_cnn_best.pth")
        print(f">>> Best model saved! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

torch.save(model.state_dict(), MODEL_DIR / "dpa_cnn_final.pth")

with open(MODEL_DIR / "history.json", 'w') as f:
    json.dump(history, f, indent=2)

elapsed = (time.time() - start_time) / 60
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Time: {elapsed:.1f} minutes")
print(f"Best Val Acc: {best_val_acc:.2f}%")
print(f"Models saved to: {MODEL_DIR}")
print(f"Expected: 70-80% (target: 89-93% requires optimization)")
print("=" * 70)