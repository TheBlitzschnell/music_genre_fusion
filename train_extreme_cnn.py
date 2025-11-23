"""
EXTREME Training Strategy for 90%+ Single Model Accuracy
Key: Generate 10,000+ samples through aggressive windowing + heavy augmentation
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
import random

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed" / "melspectrogram_normalized"
MODEL_DIR = BASE_DIR / "models" / "cnn_extreme"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# EXTREME windowing configuration
WINDOW_SIZE = 64  # 1.5 seconds instead of 3 seconds
STRIDE = 16       # 75% overlap creates 4x more windows
BATCH_SIZE = 128  # Much larger batch size with small windows
LEARNING_RATE = 0.001
NUM_EPOCHS = 300
PATIENCE = 50
NUM_CLASSES = 10

print("=" * 70)
print("EXTREME TRAINING - TARGET 90%+")
print("=" * 70)
print(f"Window size: {WINDOW_SIZE} frames (1.5 seconds)")
print(f"Stride: {STRIDE} frames (75% overlap)")
print(f"Device: {device}")


class ExtremeMixupAugmentation:
    """Super heavy augmentation pipeline"""
    def __init__(self, training=True):
        self.training = training
    
    def time_shift(self, spec, max_shift=8):
        """Circular time shift"""
        if not self.training or random.random() > 0.5:
            return spec
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(spec, shifts=shift, dims=2)
    
    def freq_shift(self, spec, max_shift=8):
        """Circular frequency shift"""
        if not self.training or random.random() > 0.5:
            return spec
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(spec, shifts=shift, dims=1)
    
    def spec_augment_heavy(self, spec):
        """Multiple SpecAugment masks"""
        if not self.training:
            return spec
        
        spec = spec.clone()
        _, n_mels, n_steps = spec.shape
        
        # 3 frequency masks
        for _ in range(3):
            if random.random() < 0.8:
                f = random.randint(8, 20)
                f0 = random.randint(0, max(1, n_mels - f))
                spec[:, f0:f0+f, :] = 0
        
        # 3 time masks  
        for _ in range(3):
            if random.random() < 0.8:
                t = random.randint(5, 15)
                t0 = random.randint(0, max(1, n_steps - t))
                spec[:, :, t0:t0+t] = 0
        
        return spec
    
    def random_gain(self, spec):
        """Random volume adjustment"""
        if not self.training or random.random() > 0.5:
            return spec
        gain = random.uniform(0.7, 1.3)
        return spec * gain
    
    def add_noise(self, spec):
        """Gaussian noise injection"""
        if not self.training or random.random() > 0.3:
            return spec
        noise = torch.randn_like(spec) * 0.02
        return spec + noise
    
    def __call__(self, spec):
        if self.training:
            spec = self.time_shift(spec)
            spec = self.freq_shift(spec)
            spec = self.spec_augment_heavy(spec)
            spec = self.random_gain(spec)
            spec = self.add_noise(spec)
        return spec


class ExtremeWindowDataset(Dataset):
    """Create massive dataset through extreme windowing"""
    def __init__(self, features, labels, window_size=64, stride=16, training=True):
        self.window_size = window_size
        self.stride = stride
        self.training = training
        self.augment = ExtremeMixupAugmentation(training=training)
        
        # Generate all windows
        self.windows = []
        self.window_labels = []
        
        for idx in range(len(features)):
            spec = features[idx]  # (128, 130)
            label = labels[idx]
            
            # Extract windows with stride
            for start in range(0, spec.shape[1] - window_size + 1, stride):
                window = spec[:, start:start+window_size]
                self.windows.append(window)
                self.window_labels.append(label)
        
        self.windows = np.array(self.windows)
        self.window_labels = np.array(self.window_labels)
        
        print(f"  Created {len(self.windows)} windows from {len(features)} samples")
        print(f"  Ratio: {len(self.windows) / len(features):.1f}x increase")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx]).unsqueeze(0)
        label = self.window_labels[idx]
        
        # Apply augmentation
        window = self.augment(window)
        
        return window, label


def mixup_data(x, y, alpha=0.3):
    """Mixup augmentation"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WideResNet(nn.Module):
    """Wide ResNet for better feature learning"""
    def __init__(self, num_classes=10):
        super(WideResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Wide residual blocks
        self.layer1 = self._make_layer(64, 128, blocks=3, stride=2)
        self.layer2 = self._make_layer(128, 256, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, 512, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, 512, blocks=3, stride=2)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        
        # First block with stride
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Residual blocks
        for _ in range(blocks - 1):
            layers.append(ResidualBlock(out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Attention-weighted pooling
        att = self.attention(x)
        x = x * att
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


def train_epoch(model, loader, criterion, optimizer, use_mixup=True):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply mixup 70% of the time
        if use_mixup and random.random() < 0.7:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.3)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
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

print(f"Original shapes: Train {train_X.shape}, Val {val_X.shape}")

# Create extreme windowed datasets
print(f"\nGenerating extreme windowed dataset...")
print("Training set (with heavy augmentation):")
train_dataset = ExtremeWindowDataset(train_X, train_y, WINDOW_SIZE, STRIDE, training=True)

print("Validation set (no augmentation):")
val_dataset = ExtremeWindowDataset(val_X, val_y, WINDOW_SIZE, STRIDE, training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

# Model
model = WideResNet(NUM_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: WideResNet")
print(f"Parameters: {total_params:,}")

# Training setup
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE, 
    epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
    pct_start=0.3, anneal_strategy='cos'
)

# Training loop
print("\n" + "=" * 70)
print("STARTING EXTREME TRAINING")
print("=" * 70)

best_val_acc = 0.0
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

import time
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, use_mixup=True)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
        }, MODEL_DIR / "best_model.pth")
        print(f">>> BEST MODEL! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# Save final
torch.save(model.state_dict(), MODEL_DIR / "final_model.pth")

with open(MODEL_DIR / "history.json", 'w') as f:
    json.dump(history, f, indent=2)

elapsed = (time.time() - start_time) / 60
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Time: {elapsed:.1f} minutes")
print(f"Best Val Acc: {best_val_acc:.2f}%")

if best_val_acc >= 90:
    print("\nSUCCESS! Achieved 90%+ target!")
elif best_val_acc >= 85:
    print("\nGood! 85%+ is strong performance.")
    print("For 90%+: Increase window overlap or train longer")
else:
    print(f"\nCurrent: {best_val_acc:.2f}%")
    print("The extreme windowing should give major boost")

print(f"Saved to: {MODEL_DIR}")
print("=" * 70)