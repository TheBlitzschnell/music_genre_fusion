"""
Advanced CNN Training - Aggressive Techniques for 90%+ Accuracy
Implements: Heavy augmentation, ResNet, Mixup, TTA, Progressive training
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
import random

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed" / "melspectrogram_normalized"
MODEL_DIR = BASE_DIR / "models" / "cnn_advanced"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    print("ERROR: Run diagnose_and_normalize_data.py first!")
    exit(1)

# Aggressive training configuration
BATCH_SIZE = 16  # Smaller for better gradient estimates
LEARNING_RATE = 0.001
NUM_EPOCHS = 200  # Much longer training
PATIENCE = 40
NUM_CLASSES = 10
MIXUP_ALPHA = 0.2
USE_MIXUP = True
USE_TTA = True  # Test-time augmentation

print("=" * 70)
print("ADVANCED CNN TRAINING - 90%+ TARGET")
print("=" * 70)
print(f"Device: {device}")
print(f"Mixup: {USE_MIXUP}")
print(f"Test-Time Augmentation: {USE_TTA}")
print(f"Max Epochs: {NUM_EPOCHS}")


class HeavyAugmentation:
    """Aggressive data augmentation"""
    def __init__(self, training=True):
        self.training = training
    
    def spec_augment(self, spec):
        """SpecAugment with multiple masks"""
        spec = spec.clone()
        _, n_mels, n_steps = spec.shape
        
        if self.training:
            # Frequency masking (2 masks)
            for _ in range(2):
                f = random.randint(15, 25)
                f0 = random.randint(0, max(1, n_mels - f))
                spec[:, f0:f0+f, :] = spec.mean()
            
            # Time masking (2 masks)
            for _ in range(2):
                t = random.randint(10, 20)
                t0 = random.randint(0, max(1, n_steps - t))
                spec[:, :, t0:t0+t] = spec.mean()
        
        return spec
    
    def __call__(self, spec):
        # Always apply SpecAugment during training
        if self.training:
            spec = self.spec_augment(spec)
            
            # Random noise injection
            if random.random() < 0.3:
                noise = torch.randn_like(spec) * 0.01
                spec = spec + noise
            
            # Random scaling
            if random.random() < 0.3:
                scale = random.uniform(0.8, 1.2)
                spec = spec * scale
        
        return spec


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AugmentedDataset(Dataset):
    def __init__(self, features, labels, training=True):
        self.features = features
        self.labels = labels
        self.training = training
        self.augment = HeavyAugmentation(training=training)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        y = self.labels[idx]
        
        # Apply augmentation
        x = self.augment(x)
        
        return x, y


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out


class AdvancedCNN(nn.Module):
    """ResNet-style architecture for music classification"""
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def train_epoch(model, loader, criterion, optimizer, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply mixup
        if use_mixup and random.random() < 0.5:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
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
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(loader), 100. * correct / total


def validate_with_tta(model, loader, criterion, n_tta=5):
    """Validation with test-time augmentation"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if USE_TTA:
                # Multiple forward passes with augmentation
                tta_outputs = []
                for _ in range(n_tta):
                    # Apply random augmentation
                    aug_inputs = inputs.clone()
                    if random.random() < 0.5:
                        # Horizontal flip in time
                        aug_inputs = torch.flip(aug_inputs, [3])
                    
                    outputs = model(aug_inputs)
                    tta_outputs.append(F.softmax(outputs, dim=1))
                
                # Average predictions
                outputs = torch.stack(tta_outputs).mean(0)
                outputs = torch.log(outputs)  # Convert back to logits
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    acc = 100. * correct / len(all_labels)
    
    return running_loss / len(loader), acc


# Load data
print("\nLoading data...")
train_X = np.load(DATA_DIR / "train_features.npy")
train_y = np.load(DATA_DIR / "train_labels.npy")
val_X = np.load(DATA_DIR / "val_features.npy")
val_y = np.load(DATA_DIR / "val_labels.npy")

print(f"Train: {train_X.shape}")
print(f"Val: {val_X.shape}")

# Create datasets with heavy augmentation
train_dataset = AugmentedDataset(train_X, train_y, training=True)
val_dataset = AugmentedDataset(val_X, val_y, training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=0, pin_memory=True)

# Model
model = AdvancedCNN(NUM_CLASSES).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

# Cosine annealing with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

# Training
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

best_val_acc = 0.0
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] LR: {current_lr:.6f}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                        optimizer, use_mixup=USE_MIXUP)
    val_loss, val_acc = validate_with_tta(model, val_loader, criterion, 
                                          n_tta=5 if USE_TTA else 1)
    
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
            'val_loss': val_loss,
        }, MODEL_DIR / "best_model.pth")
        print(f">>> BEST MODEL SAVED! Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
    
    # Save checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
        }, MODEL_DIR / f"checkpoint_epoch_{epoch+1}.pth")
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# Save final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'val_acc': val_acc,
}, MODEL_DIR / "final_model.pth")

with open(MODEL_DIR / "history.json", 'w') as f:
    json.dump(history, f, indent=2)

elapsed = (time.time() - start_time) / 60
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Training time: {elapsed:.1f} minutes")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Models saved to: {MODEL_DIR}")

if best_val_acc >= 90:
    print("\nSUCCESS! Achieved 90%+ accuracy target!")
elif best_val_acc >= 85:
    print("\nGood result! 85%+ is strong for a single model.")
    print("Consider ensemble of 3 models for 90%+")
else:
    print(f"\nCurrent: {best_val_acc:.2f}%")
    print("Recommendation: Train 3 separate models and ensemble them")

print("=" * 70)