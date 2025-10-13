import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from dataset_builder import TabImageDataset
from config import IMG_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, OUT_DIR, CSV_FILE
import os
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from quant_image import build_and_save_images
from thop import profile  # For FLOP counting (pip install thop)


def train_val_split_meta(meta_csv, test_size=0.2, random_state=42):
    meta = pd.read_csv(meta_csv)
    train, val = train_test_split(meta, test_size=test_size, stratify=meta['label'], random_state=random_state)
    train.to_csv(meta_csv.replace(".csv","_train.csv"), index=False)
    val.to_csv(meta_csv.replace(".csv","_val.csv"), index=False)
    return train, val


class RandomNoise(nn.Module):
    """Add random gaussian noise to images"""
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


def build_model(name="resnet18", num_classes=2, pretrained=False, freeze_backbone=False):
    """Build CNN model with optional pretraining and freezing"""
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # Adapt first layer for grayscale input
        if pretrained:
            # Average the RGB weights to create grayscale weights
            old_conv1 = model.conv1
            new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Average RGB channels for grayscale
            new_conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)
            model.conv1 = new_conv1
        else:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # Freeze backbone if requested
        if freeze_backbone and pretrained:
            for name_param, param in model.named_parameters():
                if 'fc' not in name_param:  # Don't freeze final classifier
                    param.requires_grad = False
                    
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        if pretrained:
            old_conv = model.features[0][0]
            new_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            model.features[0][0] = new_conv
        else:
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
        if freeze_backbone and pretrained:
            for name_param, param in model.named_parameters():
                if 'classifier' not in name_param:
                    param.requires_grad = False
                    
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        if pretrained:
            old_conv = model.features[0][0]
            new_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            model.features[0][0] = new_conv
        else:
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
        if freeze_backbone and pretrained:
            for name_param, param in model.named_parameters():
                if 'classifier' not in name_param:
                    param.requires_grad = False
                    
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
        
    return model


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_flops(model, input_shape=(1, 1, 32, 32)):
    """Count FLOPs using thop library"""
    try:
        dummy_input = torch.randn(*input_shape)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except Exception as e:
        print(f"FLOP counting failed: {e}")
        return None, None


def train_loop(model, dl_train, dl_val, device=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE, save_path=None, model_name="unknown"):
    """Enhanced training loop with detailed metrics collection"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Count parameters and FLOPs
    total_params, trainable_params = count_parameters(model)
    print(f"Model {model_name}: {total_params:,} total params, {trainable_params:,} trainable params")
    
    # Count FLOPs (use first batch to get input shape)
    try:
        sample_input = next(iter(dl_train))[0][:1]  # Get first sample
        flops, _ = count_flops(model, input_shape=sample_input.shape)
        if flops:
            print(f"FLOPs: {flops/1e6:.2f}M")
    except Exception as e:
        flops = None
        print(f"Could not calculate FLOPs: {e}")
    
    best_val = 0
    history = []
    total_train_time = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_start_time = time.time()
        
        train_loss = 0
        for x, y in tqdm(dl_train, desc=f"Epoch {epoch+1}/{epochs} train"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        
        epoch_train_time = time.time() - epoch_start_time
        total_train_time += epoch_train_time
        
        # Validation phase
        model.eval()
        preds, gts, probas = [], [], []
        val_loss = 0
        
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                
                pred = logits.argmax(dim=1).cpu().numpy()
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                
                preds.extend(pred.tolist())
                gts.extend(y.numpy().tolist())
                probas.extend(proba.tolist())
        
        # Calculate metrics
        acc = accuracy_score(gts, preds)
        f1_macro = f1_score(gts, preds, average='macro')
        f1_weighted = f1_score(gts, preds, average='weighted')
        
        # ROC-AUC (for binary classification)
        try:
            if len(np.unique(gts)) == 2:
                probas_np = np.array(probas)
                roc_auc = roc_auc_score(gts, probas_np[:, 1])
            else:
                roc_auc = None
        except Exception:
            roc_auc = None
        
        avg_train_loss = train_loss / len(dl_train)
        avg_val_loss = val_loss / len(dl_val)
        
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"val_acc={acc:.4f}, val_f1_macro={f1_macro:.4f}, val_f1_weighted={f1_weighted:.4f}"
              + (f", val_roc_auc={roc_auc:.4f}" if roc_auc else "") +
              f", time={epoch_train_time:.2f}s")
        
        history.append({
            'epoch': epoch+1, 
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'acc': acc, 
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'roc_auc': roc_auc,
            'epoch_time': epoch_train_time
        })
        
        if acc > best_val:
            best_val = acc
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
    
    # Add model info to history
    model_info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'total_train_time': total_train_time,
        'avg_epoch_time': total_train_time / epochs
    }
    
    return history, model_info

def build_dataloaders(meta_dir, img_size=IMG_SIZE, use_augmentation=False):
    """Build dataloaders with optional augmentation"""
    meta_csv = os.path.join(meta_dir, "meta.csv")
    train_csv = meta_csv.replace(".csv","_train.csv")
    val_csv = meta_csv.replace(".csv","_val.csv")
    if not os.path.exists(train_csv):
        train, val = train_val_split_meta(meta_csv)
    
    # Base transforms
    base_transform = [
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ]
    
    # Training transforms with optional augmentation
    if use_augmentation:
        train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomRotation(degrees=10),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            RandomNoise(std=0.01)  # Add noise after tensor conversion
        ])
    else:
        train_transform = T.Compose(base_transform)
    
    # Validation transforms (no augmentation)
    val_transform = T.Compose(base_transform)
    
    ds_train = TabImageDataset(train_csv, transform=train_transform)
    ds_val = TabImageDataset(val_csv, transform=val_transform)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return dl_train, dl_val


if __name__=="__main__":
    results = []
    models_to_try = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
    quant_levels = [16, 64, 256]
    img_sizes = [32, 64]

    for q in quant_levels:
        for img_size in img_sizes:
            print(f"\n### Building images (quant={q}, img_size={img_size}) ###")
            build_and_save_images(csv_path=CSV_FILE, out_dir=IMG_DIR, img_size=img_size, n_levels=q)
            for model_name in models_to_try:
                print(f"\n>>> Training {model_name} with quant={q}, img_size={img_size}")
                dl_train, dl_val = build_dataloaders(IMG_DIR, img_size=img_size)
                model = build_model(model_name, num_classes=2)
                save_path = os.path.join(OUT_DIR, f"{model_name}_q{q}_s{img_size}.pth")
                hist = train_loop(model, dl_train, dl_val, save_path=save_path)
                best_epoch = max(hist, key=lambda h: h['acc'])
                results.append({
                    "model": model_name,
                    "quant_levels": q,
                    "img_size": img_size,
                    "best_acc": best_epoch['acc'],
                    "best_f1": best_epoch['f1'],
                    "best_roc_auc": best_epoch['roc_auc'],
                })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "cnn_experiments.csv"), index=False)
    print("\n=== Finished. Results saved to outputs/cnn_experiments.csv ===")
