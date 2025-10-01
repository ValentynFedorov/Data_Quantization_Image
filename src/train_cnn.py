# src/train_cnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from dataset_builder import TabImageDataset
from config import IMG_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from config import OUT_DIR

def train_val_split_meta(meta_csv, test_size=0.2, random_state=42):
    meta = pd.read_csv(meta_csv)
    train, val = train_test_split(meta, test_size=test_size, stratify=meta['label'], random_state=random_state)
    train.to_csv(meta_csv.replace(".csv","_train.csv"), index=False)
    val.to_csv(meta_csv.replace(".csv","_val.csv"), index=False)
    return train, val

def build_model(num_classes=2):
    # use small pretrained resnet (grayscale -> 3 channel trick: repeat channel)
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_loop(model, dl_train, dl_val, device=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val = 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y in tqdm(dl_train, desc=f"Epoch {epoch+1}/{epochs} train"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_losses.append(loss.item())
        # val
        model.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device)
                logits = model(x.to(device))
                probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                pred = logits.argmax(dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                gts.extend(y.numpy().tolist())
        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds)
        print(f"Epoch {epoch+1}: val_acc={acc:.4f}, val_f1={f1:.4f}")
        history.append({'epoch':epoch+1, 'acc':acc, 'f1':f1})
        if acc > best_val:
            best_val = acc
            save_file = os.path.join(OUT_DIR, "best_resnet.pth")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)  # створюємо директорію, якщо її немає
            torch.save(model.state_dict(), save_file)
    return history

def build_dataloaders(meta_dir):
    meta_csv = os.path.join(meta_dir, "meta.csv")
    train_csv = meta_csv.replace(".csv","_train.csv")
    val_csv = meta_csv.replace(".csv","_val.csv")
    if not os.path.exists(train_csv):
        train, val = train_val_split_meta(meta_csv)
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ])
    ds_train = TabImageDataset(train_csv, transform=transform)
    ds_val = TabImageDataset(val_csv, transform=transform)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return dl_train, dl_val

if __name__=="__main__":
    meta_csv = os.path.join(IMG_DIR, "meta.csv")
    dl_train, dl_val = build_dataloaders(IMG_DIR)
    model = build_model(num_classes=2)
    train_loop(model, dl_train, dl_val)
