import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from dataset_builder import TabImageDataset
from config import IMG_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, OUT_DIR, CSV_FILE
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from quant_image import build_and_save_images

# ----------------------------
# Data split
# ----------------------------
def train_val_split_meta(meta_csv, test_size=0.2, random_state=42):
    meta = pd.read_csv(meta_csv)
    train, val = train_test_split(meta, test_size=test_size, stratify=meta['label'], random_state=random_state)
    train.to_csv(meta_csv.replace(".csv","_train.csv"), index=False)
    val.to_csv(meta_csv.replace(".csv","_val.csv"), index=False)
    return train, val


def build_model(name="resnet18", num_classes=2):
    if name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def train_loop(model, dl_train, dl_val, device=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE, save_path=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val = 0
    history = []
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(dl_train, desc=f"Epoch {epoch+1}/{epochs} train"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # val
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                gts.extend(y.numpy().tolist())
        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds)
        print(f"Epoch {epoch+1}: val_acc={acc:.4f}, val_f1={f1:.4f}")
        history.append({'epoch':epoch+1, 'acc':acc, 'f1':f1})
        if acc > best_val:
            best_val = acc
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
    return history

def build_dataloaders(meta_dir, img_size=IMG_SIZE):
    meta_csv = os.path.join(meta_dir, "meta.csv")
    train_csv = meta_csv.replace(".csv","_train.csv")
    val_csv = meta_csv.replace(".csv","_val.csv")
    if not os.path.exists(train_csv):
        train, val = train_val_split_meta(meta_csv)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    ds_train = TabImageDataset(train_csv, transform=transform)
    ds_val = TabImageDataset(val_csv, transform=transform)
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
                    "best_f1": best_epoch['f1']
                })

    # збереження результатів
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "cnn_experiments.csv"), index=False)
    print("\n=== Finished. Results saved to outputs/cnn_experiments.csv ===")
