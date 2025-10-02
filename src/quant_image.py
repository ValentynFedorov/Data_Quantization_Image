# src/quant_image.py
import os
import numpy as np
import pandas as pd
from config import CSV_FILE, IMG_DIR, IMG_SIZE, QUANT_LEVELS, PAD_TO
from utils import normalize_and_quantize, row_to_image, save_image
from tqdm import tqdm


def build_and_save_images(csv_path=CSV_FILE, out_dir=IMG_DIR, img_size=IMG_SIZE, n_levels=QUANT_LEVELS):
    df = pd.read_csv(csv_path, sep=";")

    if 'target' in df.columns:
        y = df['target']
        X = df.drop(columns=['target'])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]


    y = y.astype(str).str.strip()
    y_bin = y.apply(lambda v: 1 if v.lower() == "dropout" else 0).values

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values

    Xq, scaler = normalize_and_quantize(X, n_levels)

    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for i, row in enumerate(Xq):
        img = row_to_image(row, img_size)
        path = os.path.join(out_dir, f"rec_{i}.png")
        save_image(img, path)
        meta.append({'id': i, 'img_path': path, 'label': int(y_bin[i])})
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    return meta_df, scaler


if __name__ == "__main__":
    print("Generating quantized images...")
    build_and_save_images()
