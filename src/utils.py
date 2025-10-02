# src/utils.py
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from config import QUANT_LEVELS, PAD_TO, IMG_SIZE
import matplotlib.pyplot as plt

def load_tabular(csv_path):
    df = pd.read_csv(csv_path)
    return df

def normalize_and_quantize(X, n_levels=QUANT_LEVELS):
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X.astype(float))
    Xq = (Xs * (n_levels - 1)).round().astype(int)
    return Xq, scaler

def pad_row_to_length(row, length=PAD_TO, pad_value=0):
    if len(row) >= length:
        return row[:length]
    else:
        pad = np.full(length - len(row), pad_value, dtype=row.dtype)
        return np.concatenate([row, pad])

def row_to_image(row, img_size=IMG_SIZE):
    arr = pad_row_to_length(np.asarray(row).flatten(), img_size*img_size)
    img = arr.reshape((img_size, img_size)).astype('uint8')
    return img

def save_image(img, path, cmap='gray'):
    plt.imsave(path, img, cmap=cmap)
