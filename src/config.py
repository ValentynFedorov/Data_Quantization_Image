# src/config.py
import torch
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
ORIG_DIR = os.path.join(DATA_DIR, "original")
GEN_DIR = os.path.join(DATA_DIR, "generated")
IMG_DIR = os.path.join(GEN_DIR, "images")
OUT_DIR = os.path.join(ROOT, "outputs")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(ORIG_DIR, "student_dropout.csv")


IMG_SIZE = 32       # default image size (k x k)
QUANT_LEVELS = 64  # number of quantization levels, e.g., 16,64,256
PAD_TO = IMG_SIZE*IMG_SIZE  # pad features to fill image

# training parameters
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

