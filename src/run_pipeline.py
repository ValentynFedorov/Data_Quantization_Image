# src/run_pipeline.py
import os
from config import CSV_FILE, IMG_DIR, QUANT_LEVELS
from quant_image import build_and_save_images
from train_baselines import run_tabular_baselines
from train_cnn import build_dataloaders, build_model, train_loop
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["all","images","baselines","cnn"], default="all")
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--quant", type=int, default=256)
    args = parser.parse_args()
    # generate images
    if args.step in ("all","images"):
        print("Step: build images")
        build_and_save_images(csv_path=CSV_FILE, out_dir=IMG_DIR, img_size=args.img_size, n_levels=args.quant)
    # run baselines
    if args.step in ("all","baselines"):
        print("Step: run tabular baselines")
        run_tabular_baselines()
    # train cnn
    if args.step in ("all","cnn"):
        print("Step: train cnn on images")
        dl_train, dl_val = build_dataloaders(IMG_DIR)
        model = build_model(num_classes=2)
        train_loop(model, dl_train, dl_val)

if __name__ == "__main__":
    main()
