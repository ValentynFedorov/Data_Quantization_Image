# src/dataset_builder.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import pandas as pd

class TabImageDataset(Dataset):
    def __init__(self, meta_csv, transform=None):
        self.meta = pd.read_csv(meta_csv)
        self.transform = transform or T.Compose([T.ToTensor()])
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = Image.open(row['img_path']).convert('L')
        img = self.transform(img)
        y = row['label'] if 'label' in row.index and not pd.isna(row['label']) else -1
        return img, int(y)

if __name__ == "__main__":
    pass
