import torch
from torch.utils.data import Dataset

import pandas as pd
import os
from PIL import Image

def setKaiming_(layer,alpha=0.0):
    torch.nn.init.kaiming_normal_(layer.weight,alpha)
    torch.nn.init.zeros_(layer.bias)
    return layer

def setXavier_(layer):
    torch.nn.init.xavier_normal_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer

class ImageLoader(Dataset):
    def __init__(self, imgs_path:str, metadata_path:str, transforms=None):
        super().__init__()
        self.imgs_path = imgs_path
        self.df = pd.read_csv(metadata_path,)
        self.df = self.df.loc[:,["isic_id","image_type","diagnosis_1"]]
        self.df["diagnosis_1"] = self.df["diagnosis_1"].apply(lambda x: 1 if x == "Malignant" else 0)
        self.df["image_type"] = self.df["image_type"].apply(lambda x: 1 if x == "dermoscopic" else 0)
    
    def __getitem__(self, idx):
        return Image.open(self.imgs_path+self.df.loc[idx,"isic_id"]+".jpg")
    def __len__(self):
        return len(self.df)

if __name__ == "__main__":
    data = ImageLoader(metadata_path="/storage/SSD1/.data/milk10k/metadata.csv")