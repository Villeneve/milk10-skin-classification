#%%
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2

import matplotlib.pyplot as plt

import pandas as pd
from tqdm.notebook import tqdm
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
        self.transforms = transforms
        self.df = pd.read_csv(metadata_path,)
        self.df = self.df.loc[:,["isic_id","image_type","diagnosis_1"]]
        self.df["diagnosis_1"] = self.df["diagnosis_1"].apply(lambda x: 0 if x == "Benign" else 1)
        self.df["image_type"] = self.df["image_type"].apply(lambda x: 1 if x == "dermoscopic" else 0)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs_path+self.df.iloc[idx]["isic_id"]+".jpg")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, *self.df.iloc[idx][["image_type","diagnosis_1"]].tolist()
    def __len__(self):
        return len(self.df)

def printImage():
    for img,imgType,label in tqdm(data):
        img = img[0].permute(1,2,0).cpu().numpy()/2+1/2
        plt.imshow(img)
        plt.title(f'type={imgType.numpy()}; diag={label.numpy()}')
        plt.show()
        break
    return

if __name__ == "__main__":

    data = ImageLoader(
        imgs_path="/storage/SSD1/.data/milk10k/images/",
        metadata_path="/storage/SSD1/.data/milk10k/metadata.csv",
        transforms=v2.Compose([
            v2.Resize(256),
            v2.Pad(256//2,padding_mode='reflect'),
            v2.RandomRotation(180,interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop((256,256)),
            v2.ToImage(),
            v2.ToDtype(torch.float32,scale=True),
            v2.Normalize([.5]*3,[.5]*3)
        ])
    )
    # data = DataLoader(
    #     data,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=4,
    #     drop_last=True,
    #     pin_memory=True,
    # )

    print(data.df["diagnosis_1"].value_counts())
