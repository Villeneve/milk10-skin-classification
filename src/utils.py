#%%
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

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
    def __init__(self, imgs_path:str, metadata_path:str, transforms=None, train=True):
        super().__init__()
        self.imgs_path = imgs_path
        self.transforms = transforms
        self.train = train
        self.df = pd.read_csv(metadata_path,)
        self.df = self.df.loc[:,["isic_id","image_type","diagnosis_1"]]
        self.df = self.df[self.df['diagnosis_1']!="Indeterminate"]
        self.df["diagnosis_1"] = self.df["diagnosis_1"].apply(lambda x: 0 if x == "Benign" else 1)
        self.df["image_type"] = self.df["image_type"].apply(lambda x: 1 if x == "dermoscopic" else 0)
        self.df_train,self.df_test = train_test_split(
            self.df,
            test_size=.20,
            random_state=42,
            stratify=self.df["diagnosis_1"]
        )
    
    def __getitem__(self, idx):
        if self.train:
            img = Image.open(self.imgs_path+self.df_train.iloc[idx]["isic_id"]+".jpg")
            if self.transforms is not None:
                img = self.transforms(img)
            return img, *self.df_train.iloc[idx][["image_type","diagnosis_1"]].tolist()
        else:
            img = Image.open(self.imgs_path+self.df_test.iloc[idx]["isic_id"]+".jpg")
            if self.transforms is not None:
                img = self.transforms(img)
            return img, *self.df_test.iloc[idx][["image_type","diagnosis_1"]].tolist()

    def __len__(self):
        return len(self.df_train) if self.train else len(self.df_test)

    def counter(self):
        return self.df_train["diagnosis_1"].value_counts() if self.train else self.df_test["diagnosis_1"].value_counts()

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
            # v2.Pad(256//2,padding_mode='reflect'),
            # v2.RandomRotation(180,interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop((256,256)),
            v2.ToImage(),
            v2.ToDtype(torch.float32,scale=True),
            v2.Normalize([.5]*3,[.5]*3)
        ]),
        train=True
    )
    c = 0
    for image in tqdm(data):
        c += 1
    print(c)
    # data = DataLoader(
    #     data,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=4,
    #     drop_last=True,
    #     pin_memory=True,
    # )