#%%
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import v2
from torchvision.io import decode_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from seaborn import heatmap

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
            img = decode_image(self.imgs_path+self.df_train.iloc[idx]["isic_id"]+".jpg",mode=torchvision.io.ImageReadMode.RGB)
            if self.transforms is not None:
                img = self.transforms(img)
            return img, *self.df_train.iloc[idx][["image_type","diagnosis_1"]].tolist()
        else:
            img = decode_image(self.imgs_path+self.df_test.iloc[idx]["isic_id"]+".jpg",mode=torchvision.io.ImageReadMode.RGB)
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

@torch.inference_mode()
def accuracy(model:torch.nn.Module,dataset):
    acc = 0
    device = next(model.parameters()).device
    all_labels,all_outputs = [],[]
    mode = model.training
    model.eval()
    for img,_,label in dataset:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        all_labels.append(label)
        all_outputs.append(output.argmax(1))
    model.train(mode)
    all_labels = torch.cat(all_labels,0).cpu().numpy()
    all_outputs = torch.cat(all_outputs,0).cpu().numpy()
    matrix = confusion_matrix(all_labels,all_outputs)
    acc = balanced_accuracy_score(all_labels,all_outputs)
    f1 = f1_score(all_labels,all_outputs,average="macro")
    return acc*100,matrix,f1


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
    print(data.counter())
    # data = DataLoader(
    #     data,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=4,
    #     drop_last=True,
    #     pin_memory=True,
    # )
# %%
