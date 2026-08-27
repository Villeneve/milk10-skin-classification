#%%
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import v2
from torchvision.io import decode_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score, roc_curve, roc_auc_score
from seaborn import heatmap

import matplotlib.pyplot as plt

import numpy as np
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

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class ImageLoader(Dataset):
    def __init__(self, imgs_path, metadata_path, transforms=None, train=True):
        super().__init__()
        self.imgs_path = imgs_path
        self.transforms = transforms

        df = pd.read_csv(metadata_path)
        df = df[df["diagnosis_1"] != "Indeterminate"].copy()
        df["label"] = (df["diagnosis_1"] != "Benign").astype(int)

        derm = df[df["image_type"] == "dermoscopic"]
        clin = df[df["image_type"] != "dermoscopic"]
        pares = derm.merge(clin, on=["lesion_id", "label"], suffixes=("_d", "_c"))

        tr, te = train_test_split(
            pares, test_size=0.20, random_state=42, stratify=pares["label"]
        )
        self.df = (tr if train else te).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _abrir(self, nome):
        return decode_image(os.path.join(self.imgs_path, f"{nome}.jpg"),torchvision.io.ImageReadMode.RGB)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        derm = self._abrir(row["isic_id_d"])
        clin = self._abrir(row["isic_id_c"])

        if self.transforms:
            derm = self.transforms(derm)
            clin = self.transforms(clin)

        return derm, clin, torch.tensor(row["label"], dtype=torch.long)

def printImage():
    for img,imgType,label in tqdm(data):
        img = img[0].permute(1,2,0).cpu().numpy()/2+1/2
        plt.imshow(img)
        plt.title(f'type={imgType.numpy()}; diag={label.numpy()}')
        plt.show()
        break
    return

@torch.inference_mode()
def metrics(model:torch.nn.Module,dataset):
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
        all_outputs.append(torch.softmax(output,1))
    model.train(mode)
    all_labels = torch.cat(all_labels,0).cpu().numpy()
    all_outputs = torch.cat(all_outputs,0).cpu().numpy()
    matrix = confusion_matrix(all_labels,all_outputs.argmax(1))
    acc = accuracy_score(all_labels,all_outputs.argmax(1))
    f1 = f1_score(all_labels,all_outputs.argmax(1),average="macro")
    fpr,tpr,_ = roc_curve(all_labels,all_outputs[:,1])
    auc = roc_auc_score(all_labels,all_outputs[:,1])
    return acc,matrix,f1,[fpr,tpr],auc


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
