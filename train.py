#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchinfo import summary

from src.models import *
from src.utils import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from tqdm.autonotebook import tqdm

import os
import argparse

#%%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["inception","vgg16","resnet34"],
    required=True,
    help="Escolha o modelo para o classificador",
)
parser.add_argument(
    "-ep",
    "--epochs",
    type=int,
    default=100,
)
parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=32,
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=1e-3,
)
args = parser.parse_args()

#%%
gpu = torch.device("cuda:1")
model = get_model(args.model).to(gpu)
model.eval()
lce = torch.nn.CrossEntropyLoss(weight=torch.tensor([8187/(2*2373),8187/(2*5814)],device=gpu))
opt = torch.optim.Adam(
    model.parameters(),
    args.learning_rate,
)

#%%
data = ImageLoader(
        imgs_path="/storage/SSD1/.data/milk10k/images/",
        metadata_path="/storage/SSD1/.data/milk10k/metadata.csv",
        # transforms=v2.Compose([
        #     v2.Resize(256),
        #     # v2.Pad(256//2,padding_mode='reflect'),
        #     # v2.RandomRotation(180,interpolation=v2.InterpolationMode.BILINEAR),
        #     v2.CenterCrop((256,256)),
        #     v2.ToImage(),
        #     v2.ToDtype(torch.float32,scale=True),
        #     v2.Normalize([.5]*3,[.5]*3)
        # ]),
        transforms=model.transforms(),
        train=True
    )
data = DataLoader(
    data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_data = ImageLoader(
        imgs_path="/storage/SSD1/.data/milk10k/images/",
        metadata_path="/storage/SSD1/.data/milk10k/metadata.csv",
        # transforms=v2.Compose([
        #     v2.Resize(256),
        #     # v2.Pad(256//2,padding_mode='reflect'),
        #     # v2.RandomRotation(180,interpolation=v2.InterpolationMode.BILINEAR),
        #     v2.CenterCrop((256,256)),
        #     v2.ToImage(),
        #     v2.ToDtype(torch.float32,scale=True),
        #     v2.Normalize([.5]*3,[.5]*3)
        # ]),
        transforms=model.transforms(),
        train=False
    )
val_data = DataLoader(
    val_data,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=False
)

#%%
model.acc = None
model.loss = None
epoch_bar = tqdm(range(args.epochs),position=0)
best_acc = -1
model.eval()
for epoch in epoch_bar:
    batch_bar = tqdm(data,position=1,leave=False)

    if epoch == args.epochs//2:
        opt.param_groups[0]['lr'] = 1e-5
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        # print("\nNova etapa")

    for img,_,label in batch_bar:
        img = img.to(gpu)
        label = label.to(gpu)
        output = model(img)
        loss_ = lce(output,label)
        opt.zero_grad()
        loss_.backward()
        opt.step()

        acc_train_ = (label==output.argmax(1)).sum()/len(label)
        model.acc = acc_train_ if model.acc is None else .98*model.acc+(1-.98)*acc_train_
        model.loss = loss_.item() if model.loss is None else .98*model.loss+(1-.98)*loss_.item()
        batch_bar.set_postfix({
            "loss":f"{model.loss:.4f}",
            "acc":f"{model.acc*100:.2f}"
        })

    val_acc,matrix = accuracy(model,val_data,save=True)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),f'best_{args.model}.pt')
        heatmap(matrix,fmt='.2f',annot=True)
        plt.savefig(f'confusion_matrix_{args.model}.png')
        plt.close()
    epoch_bar.set_postfix({
        "val_acc":f"{val_acc:.2f}"
    })