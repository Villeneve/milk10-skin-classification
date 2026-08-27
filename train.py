#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchinfo import summary

from src.models import *
from src.utils import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from seaborn import heatmap
from tqdm.autonotebook import tqdm

import os
import argparse

#%%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["inception","vgg16","resnet34","efficient"],
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
parser.add_argument(
    '-gpu',
    type=int,
    default=0,
    required=True,
)
parser.add_argument(
    '--bs_val',
    type=int,
    default=64,
    required=False,
)
args = parser.parse_args()

#%% --------------------------------------------------------------------
### ---- Declara os modelos, otimizadores, augmentations e losses ------
### --------------------------------------------------------------------
gpu = torch.device(f"cuda:{args.gpu}")
model = get_model(args.model).to(gpu)
model.eval()
lce = torch.nn.CrossEntropyLoss(weight=torch.tensor([8187/(2*2373),8187/(2*5814)],device=gpu))
opt = torch.optim.AdamW(
    model.parameters(),
    args.learning_rate,
    weight_decay=1e-4
)
aug_transform=v2.Compose([
    v2.Resize(model.transforms().resize_size[0]),
    v2.ToDtype(torch.float32,scale=True),
    v2.Pad(int(model.transforms().crop_size[0]/2**.5-model.transforms().resize_size[0]/2+1),padding_mode='reflect'),
    v2.RandomRotation(180,interpolation=v2.InterpolationMode.BILINEAR),
    v2.CenterCrop((model.transforms().crop_size[0],model.transforms().crop_size[0])),
    v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
val_tf = v2.Compose([
    v2.Resize(model.transforms().resize_size[0]),
    v2.CenterCrop((model.transforms().crop_size[0],model.transforms().crop_size[0])),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

#%%
data = ImageLoader(
        imgs_path="/storage/SSD1/.data/milk10k/images/",
        metadata_path="/storage/SSD1/.data/milk10k/metadata.csv",
        train=True,
    )
data = DataLoader(
    data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_data = ImageLoader(
        imgs_path="/storage/SSD1/.data/milk10k/images/",
        metadata_path="/storage/SSD1/.data/milk10k/metadata.csv",
        train=False,
        transforms=val_tf
    )
val_data = DataLoader(
    val_data,
    batch_size=args.bs_val,
    shuffle=False,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=False,
    persistent_workers=True
)

#%%
acc = None
loss = None
epoch_bar = tqdm(range(args.epochs),position=0)
best_f1 = -1
model.eval()
acc
history = {
    "acc":[],
    "val_acc":[],
    "loss":[],
    "val_loss":[],
}
for epoch in epoch_bar:
    batch_bar = tqdm(data,position=1,leave=False)
    # print(torch.cuda.memory_summary(gpu))

    if epoch == args.epochs//3:
        opt.param_groups[0]['lr'] = 1e-4
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        # print("\nNova etapa")

    for derm,_,label in batch_bar:
        derm = derm.to(gpu, non_blocking=True)
        derm = aug_transform(derm)
        label = label.to(gpu, non_blocking=True)
        output = model(derm)
        loss_ = lce(output,label)
        opt.zero_grad()
        loss_.backward()
        opt.step()

        acc_train_ = (label==output.argmax(1)).sum().item()/len(label)
        acc = acc_train_ if acc is None else .98*acc+(1-.98)*acc_train_
        loss = loss_.item() if loss is None else .98*loss+(1-.98)*loss_.item()
        batch_bar.set_postfix({
            "loss":f"{loss:.4f}",
            "acc":f"{acc*100:.2f}"
        })

    val_acc,matrix,f1,roc,auc = metrics(model,val_data)
    history["acc"].append(acc)
    history["val_acc"].append(val_acc)

    # Save model logs and weights
    if f1 > best_f1:
        best_f1 = f1
        dict_save = {
            "model_name":args.model,
            "model":model.state_dict(),
            "opt":opt.state_dict(),
            "epochs":args.epochs,
            "acc":val_acc,
            "f1_score":f1,
            "matrix":matrix,
        }
        torch.save(dict_save,f'best_{args.model}.pt')

        # HeatMap
        annot_labels = np.empty_like(matrix,dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                annot_labels[i,j]=f"N: {matrix[i,j]}\nP: {matrix[i,j]/matrix.sum(0)[j]*100:.2f}\nR: {matrix[i,j]/matrix.sum(1)[i]*100:.2f}"
        heatmap(
            matrix/matrix.sum(1,keepdims=True)*100,
            fmt="",
            annot=annot_labels,
            xticklabels=["benign","malignant"],
            yticklabels=["benign","malignant"],
        )
        plt.xlabel("Predict"); plt.ylabel("Real")
        plt.title(f"F1 = {f1*100:.2f}%")
        plt.savefig(f'plots/confusion_matrix_{args.model}.png')
        plt.close()

        # Curva ROC
        fpr, tpr = roc
        plt.figure()
        plt.title(f"AUC = {auc*100:.2f}%")
        plt.plot(fpr,tpr)
        plt.plot([0,1],[0,1],"--")
        plt.savefig(f"plots/roc_{args.model}.png")
        plt.close()

    # Curva acc
    plt.figure()
    plt.plot(np.array(history["acc"])*100,label="acc")
    plt.plot(np.array(history["val_acc"])*100,label="val_acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("%")
    plt.savefig(f"plots/accuracy_{args.model}.png")
    plt.close()

    epoch_bar.set_postfix({
        "val_acc":f"{val_acc:.2f}"
    })