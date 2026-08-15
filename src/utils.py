import torch
from torch.utils.data import Dataset

import pandas as pd
import os

def setKaiming_(layer,alpha=0.0):
    torch.nn.init.kaiming_normal_(layer.weight,alpha)
    torch.nn.init.zeros_(layer.bias)
    return layer

def setXavier_(layer):
    torch.nn.init.xavier_normal_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer

class ImageLoader(Dataset):
    def __init__(self, imgs_path:str, metadata_path:str):
        super().__init__()
    def __getitem__(self, key):
        pass
    def forward(self, x:torch.Tensor):
        return