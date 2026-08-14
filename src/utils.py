import torch

import os

def setKaiming_(layer,alpha=0.0):
    torch.nn.init.kaiming_normal_(layer.weight,alpha)
    torch.nn.init.zeros_(layer.bias)
    return layer

def setXavier_(layer):
    torch.nn.init.xavier_normal_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer

class Data(torch.utils.data.dataset):
    def __init__(self):
        super().__init__()
    def __get
    def forward(self, x:torch.Tensor):
        return