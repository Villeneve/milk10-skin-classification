import torch
import torch.nn as nn

from torchvision import models
from torchinfo import summary

from utils import setXavier_, setKaiming_

import argparse


def get_model(model_name:str, num_outs:int=2)->torch.nn.Module:
    if model_name == "inception":
        md = models.Inception3(init_weights=models.Inception_V3_Weights.DEFAULT,aux_logits=False)
        md.fc = nn.Sequential(
            nn.Dropout(.25),
            setXavier_(nn.Linear(2048,num_outs)),
        )
        return md
    if model_name == "vgg16":
        md = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        md.classifier = nn.Sequential(
            nn.Dropout(.25),
            setXavier_(nn.Linear(25088,num_outs)),
        )
        return md
    if model_name == "resnet34":
        md = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        md.fc = nn.Sequential(
            nn.Dropout(.25),
            setXavier_(nn.Linear(512,num_outs)),
        )
        return md

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["inception","vgg16","resnet34"],
        required=True,
        help="Escolha o modelo para o classificador",
    )
    args = parser.parse_args()
    with torch.inference_mode():
        get_model(args.model).train()(torch.randn(1,3,256,256))
    summary(get_model(args.model),(1,3,256,256),verbose=1,device=torch.device('cpu'))
    # print(f"Model: {args.model}")