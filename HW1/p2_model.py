# Define model
import torch
import torchvision
from torchvision import models
import torch.nn as nn
import random

class Resnet_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights = None)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 65),
            #nn.ReLU()

        )
    def forward(self, x):
        x = self.backbone(x)      
        x = self.classifier(x)
        return x
    def mode_setting(self, mode):
        if mode == 'A':    # No Pre-train
            print("Train full model (backbone + classifier)")

        elif mode == 'B':
            print("Train full model (backbone + classifier) using SL pre-trained backcone")
            self.backbone.load_state_dict(torch.load('./hw1_data/p2_data/pretrain_model_SL.pt'), strict=False)

        elif mode == 'C':
            print("Train full model (backbone + classifier) using SSL pre-trained backcone")
            self.backbone.load_state_dict(torch.load('./p2_pre_checkpoint/pretrain_model_SSL_2.pt'), strict=False)

        elif mode == 'D':
            print("Train Classifier (Freeze Backbone) using SL pre-trained backcone")
            self.backbone.load_state_dict(torch.load('./hw1_data/p2_data/pretrain_model_SL.pt'), strict=False)
            for par in self.backbone.parameters():
                par.requires_grad = False

        elif mode == 'E':
            print("Train Classifier (Freeze Backbone) using SSL pre-trained backcone")
            self.backbone.load_state_dict(torch.load('./p2_pre_checkpoint/pretrain_model_SSL_2.pt'), strict=False)
            for par in self.backbone.parameters():
                par.requires_grad = False
        elif mode == 'Infer':
            print("Testing mode")                      
        else:
            print("Invalid training mode")

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)