import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
# Model modified from https://github.com/NaJaeMin92/pytorch_DANN/blob/master
# Please see the part of "utils.py"
class GRL(Function):
    @staticmethod
    def forward(ctx, x, lambda_):              # ctx ~ self
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        output = grad.neg() * ctx.lambda_
        return output, None
    
# Model modified from https://github.com/NaJaeMin92/pytorch_DANN/blob/master    
# Please see the part of "model.py"
class NN_USPS(nn.Module):
    def __init__(self):
        super(NN_USPS, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=8 * 28 * 28, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=8 * 28 * 28, out_features=256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2)
        )
        
    def forward(self, x, lambda_):
        x = self.extractor(x)
        x = x.view(-1, 8 * 28 * 28)
        class_out = self.classifier(x)
        reversed_input = GRL.apply(x, lambda_)
        domain_out = self.discriminator(reversed_input)
        return class_out, domain_out
    
# Model modified from https://github.com/NaJaeMin92/pytorch_DANN/blob/master    
# Please see the part of "model.py"
class NN_SVHN(nn.Module):
    def __init__(self):
        super(NN_SVHN, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout(), ###
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()

        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4 * 28 * 28, out_features=1024),
            nn.BatchNorm1d(1024),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.BatchNorm1d(128),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=4 * 28 * 28, out_features=256),
            nn.BatchNorm1d(256),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2)
        )
        
    def forward(self, x, lambda_):
        x = self.extractor(x)
        x = x.view(-1, 4 * 28 * 28)
        class_out = self.classifier(x)
        reversed_input = GRL.apply(x, lambda_)
        domain_out = self.discriminator(reversed_input)
        return class_out, domain_out

    

