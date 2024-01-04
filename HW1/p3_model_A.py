import torch
import torch.nn as nn
import torchvision.models as models

class FCN_32(nn.Module):
    def __init__(self, n_class=7):
        super(FCN_32, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.features = self.vgg16.features
        self.conv = nn.Sequential(
            nn.Conv2d(512, 1024, 4),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.Conv2d(1024, 4096, 4),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Conv2d(1024, 2048, 4),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.Conv2d(2048, 4096, 3),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),
        )
        self.score_fr = nn.Conv2d(2048, n_class, 1)
        self.upscore = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=224, stride=32, bias=False)
        #self.upscore1 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=28, stride=4, bias=False)
        #self.upscore2 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=8, stride=8, bias=False)
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.score_fr(x)
        x = self.upscore(x)
        #x = self.upscore1(x)
        #x = self.upscore2(x)  
        return x