import torch
import torch.nn as nn
import torchvision.models as models

class FCN8s(nn.Module):
    def __init__(self, n_class=7):
        super(FCN8s, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.features = self.vgg16.features
        self.features[0].padding = (100, 100)
      
        # fc6
        self.fc6 = nn.Conv2d(512, 1024, 4)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(1024, 2048, 4)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(2048, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        
         
    def forward(self, x):
        h = x
        h = self.features[1](self.features[0](h))
        h = self.features[3](self.features[2](h))
        h = self.features[4](h)

        h = self.features[6](self.features[5](h))
        h = self.features[8](self.features[7](h))
        h = self.features[9](h)

        h = self.features[11](self.features[10](h))
        h = self.features[13](self.features[12](h))
        h = self.features[15](self.features[14](h))
        h = self.features[16](h)
        pool3 = h  # 1/8

        h = self.features[18](self.features[17](h))
        h = self.features[20](self.features[19](h))
        h = self.features[22](self.features[21](h))
        h = self.features[23](h)
        pool4 = h  # 1/16

        h = self.features[25](self.features[24](h))
        h = self.features[27](self.features[26](h))
        h = self.features[29](self.features[28](h))
        h = self.features[30](h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h
    
def DeepLabV3_resnet101():
    model = models.segmentation.deeplabv3_resnet101(weight = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
                                                    num_classes = 7,
                                                    aux_loss = True)
    return model

#model = DeepLabV3_resnet101()
#print(model)