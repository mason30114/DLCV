import torch
import torch.nn as nn
import torchvision.models as models
# Define pre-trained model
def get_model_pre():
    model = models.resnet101(pretrained = True)
    num_features = model.fc.in_features
    #print(num_features)
    model.fc = nn.Sequential(
              nn.Dropout(0.8),
              nn.Linear(num_features, 50),
              #nn.ReLU(),
              #nn.Linear(512, 50)
    )
    return model
#get_model_pre()