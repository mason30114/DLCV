# Define model
#from torchvisions import models
import torch
import torch.nn as nn
import torchvision.models as models
def get_model():
    model = models.resnet50(pretrained = False)
    num_features = model.fc.in_features
    #print(num_features)
    model.fc = nn.Sequential(
              #nn.Dropout(0.8),
              nn.Linear(num_features, 50),
              #nn.ReLU(),
              #nn.Linear(512, 50)
    )
    #print(model)
    return model
#print(get_model())
#get_model()