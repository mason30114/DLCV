import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
from PIL import Image


class P1Dataset(Dataset):
    def __init__(self, root, transforms, mode = 'train'):
        self.images = None
        self.labels = None
        self.filenames = []  
        self.root = root
        self.mode = mode
        self.transform = transforms
        # read filenames
        if self.mode == 'test':
            filenames = glob.glob(os.path.join(root, '*.png'))
            #print(filenames)
            for fn in filenames:
                self.filenames.append(fn) # filename
        else:        
            for i in range(50):
                #print(root)
                filenames = glob.glob(os.path.join(root, str(i)+'_*.png'))
                #print(filenames)
                for fn in filenames:
                    self.filenames.append((fn, i)) # (filename, label) pair


        self.len = len(self.filenames)

    def __getitem__(self,idx):
        if self.mode == 'test':
            image_fn = self.filenames[idx]
        else:
            image_fn, label = self.filenames[idx]            
        #print(image_fn)
        image = Image.open(image_fn)
        #image = Image.open(os.path.join(self.root, image_fn))           # 環境問題
        if self.transform is not None:
            image = self.transform(image)
        
        if self.mode == 'test':
            return image
        else:
            return image, label            

    def __len__(self):
        return self.len

'''# Data Normalization
    data_tmp = P1Dataset('./hw1_data/p1_data/train_50',transforms=transforms.ToTensor())
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for x, _ in data_tmp:
        mean += x.mean(dim=(1, 2))
        std += x.std(dim=(1, 2))
    mean /= len(data_tmp)
    std /= len(data_tmp)
    print(mean, std)     '''              # tensor([0.5077, 0.4813, 0.4312]) tensor([0.2000, 0.1986, 0.2034])
