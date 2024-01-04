import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
import csv
from PIL import Image


import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import clip
from p1_parser import arg_parse

class P1Dataset(Dataset):
    def __init__(self, root, transforms, mode = 'val'):
        self.images = None
        self.labels = None
        self.filenames = []  
        self.root = root
        self.mode = mode
        self.transform = transforms
        args = arg_parse()
        self.device = args.device
        _, self.preprocess = clip.load('ViT-B/32', self.device)
        # read filenames
        if self.mode == 'test':
            filenames = glob.glob(os.path.join(root, '*.png'))
            for fn in filenames:
                self.filenames.append(fn) # filename
        else:        
            for i in range(50):
                filenames = glob.glob(os.path.join(root, str(i)+'_*.png'))
                for fn in filenames:
                    self.filenames.append((fn, i)) # (filename, label) pair


        self.len = len(self.filenames)

    def __getitem__(self,idx):
        if self.mode == 'test':
            image_fn = self.filenames[idx]
        else:
            image_fn, label = self.filenames[idx]     

        image = Image.open(image_fn)

        #if self.transform is not None:
        #    image = self.transform(image)
        image = self.preprocess(image).to(self.device)
        
        if self.mode == 'test':
            return image
        else:
            return image, label   
                 
    def __len__(self):
        return self.len
    
    def get_fn(self,idx):
        t = self.filenames[idx]
        return t[0]
