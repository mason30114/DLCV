import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
#from mean_iou_evaluate import read_masks
import glob
import os
import numpy as np
from PIL import Image


class MiniDataset(Dataset):
    def __init__(self, root, transforms):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        #self.mode = mode
        self.transform = transforms

        # read filenames
        image_fn = glob.glob(os.path.join(root, '*.jpg'))
        #image_fn.sort()

        for im in image_fn:
            self.filenames.append(im) # (filename, mask) pair

        self.len = len(self.filenames)

    def __getitem__(self,idx):
        image_fn = self.filenames[idx]
        #print(image_fn)
        image = Image.open(image_fn)
        #image = Image.open(os.path.join(self.root, image_fn))           # 環境問題
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return self.len

class OfficeDataset(Dataset):
    def __init__(self, root, transforms, mode = 'train'):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.mode = mode
        self.transform = transforms
        # read filenames
        if self.mode == 'test':
            filenames = glob.glob(os.path.join(root, '*.jpg'))
            #print(filenames)
            for fn in filenames:
                self.filenames.append(fn) # filename
        else:        
            for i in range(50):
                #print(root)
                filenames = glob.glob(os.path.join(root, str(i)+'_*.jpg'))
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