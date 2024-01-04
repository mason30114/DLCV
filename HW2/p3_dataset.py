import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
import csv
from PIL import Image


class MNISTMDataset(Dataset):
    def __init__(self, root, transforms, mode = 'train'):
        self.images = None
        self.labels = None
        self.filenames = []  
        self.root = root
        self.mode = mode
        self.transform = transforms
        
        if self.mode == 'train':
            csv_fn = './hw2_data/digits/mnistm/train.csv'

        elif self.mode == 'val':
            csv_fn = './hw2_data/digits/mnistm/val.csv'
        else:
            csv_fn = './hw2_data/digits/mnistm/test.csv' 

        if self.mode == 'test':
            filenames = glob.glob(os.path.join(root, '*.png'))
            for fn in filenames:
                self.filenames.append(fn) # filename  
        else:      
            with open(csv_fn) as csv_file:
                reader = csv.reader(csv_file)   
                next(reader, None)
                for row in reader:
                    fn = row[0]
                    lb = row[1]
                    full_fn = os.path.join(self.root, fn)
                    self.filenames.append((full_fn, lb))      # (fn, lb)
            self.filenames = sorted(self.filenames)  

        self.len = len(self.filenames)

    def __getitem__(self,idx):
        if self.mode == 'test':
            image_fn = self.filenames[idx]
        else:
            image_fn, label = self.filenames[idx]            
        image = Image.open(image_fn).convert('RGB')         
        if self.transform is not None:
            image = self.transform(image)
        
        if self.mode == 'test':
            return image
        else:
            return image, label            

    def __len__(self):
        return self.len
    

class SVHNDataset(Dataset):
    def __init__(self, root, transforms, mode = 'train'):
        self.images = None
        self.labels = None
        self.filenames = []  
        self.root = root
        self.mode = mode
        self.transform = transforms
        
        if self.mode == 'train':
            csv_fn = './hw2_data/digits/svhn/train.csv'

        elif self.mode == 'val':
            csv_fn = './hw2_data/digits/svhn/val.csv'
        else:
            csv_fn = './hw2_data/digits/svhn/test.csv' 

        if self.mode == 'test':
            filenames = glob.glob(os.path.join(root, '*.png'))
            for fn in filenames:
                self.filenames.append(fn) # filename  
            self.filenames = sorted(self.filenames)  
        else:      
            with open(csv_fn) as csv_file:
                reader = csv.reader(csv_file)   
                next(reader, None)
                for row in reader:
                    fn = row[0]
                    lb = row[1]
                    full_fn = os.path.join(self.root, fn)
                    self.filenames.append((full_fn, lb))      # (fn, lb)
            self.filenames = sorted(self.filenames)  

        self.len = len(self.filenames)

    def __getitem__(self,idx):
        if self.mode == 'test':
            image_fn = self.filenames[idx]
        else:
            image_fn, label = self.filenames[idx]            
        image = Image.open(image_fn).convert('RGB')         
        if self.transform is not None:
            image = self.transform(image)
        
        if self.mode == 'test':
            return image
        else:
            return image, label            

    def __len__(self):
        return self.len


class USPSDataset(Dataset):
    def __init__(self, root, transforms, mode = 'train'):
        self.images = None
        self.labels = None
        self.filenames = []  
        self.root = root
        self.mode = mode
        self.transform = transforms
        
        if self.mode == 'train':
            csv_fn = './hw2_data/digits/usps/train.csv'

        elif self.mode == 'val':
            csv_fn = './hw2_data/digits/usps/val.csv'
        else:
            csv_fn = './hw2_data/digits/usps/test.csv' 

        if self.mode == 'test':
            filenames = glob.glob(os.path.join(root, '*.png'))
            for fn in filenames:
                self.filenames.append(fn) # filename  
            self.filenames = sorted(self.filenames)  
        else:      
            with open(csv_fn) as csv_file:
                reader = csv.reader(csv_file)   
                next(reader, None)
                for row in reader:
                    fn = row[0]
                    lb = row[1]
                    full_fn = os.path.join(self.root, fn)
                    self.filenames.append((full_fn, lb))      # (fn, lb)
            self.filenames = sorted(self.filenames)  

        self.len = len(self.filenames)

    def __getitem__(self,idx):
        if self.mode == 'test':
            image_fn = self.filenames[idx]
        else:
            image_fn, label = self.filenames[idx]            
        image = Image.open(image_fn).convert('RGB')         
        if self.transform is not None:
            image = self.transform(image)
        
        if self.mode == 'test':
            return image
        else:
            return image, label            

    def __len__(self):
        return self.len
    
    