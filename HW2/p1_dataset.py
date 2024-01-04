import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
import csv
from PIL import Image


class P1Dataset(Dataset):
    def __init__(self, root, transforms, mode = 'train'):
        self.images = None
        self.labels = None
        self.filenames = []  
        self.root = root
        self.mode = mode
        self.transform = transforms
        
        if self.mode == 'train':
            csv_fn = ['./hw2_data/digits/mnistm/train.csv', './hw2_data/digits/mnistm/val.csv']

        elif self.mode == 'sparse':
            csv_fn = ['./hw2_data/digits/mnistm/sparse.csv']

        elif self.mode == 'val':
            csv_fn = ['./hw2_data/digits/mnistm/train.csv', './hw2_data/digits/mnistm/val.csv']

        else:
            csv_fn = './hw2_data/digits/mnistm/test.csv' 

        if self.mode == 'test':
            filenames = glob.glob(os.path.join(root, '*.png'))
            #print(filenames)
            for fn in filenames:
                self.filenames.append(fn) # filename  
        else:      
            for f in csv_fn:
                with open(f) as csv_file:
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
        #print(image_fn)
        image = Image.open(image_fn).convert('RGB')
        #image = Image.open(os.path.join(self.root, image_fn))           
        if self.transform is not None:
            image = self.transform(image)
        
        if self.mode == 'test':
            return image
        else:
            return image, label            

    def __len__(self):
        return self.len
    
class GENDataset(Dataset):
    def __init__(self):
        self.filenames = []          
        csv_fn = './generation_test/gen.csv' 

        with open(csv_fn) as csv_file:
            reader = csv.reader(csv_file)   
            next(reader, None)
            for row in reader:
                lb = row[0]
                self.filenames.append((lb))      # (lb)
            self.filenames = sorted(self.filenames)  

        self.len = len(self.filenames)

    def __getitem__(self,idx):
        label = self.filenames[idx]            
        #image = Image.open(os.path.join(self.root, image_fn))           
        return label            

    def __len__(self):
        return self.len


