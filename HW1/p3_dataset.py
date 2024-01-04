import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from mean_iou_evaluate import read_masks
import glob
import os
import numpy as np
from PIL import Image


class P3Dataset(Dataset):
    def __init__(self, root, transforms, mode = 'train'):
        self.images = None
        self.labels = None
        self.filenames = []  
        self.root = root
        self.mode = mode
        self.transform = transforms

        # read filenames     
        if mode == 'test':
            image_fn = glob.glob(os.path.join(root, '*_sat.jpg'))
            #image_fn.sort()       
            for im in image_fn:
                self.filenames.append(im) # (filename)   

        else:
            image_fn = glob.glob(os.path.join(root, '*_sat.jpg'))
            mask_fn = glob.glob(os.path.join(root, '*_mask.png'))
            image_fn.sort()
            mask_fn.sort()

            for im,mk in zip(image_fn,mask_fn):
                self.filenames.append((im, mk)) # (filename, mask) pair

        self.len = len(self.filenames)

    def __getitem__(self,idx):
        #======get a sample from the dataset======
        if self.mode == 'test':
            image_fn = self.filenames[idx]       
            image = Image.open(image_fn)
        else:
            image_fn, mask_fn = self.filenames[idx]        
            image = Image.open(image_fn)
            mask_img = Image.open(mask_fn)
        
        #flip_mode = np.random.randint(0,4)
        if self.mode == 'train':
            flip_mode = np.random.randint(0,4)
            if (flip_mode % 4) == 0:          # both flip
                image = horizontal_flip(np.array(image))
                mask_img = horizontal_flip(np.array(mask_img))
                image = vertical_flip(np.array(image))
                mask_img = vertical_flip(np.array(mask_img))
            elif (flip_mode % 4) == 1:          # ver flip
                image = vertical_flip(np.array(image))
                mask_img = vertical_flip(np.array(mask_img))
            elif (flip_mode % 4) == 2:          # hor flip
                image = horizontal_flip(np.array(image))
                mask_img = horizontal_flip(np.array(mask_img))
            else:          # none
                pass
        #image = Image.fromarray(image)
        #mask_img = Image.fromarray(mask_img)       
        image_c = image.copy()
        if self.mode == 'test':
            pass
        else:
            mask_c = mask_img.copy()
            #image_c = image.copy()
            #masks = np.empty((512, 512))
            mask= transforms.ToTensor()(mask_c)                 #(C,H,W) #[0,255]->[0,1]
            masks = torch.zeros(mask.size()[1:3])               #(512*512)
            #mask = np.array(mask_img)
            #mask = (mask >= 128).astype(int)
            mask = 4 * mask[0, :, :] + 2 * mask[1, :, :] + mask[2, :, :]
            masks[ mask == 3 ] = 0  # (Cyan: 011) Urban land 
            masks[ mask == 6 ] = 1  # (Yellow: 110) Agriculture land 
            masks[ mask == 5 ] = 2  # (Purple: 101) Rangeland 
            masks[ mask == 2 ] = 3  # (Green: 010) Forest land 
            masks[ mask == 1 ] = 4  # (Blue: 001) Water 
            masks[ mask == 7 ] = 5  # (White: 111) Barren land 
            masks[ mask == 0 ] = 6  # (Black: 000) Unknown
            masks[ mask == 4 ] = 6  # (Red: 100) Unknown
        

        if self.transform is not None:
            image_t = self.transform(image_c)

        if self.mode == 'test':
            return image_t
        else:
            return image_t, masks.long()                           #masks=labels
    
    def __len__(self):
        return self.len
    
    def get_path(self, i):
        if self.mode == 'test':
            im = self.filenames[i]            
        else:
            im, mk = self.filenames[i]
        return im
    
def horizontal_flip(image):
    image = image[:, ::-1, :]
    return image


def vertical_flip(image):
    image = image[::-1, :, :]
    return image


