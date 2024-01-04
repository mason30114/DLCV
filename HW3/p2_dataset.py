import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
import json
from PIL import Image
from tokenizer import BPETokenizer
from p2_parser import arg_parse

class P2Dataset(Dataset):
    def __init__(self, img_root, json_fn, transforms, sparse = False, mode = 'train'):
        self.filenames = []  
        #self.filenames_m = []
        #self.img_root = img_root
        #self.json_fn = json_fn
        self.mode = mode
        self.transform = transforms 
        self.sparse = sparse
        #self.f = json.load(json_fn.open(mode = 'r')) 
        if self.mode == 'test':
            filenames = glob.glob(os.path.join(img_root, '*.jpg'))
            for fn in filenames:
                self.filenames.append(fn) # filename             
        else:
            with open(json_fn) as f:
                lib = json.load(f) 
            if self.mode == 'train':
                if self.sparse == True:     
                    i = 0                                     
                    for data in lib['annotations']:
                        if i % 50 == 0:
                            for p in lib['images']:
                                if p['id'] == data['image_id']:
                                    self.filenames.append((os.path.join(img_root, p['file_name']), data['caption']))
                                    break
                                else:
                                    pass
                        i += 1

                else:                                        
                    for data in lib['annotations']:
                        for p in lib['images']:
                            if p['id'] == data['image_id']:
                                self.filenames.append((os.path.join(img_root, p['file_name']), data['caption']))
                                break
                            else:
                                pass
            elif self.mode == 'val':
                if self.sparse == True:     
                    i = 0                                     
                    for data in lib['annotations']:
                        if i % 5 == 0:
                            for p in lib['images']:
                                if p['id'] == data['image_id']:
                                    self.filenames.append((os.path.join(img_root, p['file_name']), data['caption']))
                                    break
                                else:
                                    pass
                        i += 1

                else:                                        
                    for data in lib['annotations']:
                        for p in lib['images']:
                            if p['id'] == data['image_id']:
                                self.filenames.append((os.path.join(img_root, p['file_name']), data['caption']))
                                break
                            else:
                                pass            
        self.len = len(self.filenames)
        

    def __getitem__(self,idx):       
        if self.mode == 'test':
            image_fn = self.filenames[idx]
        else:
            #image_fn, captions = self.filenames[idx]  
            image_fn, caption = self.filenames[idx]          
        image = Image.open(image_fn).convert('RGB')
        #image = Image.open(os.path.join(self.root, image_fn)).convert('RGB')           
        if self.transform is not None:
            image = self.transform(image)
        
        if self.mode != 'test':
            args = arg_parse()    
            tokenizer = BPETokenizer(encoder_file = args.encoder_fn, vocab_file = args.vocab_fn)
            # caption for model input
            caption_m = tokenizer.encode(caption)               # [b, 1024]
            caption_m.insert(0, 50256)
            caption_m.append(50256)
            while len(caption_m) < args.max_len:
                caption_m.append(50256)
            caption_m = torch.LongTensor(caption_m)

            # caption for gt
            caption_l = tokenizer.encode(caption)
            #caption_l.insert(0, 50256)
            caption_l.append(50256)
            while len(caption_l) < args.max_len:
                caption_l.append(-100)
            caption_l = torch.LongTensor(caption_l)
            #caption_l = caption_l.to(args.device)
        
        if self.mode == 'test':
            image_fn = os.path.split(image_fn)[1]
            return image, image_fn
        elif self.mode == 'val':
            image_fn = os.path.split(image_fn)[1]
            return image, caption_m, caption_l, image_fn  
        else:
            return image, caption_m, caption_l        

    def __len__(self):
        return self.len
    


if __name__ == '__main__':
    img_root = './hw3_data/p2_data/images/val/'
    json_fn = './hw3_data/p2_data/val.json'
    encoder_file = './encoder.json'
    vocab_file = 'vocab.bpe'
    dataset = P2Dataset(img_root, json_fn, transforms.Compose([transforms.ToTensor()]), mode = 'train')
    tokenizer = BPETokenizer(encoder_file = encoder_file, vocab_file = vocab_file)
    dat = dataset[0]
    dat1 = dataset[0]
    dat2 = dataset[0]
    dat3 = dataset[0]
    dat4 = dataset[0]
    img = dat[0]
    cont_m = dat[1]
    cont_l = dat[2]
    #cont_n = tokenizer.encode(cont)
    max_len = 0
    #for dat in dataset:
    #    if len(dat[1]) > max_len:
    #        max_len = len(dat[1])
    #        print("")
    #print(max_len)
    cont = 'two'
    cont_p = ' two'   

    #print(img)
    print(tokenizer.encode(cont))
    print(tokenizer.encode(cont_p))
    
    
    