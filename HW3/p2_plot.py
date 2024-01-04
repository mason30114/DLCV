import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import tqdm
from tqdm import tqdm
import numpy as np
from torchvision import datasets, transforms, models
from p2_parser import arg_parse
#from p2_model_1 import Transfomer
#from p2_model_ad import AD_Transfomer
#from p2_model_pt import PT_Transfomer
#from p2_model_lora import LORA_Transfomer
from p2_model_plot import LORA_Transfomer
from p2_dataset import P2Dataset
from tokenizer import BPETokenizer
import loralib as lora
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize

def inference(args, model, image_tensor):
    with torch.no_grad():
        #for images, fn in tqdm(test_loader):
        #    images = images.to(args.device)
        #    img_feat = model.encoder.forward_features(images) 
        img_feat = model.encoder.forward_features(image_tensor)        
        current_word = torch.full((1, 1), 50256).long().to(args.device)
        pred = []
        pred_l = []
        attn = []
        for current_idx in range(0, args.max_len):
            output, current_attn = model.decoder(img_feat, current_word)
            attn.append(current_attn)
            #output = model.decoder(img_feat, current_word)
            output_pred = output.argmax(dim = 2)
            next_input = []
            current_pred = output_pred.squeeze(0)[current_idx].item()
                        
            if current_pred == 50256 or current_idx == args.max_len-1:  
                #output_w = tokenizer.decode(pred)
                tmp = []
                tmp.append(current_pred)
                pred_l.append(tmp)
                break
            else:                  
                pred.append(current_pred)
                tmp = []
                tmp.append(current_pred)
                pred_l.append(tmp)
                for i in pred: 
                    next_input.append(i)
                next_input.insert(0, 50256)
                next_input = torch.LongTensor(next_input)            
                current_word = next_input.to(torch.int64).unsqueeze(0).to(args.device)
                current_idx += 1

        #output_w = output_w.split('<e>')
        #output_w = output_w[0]
        #print(output_w)
    return pred_l, attn
   

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    print(sum([p.numel() for n, p in state.items()]))
    model.load_state_dict(state, strict=False)
    print('model loaded from %s' % checkpoint_path)

if __name__ == "__main__":

    args = arg_parse()
    tokenizer = BPETokenizer(encoder_file = args.encoder_fn, vocab_file = args.vocab_fn)
    torch.manual_seed(1)  # torch+CPU
    torch.cuda.manual_seed(1)  # torch+GPU
    root = args.plot_image_dir
    fn_list = ['bike', 'girl', 'sheep', 'ski', 'umbrella', '000000179758', '4406961500']

    model = LORA_Transfomer(args.encoder, args.decoder_checkpoint).to(args.device)     
    print(args.plot_type)
    model = model.to(args.device)
    load_checkpoint(args.p2_ckpt, model)
    for param in model.parameters():
        param.requires_grad = False  
    model.eval()
    #image_fn = './hw3_data/p3_data/images/bike.jpg'
    for fn in fn_list:
        image_fn = os.path.join(args.plot_image_dir, fn +'.jpg')            
        image = Image.open(image_fn).convert('RGB')
        trans = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])])
        image_tensor = trans(image)
        image_tensor = image_tensor.to(args.device)
        image_tensor = image_tensor.unsqueeze(0)
        pred, attn = inference(args, model, image_tensor)
        #print(pred)
        #word = word.split(' ')
        attn = attn[-1]                      #[mh, wl, 14^2+1]
        #print(attn.shape)
        word_len = attn.shape[1]    
        #print(word_len)
        #attn = attn.squeeze(0)               #[mh, wl, 14^2+1]
        #attn = attn.sum(dim = 1)            
        attn = attn.squeeze(0)               # [wl, 14^2+1] 
        attn = attn[:, 1:]
        attn = attn.reshape(-1, 16, 16)      # [wl, 14, 14]
        #print(attn.shape)
        print(fn)
        fig, ax = plt.subplots(4, 4, figsize = (20, 10))
        ax = ax.flatten()  
        for a in ax:
            a.set_xticks([])
            a.set_yticks([]) 
            a.axis('off')   
        for w in range(word_len+1):
            img = ax[w].imshow(image)
            #if w == 0 or w == word_len-1:
            if w == 0:
                ax[w].set_title('|<endoftext>|')        
            else:
                word = tokenizer.decode(pred[w-1])
                ax[w].set_title(word)  
                curr_att = attn[w-1]
                curr_att -= torch.min(curr_att)
                curr_att /= torch.max(curr_att)    
                curr_att = curr_att.cpu()      
                curr_att = resize(curr_att.unsqueeze(0), [image.size[1], image.size[0]]).squeeze(0) 
                curr_att = np.uint8(curr_att * 255)
                mask = ax[w].imshow(curr_att, cmap = 'jet', alpha = 0.7, extent = img.get_extent())            

        plt.savefig(os.path.join(args.p2_plot_dir, fn + '_attn.jpg'))



