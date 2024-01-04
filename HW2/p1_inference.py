import torch
import numpy as np
import torchvision
from p1_model_m import UNet
from torchvision.utils import save_image, make_grid
import random
from pathlib import Path
import sys
import argparse
import os
#from p1_parser import arg_parse
from p1_train import reverse_n_save, load_checkpoint, DDPM_param

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_hw2')
    parser.add_argument('--img_path', default = './hw2_data/digits/mnistm/data/')       
    parser.add_argument('--num_epoch', default = 100)
    parser.add_argument('--batch_size', default = 256)   
    parser.add_argument('--T', default = 400)        
    parser.add_argument('--lr', default = 1e-4)      
    parser.add_argument('--device', type=torch.device,default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default = './gen_image_val/') 
    parser.add_argument('--drop_prob', default = 0.1)
    parser.add_argument('--w', default = 2.0)
    parser.add_argument('--out_dir', default = sys.argv[1])
    parser.add_argument('--out_dir_grid', default = './test_gen_image2_grid/')
    parser.add_argument('--ckpt_fn', default = 'p1_ckpt.pth')
    parser.add_argument('--make_grid', default = False)
    #parser.add_argument('--by_step', default = False)   
    parser.add_argument('--beta1', default = 1e-4)
    parser.add_argument('--beta2', default = 2e-2)    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = arg_parse()
    out_dir = args.out_dir
    random.seed(0)
    torch.manual_seed(0)    
    out_dir_grid = args.out_dir_grid
    ckpt_fn = args.ckpt_fn
    model = UNet(in_channels = 3, n_classes=10)
    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(ckpt_fn, model, optimizer)
    model = model.to(args.device)
    param = DDPM_param(args)  
    #load_checkpoint(ckpt_fn, model, optimizer)
    model.eval()
    with torch.no_grad():
        mode = 'infer'
        for class_idx in range(10):
            cnt = 0
            x_i, x_h = reverse_n_save(model, args = args, param = param, mode = mode, class_idx = class_idx)
            for image in x_i:
                cnt += 1
                if cnt < 10:
                    save_image(image, os.path.join(out_dir, str(class_idx) + '_00' + str(cnt) + '.png'))
                elif cnt < 100:
                    save_image(image, os.path.join(out_dir, str(class_idx) + '_0' + str(cnt) + '.png'))
                else:
                    save_image(image, os.path.join(out_dir, str(class_idx) + '_' + str(cnt) + '.png'))

        if args.make_grid == True:
            mode = 'make_grid'
            x_i, x_h = reverse_n_save(model, args = args, param = param, mode = mode, class_idx = 0)
            grid = make_grid(x_i, nrow=10)                                 
            save_image(grid, out_dir_grid + 'grid.png')
            for images in x_h:
                save_image(images[0], out_dir_grid + '{}.png'.format(images[1]))    

    