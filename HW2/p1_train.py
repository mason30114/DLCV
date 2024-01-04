import torch
import torch.nn as nn
import os
from p1_dataset import P1Dataset, GENDataset
from torchvision import datasets, transforms, models
from p1_model_m import UNet
import argparse
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.datasets import MNIST
import random
from p1_parser import arg_parse
from digit_classifier import DATA, Classifier, val_get_acc

class DDPM_param():
    def __init__(self, args):
        self.betas = ((args.beta2 - args.beta1) * torch.arange(0, args.T + 1, dtype=torch.float32) / args.T + args.beta1).to(args.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)    

def forward_process(x0, t, param):
    noise = torch.randn_like(x0)
    xt = (torch.sqrt(param.alphas_cumprod[t, None, None, None]) * x0 + torch.sqrt(1-param.alphas_cumprod[t, None, None, None]) * noise)
    return xt, noise 
     
def train(model, optimizer, scheduler, args, param):
    criterion = nn.MSELoss()
    scaler = GradScaler()    
    iteration = 0
    best_acc = 0.0
    for epoch in range(args.num_epoch):
        train_loss = []
        total_train_loss = 0.0
        for batch_idx, (images, context) in enumerate(train_loader):
            model.train()
            with torch.autocast(device_type='cuda' if args.device != 'cpu' else 'cpu', dtype=torch.float16):
                optimizer.zero_grad()
                images = images.to(args.device)
                context = np.array(context, int)
                context = torch.Tensor(context)
                context = context.to(args.device)            
                context_mask = torch.bernoulli(torch.zeros_like(context) + args.drop_prob).to(args.device)                
                t = torch.randint(0, args.T, (len(images),)).long().to(args.device)
                xt, noise = forward_process(images, t, param)                   
                noise_pred = model(xt, context, t/args.T, context_mask)            
                loss = criterion(noise, noise_pred)
            train_loss.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)  
            scaler.update()
            if iteration % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(images), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            iteration += 1
        total_train_loss = sum(train_loss) / len(train_loss)
        print('Train Epoch: {}  Train Loss:{:.6f}'.format(epoch, total_train_loss))
        scheduler.step()

        model.eval() 
        mode = 'val'
        for class_idx in range(10):
            cnt = 0
            x_s, x_h = reverse_n_save(model, args = args, param = param, mode = mode, class_idx = class_idx)
            for image in x_s:
                cnt += 1
                save_image(image, args.save_dir + f"{class_idx}_{cnt}.png")
        acc = val_get_acc(args.save_dir)
        print("Epoch: {}  Acc = {}".format(epoch, acc))
        if acc > best_acc:
            best_acc = acc 
            print("Save best model: Best Acc = {}".format(best_acc))
            save_checkpoint(args.ckpt_dir + 'best_model_{}.pth'.format(epoch), model, optimizer)

# Usage of Guiding Reverese: https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py
# Please see sample() in "class DDPM()"
@torch.no_grad()
def reverse_n_save(model, args, param, mode, class_idx): 

    # total num of images generated
    nun_of_images = 0

    if mode == 'val':
        nun_of_images = 10
        c_s = torch.ones(nun_of_images, device=args.device, dtype=torch.int64) * class_idx
        c_s = c_s.repeat(2)

    elif mode == 'infer':
        nun_of_images = 100
        c_s = torch.ones(nun_of_images, device=args.device, dtype=torch.int64) * class_idx
        c_s = c_s.repeat(2)

    elif mode == 'make_grid':
        nun_of_images = 100
        c_s = torch.empty((nun_of_images,), dtype=torch.int64).to(args.device)
        for i in range(0, 10):
            c_t = torch.ones((10,), device=args.device, dtype=torch.int64).to(args.device) 
            c_t = c_t * i
            c_s[i*10:i*10+10,] = c_t
        c_s = c_s.repeat(2)

    elif mode == 'by_step':
        nun_of_images = 1
        c_s = torch.arange(0, 1).to(args.device) 
        c_s = c_s.repeat(2)
    
    # define random noise
    x_s = torch.randn(nun_of_images, *(3, 28, 28)).to(args.device)

    # define context mask
    context_mask = torch.zeros_like(c_s).to(args.device)
    context_mask[nun_of_images:] = 1

    # for reverse record
    x_h = []
    print("Reverse in progress ... Mode = {}".format(mode))

    for i in range(args.T, 0, -1):

        x_s = x_s.repeat(2, 1, 1, 1)
        t_s = torch.tensor([i/args.T]).to(args.device)
        t_s = t_s.repeat(nun_of_images, 1, 1, 1)
        t_s = t_s.repeat(2, 1, 1, 1)

        if i > 1:
            z = torch.randn(nun_of_images, *(3, 28, 28)).to(args.device) 
        else:
            z = 0
        
        # guiding reverse process
        eps = model(x_s, c_s, t_s, context_mask)
        eps1 = eps[:nun_of_images]
        eps2 = eps[nun_of_images:]
        eps = (1 + args.w) * eps1 - args.w * eps2
        x_s = x_s[:nun_of_images]
        x_s = ((1/torch.sqrt(param.alphas[i])) * (x_s - eps * ((1-param.alphas[i])/torch.sqrt(1-param.alphas_cumprod[i]))) + torch.sqrt(param.betas[i]) * z)
        if i%80 == 0 or i < 5:
            x_h.append((x_s[0], i))
    return x_s, x_h

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),}
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location = "cpu")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


if __name__ == '__main__':
    args = arg_parse()
    train_dataset = P1Dataset(args.img_path, transforms= transforms.Compose([transforms.ToTensor()]), mode = 'train')
    valid_dataset = P1Dataset(args.img_path, transforms= transforms.Compose([transforms.ToTensor()]), mode = 'val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 5)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 5)
    param = DDPM_param(args)
    model = UNet(in_channels = 3, n_classes=10).to(args.device)   
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-6)
    train(model, optimizer, scheduler, args, param)

