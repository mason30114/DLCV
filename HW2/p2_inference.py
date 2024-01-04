import torch
import argparse
from UNet import UNet
import torchvision
from torchvision.utils import save_image, make_grid
from utils import beta_scheduler
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import sys
import os
import torchvision.transforms as transforms
import glob

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_hw2') 
    parser.add_argument('--T', default = 1000)  
    parser.add_argument('--time_step', default = 50)               
    parser.add_argument('--device', type=torch.device,default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default = sys.argv[2])
    parser.add_argument('--save_dir_exp', default = './p2_gen_image_exp/')  
    parser.add_argument('--ckpt_fn', default = sys.argv[3])
    parser.add_argument('--noise_dir', default = sys.argv[1]) 
    parser.add_argument('--gen_type', default = 'infer')
    parser.add_argument('--get_loss', default = False)        
    parser.add_argument('--beta1', default = 1e-4)
    parser.add_argument('--beta2', default = 2e-2)    
    args = parser.parse_args()
    return args

class DDIM_param():
    def __init__(self, args):
        self.timesteps = args.T
        self.betas = beta_scheduler(n_timestep=args.T+1, linear_start=args.beta1, linear_end=args.beta2)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
    
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu().to(int))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Reverse process modified from https://zhuanlan.zhihu.com/p/565698027
# Please see the part of "代码实现"
@torch.no_grad()
def reverse_n_save(model, args, param, x_s, eta): 
    
    ddim_timesteps = args.time_step
    step = param.timesteps // ddim_timesteps
    ddim_timestep_seq = np.asarray(list(range(0, param.timesteps, step)))
    ddim_timestep_seq = ddim_timestep_seq + 1
    ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
    clip_denoised = True

    # for reverse record
    print("Reverse in progress ... ")

    for i in range(ddim_timesteps-1, 0, -1):
        t = torch.full((1,), ddim_timestep_seq[i], device=args.device, dtype=torch.float32)
        prev_t = torch.full((1,), ddim_timestep_prev_seq[i], device=args.device, dtype=torch.float32)
        pred_noise = model(x_s, t)
        alpha_cumprod_t = extract(param.alphas_cumprod, t, x_s.shape)
        alpha_cumprod_t_prev = extract(param.alphas_cumprod, prev_t, x_s.shape)
        pred_x0 = (x_s - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)

        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, min=-1., max=1.) 
        
        sigmas_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x_s)
        x_s = x_prev

    return x_s

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    print('model loaded from %s' % checkpoint_path)


if __name__ == '__main__':
    args = arg_parse()
    param = DDIM_param(args)
    model = UNet().to(args.device)   
    load_checkpoint(args.ckpt_fn, model)

    if args.gen_type == 'infer':
        noise_filenames = glob.glob(os.path.join(args.noise_dir, '*.pt'))
        for i in range(10):
            x_s = torch.randn((1, 3, 256, 256), device=args.device, dtype=torch.float32)
            noise_fn = noise_filenames[i]
            noise_fn_last = noise_fn.split('/')[-1]
            noise_fn_last = noise_fn_last.split('.')[0]
            x_s = torch.load(noise_fn)
            x_pred = reverse_n_save(model, args, param, x_s, eta = 0.0)
            x_pred0 = x_pred.reshape(1, 3, 256, 256)
            save_image(x_pred0, os.path.join(args.save_dir, noise_fn_last + '.png'), normalize=True)#, value_range=(-1, 1))

        if args.get_loss == True:
            total_loss_255 = 0.0
            criterion = nn.MSELoss()
            #gt = torch.zeros((10, 3, 256, 256), device=args.device, dtype=torch.float32)
            gt = np.zeros((10, 3, 256, 256), dtype = np.float32)
            #pd = torch.zeros((10, 3, 256, 256), device=args.device, dtype=torch.float32)
            pd = np.zeros((10, 3, 256, 256), dtype = np.float32)
            for i in range(10):
                gt_path = './hw2_data/face/GT/0' + str(i) + '.png'
                read_path = os.path.join(args.save_dir,  '0'+str(i)+ '.png')
                transform = torchvision.transforms.Compose([transforms.ToTensor()])

                x_pred_r = Image.open(read_path)    
                x_gt = Image.open(gt_path)
                transform = torchvision.transforms.Compose([transforms.ToTensor()])
                x_pred_r0 = transform(x_pred_r)
                x_gt0 = transform(x_gt)
                x_pred_r0 = x_pred_r0.reshape(1, 3, 256, 256)#.to(args.device)    
                x_gt0 = x_gt0.reshape(1, 3, 256, 256)#.to(args.device)
                x_gt0 = np.array(x_gt0)                #
                x_pred_r0 = np.array(x_pred_r0)        #
                x_gt0*=255
                x_pred_r0*=255            
                gt[i, :, :, :] = x_gt0
                pd[i, :, :, :] = x_pred_r0
        
            #total_loss_255 = criterion(gt, pd).item()
            total_loss_255 = np.mean((gt - pd)**2)
            print("Average MSE Loss on 10 images ([0, 255] scale) = {}".format(total_loss_255))

    elif args.gen_type == 'eta_trans':
        x_t = torch.zeros((20, 3, 256, 256), device=args.device, dtype=torch.float32)
        etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for j in range(5):
            for i in range(4):
                x_s = torch.randn((1, 3, 256, 256), device=args.device, dtype=torch.float32)
                noise_fn = './hw2_data/face/noise/0' + str(i) + '.pt'
                x_s = torch.load(noise_fn)
                x_pred = reverse_n_save(model, args, param, x_s, eta = etas[j])
                x_pred0 = x_pred.reshape(1, 3, 256, 256)
                x_t[j*4+i, :, :, :] = x_pred0
        grid = make_grid(x_t, nrow = 4)
        save_image(grid, args.save_dir_exp + f"eta_trans_grid.png", normalize=True)#, value_range=(-1, 1))

    elif args.gen_type == 'interpolation':
        x_t = torch.zeros((11, 3, 256, 256), device=args.device, dtype=torch.float32)
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
        inter_type = 'slerp'
        if inter_type == 'slerp':
            for i in range(11):
                    x_s_0 = torch.randn((1, 3, 256, 256), device=args.device, dtype=torch.float32)
                    x_s_1 = torch.randn((1, 3, 256, 256), device=args.device, dtype=torch.float32)
                    noise_fn_0 = './hw2_data/face/noise/0' + str(0) + '.pt'
                    noise_fn_1 = './hw2_data/face/noise/0' + str(1) + '.pt'
                    x_s_0 = torch.load(noise_fn_0)
                    x_s_1 = torch.load(noise_fn_1)
                    nx_s_0 = x_s_0 / torch.norm(x_s_0, p=2, keepdim=True)
                    nx_s_1 = x_s_1 / torch.norm(x_s_1, p=2, keepdim=True)
                    inner = torch.inner(nx_s_0, nx_s_1).sum()
                    theta = torch.arccos(inner)
                    x_s = torch.sin((1-alphas[i])*theta) / torch.sin(theta) * x_s_0 + torch.sin(alphas[i]*theta) / torch.sin(theta) * x_s_1
                    x_pred = reverse_n_save(model, args, param, x_s, eta = 0.0)
                    x_pred0 = x_pred.reshape(1, 3, 256, 256)
                    x_t[i, :, :, :] = x_pred0
            grid = make_grid(x_t, nrow = 11)
            save_image(grid, args.save_dir_exp + f"inter_grid_slerp.png", normalize=True)#, value_range=(-1, 1))

        elif inter_type == 'linear':
            for i in range(11):
                    x_s_0 = torch.randn((1, 3, 256, 256), device=args.device, dtype=torch.float32)
                    x_s_1 = torch.randn((1, 3, 256, 256), device=args.device, dtype=torch.float32)
                    noise_fn_0 = './hw2_data/face/noise/0' + str(0) + '.pt'
                    noise_fn_1 = './hw2_data/face/noise/0' + str(1) + '.pt'
                    x_s_0 = torch.load(noise_fn_0)
                    x_s_1 = torch.load(noise_fn_1)
                    x_s = (1.0-alphas[i]) * x_s_0 + (alphas[i]) * x_s_1
                    x_pred = reverse_n_save(model, args, param, x_s, eta = 0.0)
                    x_pred0 = x_pred.reshape(1, 3, 256, 256)
                    x_t[i, :, :, :] = x_pred0
            grid = make_grid(x_t, nrow = 11)
            save_image(grid, args.save_dir_exp + f"inter_grid_linear.png", normalize=True)#, value_range=(-1, 1))






    


