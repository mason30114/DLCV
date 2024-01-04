import argparse
import torch
import sys

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_hw2')
    parser.add_argument('--mnistm_image_dir', default = './hw2_data/digits/mnistm/data/')   
    parser.add_argument('--svhn_image_dir', default = './hw2_data/digits/svhn/data/')     
    parser.add_argument('--usps_image_dir', default = './hw2_data/digits/usps/data/')         
    parser.add_argument('--num_epoch', default = 200)
    parser.add_argument('--batch_size', default = 1024)           
    parser.add_argument('--lr', default = 2e-3)
    parser.add_argument('--weight_decay', default = 0.002)
    parser.add_argument('--momentum', default = 0.9)       
    parser.add_argument('--device', type=torch.device,default='cuda' if torch.cuda.is_available() else 'cpu')
    #parser.add_argument('--device', type=torch.device,default='cpu')
    parser.add_argument('--dataset_type', default = 'svhn') 
    parser.add_argument('--train_type', default = 'Source')     
    parser.add_argument('--ckpt_dir', default = './p3_ckpt/')   
    parser.add_argument('--svhn_ckpt', default = './p3_ckpt/ss_svhn_dann3_2_44.43.pth')   
    parser.add_argument('--usps_ckpt', default = './p3_ckpt/cc_usps_dann2_11_78.90.pth')   

    args = parser.parse_args()
    return args