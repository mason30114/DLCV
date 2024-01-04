import argparse
import torch
import sys
import pathlib

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_hw2')
    #parser.add_argument('--img_path', default = './hw3_data/p1_data/val/')   
    parser.add_argument('--batch_size', default = 1)   
    parser.add_argument('--plot', default = False)   
    parser.add_argument('--plot_dir', default = './p1_plot/')   
    parser.add_argument('--img_path', default = sys.argv[1])  
    parser.add_argument('--id2label_fn', default = sys.argv[2])  
    parser.add_argument('--out_fn', default = sys.argv[3])           
    #parser.add_argument('--id2label_fn', type=pathlib.Path, default = './hw3_data/p1_data/id2label.json') 
    parser.add_argument('--device', type=torch.device,default='cuda' if torch.cuda.is_available() else 'cpu') 
    #parser.add_argument('--device', type=torch.device,default = 'cpu') 
    args = parser.parse_args()
    return args    