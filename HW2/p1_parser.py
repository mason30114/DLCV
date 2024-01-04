import argparse
import torch
import sys

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_hw2')
    parser.add_argument('--img_path', default = './hw2_data/digits/mnistm/data/')       
    parser.add_argument('--num_epoch', default = 100)
    parser.add_argument('--batch_size', default = 256)   
    parser.add_argument('--T', default = 400)        
    parser.add_argument('--lr', default = 1e-4)      
    parser.add_argument('--device', type=torch.device,default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default = './p1_val/') 
    parser.add_argument('--drop_prob', default = 0.1)
    parser.add_argument('--w', default = 2.0)
    #parser.add_argument('--out_dir', default = sys.argv[1])
    parser.add_argument('--out_dir_grid', default = './test_gen_image2_grid/')
    parser.add_argument('--ckpt_fn', default = './eval_example_2/model_99.pth')
    parser.add_argument('--ckpt_dir', default = './eval_example_2/')
    parser.add_argument('--make_grid', default = False)
    parser.add_argument('--by_step', default = False)   
    parser.add_argument('--beta1', default = 1e-4)
    parser.add_argument('--beta2', default = 2e-2)    
    args = parser.parse_args()
    return args