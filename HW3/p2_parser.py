import argparse
import torch
import sys
import pathlib

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_hw3')
    parser.add_argument('--batch_size', default = 8)      
    parser.add_argument('--train_img_root', default = './hw3_data/p2_data/images/train/')  
    parser.add_argument('--val_img_root', default = './hw3_data/p2_data/images/val/') 
    parser.add_argument('--train_json_fn', default = './hw3_data/p2_data/train.json')  
    parser.add_argument('--val_json_fn', default = './hw3_data/p2_data/val.json')  
    parser.add_argument('--encoder', default = 'vit_gigantic_patch14_clip_224')
    #parser.add_argument('--encoder', default = 'vit_large_patch14_224_clip_laion2b')
    parser.add_argument('--decoder_checkpoint', default = './hw3_data/p2_data/decoder_model.bin')
    parser.add_argument('--output_fn', default = './p2_output/pred_new.json')
    parser.add_argument('--encoder_fn', default = './encoder.json')
    parser.add_argument('--vocab_fn', default = './vocab.bpe')
    parser.add_argument('--ckpt_dir', default = './p2_lora_ckpt/')
    parser.add_argument('--p2_ckpt', default = './p2_lora.pth')
    parser.add_argument('--p2_plot_dir', default = './hw3_data/p3_data/images')
    parser.add_argument('--plot_image_dir', default = './hw3_data/p3_data/images/')
    parser.add_argument('--lr', default = 3e-5)
    parser.add_argument('--max_len', default = 60)
    parser.add_argument('--weight_decay', default = 0)
    parser.add_argument('--num_epoch', default = 10)       
    parser.add_argument('--infer_type', default = 'pt')   
    parser.add_argument('--plot_type', default = 'lora')  
    parser.add_argument('--prompt', default = 'The image shows that ') 
    parser.add_argument('--device', type=torch.device,default='cuda' if torch.cuda.is_available() else 'cpu') 
    parser.add_argument('--annotation_file', default = './hw3_data/p2_data/val.json')
    parser.add_argument('--images_root', default = './hw3_data/p2_data/images/val/')
  

    args = parser.parse_args()
    return args    