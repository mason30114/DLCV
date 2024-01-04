import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import tqdm
from tqdm import tqdm
import numpy as np
from torchvision import datasets, transforms, models
#from p2_parser import arg_parse
from p2_model_1 import Transfomer
from p2_model_ad import AD_Transfomer
from p2_model_pt import PT_Transfomer
from p2_model_lora import LORA_Transfomer
from p2_dataset import P2Dataset
from tokenizer import BPETokenizer
import loralib as lora
import json
import argparse
import sys
#from p2_evaluate import get_score, getGTCaptions, readJSON, CIDERScore, CLIPScore

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_hw3')
    parser.add_argument('--batch_size', default = 8)      
    parser.add_argument('--train_img_root', default = './hw3_data/p2_data/images/train/')  
    parser.add_argument('--val_img_root', default = './hw3_data/p2_data/images/val/') 
    parser.add_argument('--train_json_fn', default = './hw3_data/p2_data/train.json')  
    parser.add_argument('--val_json_fn', default = './hw3_data/p2_data/val.json')  
    parser.add_argument('--encoder', default = 'vit_gigantic_patch14_clip_224')
    parser.add_argument('--decoder_checkpoint', default = './hw3_data/p2_data/decoder_model.bin')
    parser.add_argument('--output_fn', default = './p2_output/pred_new.json')
    parser.add_argument('--encoder_fn', default = './encoder.json')
    parser.add_argument('--vocab_fn', default = './vocab.bpe')
    parser.add_argument('--ckpt_dir', default = './p2_lora_ckpt/')
    parser.add_argument('--p2_ckpt', default = './p2_lora.pth')
    parser.add_argument('--lr', default = 3e-5)
    parser.add_argument('--max_len', default = 60)
    parser.add_argument('--weight_decay', default = 0)
    parser.add_argument('--num_epoch', default = 10)       
    parser.add_argument('--infer_type', default = 'lora')   
    parser.add_argument('--plot_type', default = 'lora')  
    parser.add_argument('--prompt', default = 'The image shows that ') 
    parser.add_argument('--device', type=torch.device,default='cuda' if torch.cuda.is_available() else 'cpu') 
    parser.add_argument('--annotation_file', default = './hw3_data/p2_data/val.json')
    parser.add_argument('--images_root', default = './hw3_data/p2_data/images/val/')

    parser.add_argument('--test_images_root', default = sys.argv[1])
    parser.add_argument('--test_json_fn', default = sys.argv[2]) 
    parser.add_argument('--decoder_weight', default = sys.argv[3])    

    args = parser.parse_args()
    return args    

def inference(args, model):
    tokenizer = BPETokenizer(encoder_file = args.encoder_fn, vocab_file = args.vocab_fn)
    output_dict = {}
    with torch.no_grad():
        for images, fn in tqdm(test_loader):
            images = images.to(args.device)
            img_feat = model.encoder.forward_features(images)        
            current_word = torch.full((1, 1), 50256).long().to(args.device)
            pred = []
            for current_idx in range(0, args.max_len):
                output = model.decoder(img_feat, current_word)
                output_pred = output.argmax(dim = 2)
                next_input = []
                current_pred = output_pred.squeeze(0)[current_idx].item()
                    
                if current_pred == 50256 or current_idx == args.max_len-1:
                #    if pred[0] == 257:
                #        pred[0] = 64
                #    elif pred[0] == 734:
                #        pred[0] = 11545   
                    output_w = tokenizer.decode(pred)
                    break
                else:
                    #if current_idx == 0 and current_pred == 257:
                    #    current_pred = 64
                    #elif current_idx == 0 and current_pred == 734:
                    #    current_pred = 11545                   
                    pred.append(current_pred)
                    for i in pred: 
                        next_input.append(i)
                    next_input.insert(0, 50256)
                    next_input = torch.LongTensor(next_input)            
                    current_word = next_input.to(torch.int64).unsqueeze(0).to(args.device)
                    current_idx += 1

            output_w = output_w.split('<e>')
            output_w = output_w[0]
            #print(output_w)
            out_fn = fn[0].split('.')[0]
            output_dict[out_fn] = output_w

    with open(args.test_json_fn, "w") as f:
        out = json.dumps(output_dict, sort_keys=True, indent=4)
        f.write(out)
    #cider_score, clip_score = get_score(args, args.output_fn)
    #print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")    

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location=torch.device("cpu") )
    print(sum([p.numel() for n, p in state.items()]))
    model.load_state_dict(state, strict=False)
    print('model loaded from %s' % checkpoint_path)

if __name__ == "__main__":

    args = arg_parse()

    # Define validation dataset
    test_dataset = P2Dataset(args.test_images_root, ' ', transforms = transforms.Compose([
                                                                      transforms.ToTensor(),
                                                                      transforms.Resize((224, 224)),
                                                                      transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])])
                                                                     , mode = 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers = 0)
    torch.manual_seed(1)  # torch+CPU
    torch.cuda.manual_seed(1)  # torch+GPU
    if args.infer_type == 'pt':
        model = PT_Transfomer(args.encoder, args.decoder_weight).to(args.device) 
    elif args.infer_type == 'ad':
        model = AD_Transfomer(args.encoder, args.decoder_weight).to(args.device)
    elif args.infer_type == 'lora':
        model = LORA_Transfomer(args.encoder, args.decoder_weight).to(args.device)
    else:
        model = Transfomer(args.encoder, args.decoder_weight).to(args.device)       
    print(args.infer_type)
    load_checkpoint(args.p2_ckpt, model)
    model = model.to(args.device)
    for param in model.parameters():
        param.requires_grad = False  
    print(f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")
    print(len(test_dataset))
    #load_checkpoint(args.p2_ckpt, model)
    model.eval()
    inference(args, model)