import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import tqdm
from tqdm import tqdm
import torchvision
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from p2_parser import arg_parse
from p2_model_ad import AD_Transfomer
from decoder import Decoder, Config
from p2_dataset import P2Dataset
from torchvision import datasets, transforms, models
from tokenizer import BPETokenizer
from torch.cuda.amp import autocast, GradScaler
#import loralib as lora
import json
from p2_evaluate import get_score, getGTCaptions, readJSON, CIDERScore, CLIPScore
from collections import defaultdict
from argparse import ArgumentParser
from PIL import Image
import clip
import language_evaluation
from p2_inference import inference
def train(args, model, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss(ignore_index=-100) 
    iteration = 0
    tokenizer = BPETokenizer(encoder_file = args.encoder_fn, vocab_file = args.vocab_fn)
    #scaler = GradScaler()  
    for epoch in range(args.num_epoch):
        train_loss = []
        total_train_loss = 0.0
        #model.decoder.train()
        model.train()
        for batch_idx, (images, caption_m, caption_l) in tqdm(enumerate(train_loader)):
        #for batch_idx, (images, caption_m, caption_l, fn) in tqdm(enumerate(valid_loader)):
        #for images, caption_m, caption_l, fn in tqdm(valid_loader):
            #with torch.autocast(device_type = args.device):
            #model.decoder.train()
            with autocast():
                images = images.to(args.device)
                caption_m = caption_m.to(args.device)
                caption_l = caption_l.to(args.device)
                #tokenizer = BPETokenizer(encoder_file = args.encoder_fn, vocab_file = args.vocab_fn)
                optimizer.zero_grad()
                output = model(images, caption_m)
                loss = criterion(output.view(-1, 50257), caption_l.view(-1))
                train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)  
            #scaler.step(scheduler) 
            scheduler.step()  
            #scaler.update()              
            if iteration % 100 == 0:
                print('Train Epoch: {} Loss: {:.6f} lr = {:.6f}'.format(epoch, loss.item(), optimizer.param_groups[0]['lr']))      
                #print(output[0])
                pred = output[0].argmax(dim = -1)
                pred = pred.squeeze(0)
                pred = pred.tolist()
                output = tokenizer.decode(pred) 
                output_w = output.replace('<|endoftext|>', '<e>', args.max_len)
                output_w = output_w.split('<e>')
                output_w = output_w[0]               
                gt = caption_m[0].squeeze(0)
                gt = gt.tolist()
                gt = tokenizer.decode(gt)
                gt_w = gt.replace('<|endoftext|>', '', args.max_len)  
                print('gt = {}\n'.format(gt_w))  
                print('pred = {}\n'.format(output_w))     
            iteration += 1
            #scheduler.step()  

        total_train_loss = sum(train_loss) / len(train_loss)
        print("Epoch:{}  Average train Loss:{:.4f}".format(epoch, total_train_loss))         
        #ckpt_fn =  os.path.join(args.ckpt_dir, 'gig16_ep0.pth')
        with torch.no_grad():
            model.eval()
            #load_checkpoint(ckpt_fn, model)
        #print(f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")
        #lora_save_checkpoint(ckpt_fn, model)
        #cider_score, clip_score = eval_n_save(args, model)
            cider_score, clip_score = eval_n_save(args, model)
            ckpt_fn =  os.path.join(args.ckpt_dir, 'gigantic_AD_ep{}_CD{:3f}_CL{:3f}.pth'.format(epoch, cider_score, clip_score))
            #eval_n_save(args, model)
            lora_save_checkpoint(ckpt_fn, model)
            #lora_save_checkpoint(ckpt_fn, model)
            #ckpt_fn =  os.path.join(args.ckpt_dir, 'gigF_ep{}_CD{:3f}_CL{:3f}.pth'.format(epoch, cider_score, clip_score))
            #lora_save_checkpoint(ckpt_fn, model)
            #print(f"Epoch: {epoch} CIDEr: {cider_score} | CLIPScore: {clip_score}")
@torch.no_grad()
def eval_n_save(args, model):
    print("Validation....")
    tokenizer = BPETokenizer(encoder_file = args.encoder_fn, vocab_file = args.vocab_fn)
    output_dict = {}
    #for images, caption_m, caption_l, fn in tqdm(valid_t_loader):
    for images, caption_m, caption_l, fn in tqdm(valid_loader):
        #model.eval()
        images = images.to(args.device)
        caption_m = caption_m.to(args.device)
        caption_l = caption_l.to(args.device)
        img_feat = model.encoder.forward_features(images)        
        current_word = torch.full((1, 1), 50256).long().to(args.device)
        pred = []
        for current_idx in range(0, args.max_len):
            output = model.decoder(img_feat, current_word)
            output_pred = output.argmax(dim = 2)
            next_input = []
            current_pred = output_pred.squeeze(0)[current_idx].item()
            if current_pred == 50256 or current_idx == args.max_len-1:
                output_w = tokenizer.decode(pred)
                break
            else:
                #if current_idx == 0 and current_pred == 257:
                #    current_pred = 64
                pred.append(current_pred)
                for i in pred: 
                    next_input.append(i)
                next_input.insert(0, 50256)
                next_input = torch.LongTensor(next_input)            
                current_word = next_input.to(torch.int64).unsqueeze(0).to(args.device)
                current_idx += 1

        output_w = output_w.split('<e>')
        output_w = output_w[0]
        out_fn = fn[0].split('.')[0]
        output_dict[out_fn] = output_w
        val_json = 'pred.json'
    with open(val_json, "w") as f:
        out = json.dumps(output_dict, sort_keys=True, indent=4)
        f.write(out)
    cider_score, clip_score = get_score(args, val_json)
    return cider_score, clip_score

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state, strict=False)
    print('model loaded from %s' % checkpoint_path)


def lora_save_checkpoint(checkpoint_path, model):
    '''lora.mark_only_lora_as_trainable(model)
    for layer in model.decoder.transformer.h: 
        for name, param in layer.cross_attn.named_parameters():
            param.requires_grad = True 
        for name, param in layer.ln_3.named_parameters():
            param.requires_grad = True 
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False'''  
    trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
    save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
    #print(len(save_weights))
    torch.save(save_weights, checkpoint_path)

def test(checkpoint_path, model):
    #lora.mark_only_lora_as_trainable(model)
    for layer in model.decoder.transformer.h: 
        for name, param in layer.cross_attn.named_parameters():
            param.requires_grad = True 
        for name, param in layer.ln_3.named_parameters():
            param.requires_grad = True 
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False  
    trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
    save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
    #print(len(save_weights))
    torch.save(save_weights, checkpoint_path)
    state_dict = torch.load(checkpoint_path)
    #print(state_dict)
    print(sum([p.numel() for n, p in state_dict.items()]))    
if __name__ == "__main__":

    args = arg_parse()

    # Define training dataset
    train_dataset = P2Dataset(args.train_img_root, args.train_json_fn, transforms.Compose([
                                                                                     transforms.RandomHorizontalFlip(0.3),
                                                                                     transforms.ColorJitter(),
                                                                                     transforms.ToTensor(),
                                                                                     transforms.Resize((224, 224)),
                                                                                     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
                                                                                    , sparse = False   #####################
                                                                                    , mode = 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 0)


    # Define validation dataset
    valid_dataset = P2Dataset(args.val_img_root, args.val_json_fn, transforms = transforms.Compose([
                                                                                  transforms.ToTensor(),
                                                                                  transforms.Resize((224, 224)),
                                                                                  transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
                                                                                  , sparse = True
                                                                                  , mode = 'val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers = 0)

    # Define validation dataset
    valid_t_dataset = P2Dataset(args.images_root, args.annotation_file, transforms = transforms.Compose([
                                                                                  transforms.ToTensor(),
                                                                                  transforms.Resize((224, 224)),
                                                                                  transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
                                                                                  , sparse = False
                                                                                  , mode = 'val')
    #valid_t_dataset = valid_t_dataset[0:24]
    valid_t_loader = torch.utils.data.DataLoader(valid_t_dataset, batch_size=1, shuffle=False, num_workers = 0)

    torch.manual_seed(1)  # torch+CPU
    torch.cuda.manual_seed(1)  # torch+GPU

    model = AD_Transfomer(args.encoder, args.decoder_checkpoint).to(args.device) 

    model = model.to(args.device)
    for name, param in model.named_parameters():                    ###########################
        param.requires_grad = False
    #lora.mark_only_lora_as_trainable(model)
    #print(f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M") 
    for layer in model.decoder.transformer.h: 
        for name, param in layer.cross_attn.named_parameters():
            param.requires_grad = True 
        for name, param in layer.ln_3.named_parameters():
            param.requires_grad = True 

        for name, param in layer.attn.adapter.named_parameters():
            param.requires_grad = True 

    for name, param in model.encoder.named_parameters():
        param.requires_grad = False
    #print(f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M") 
    parms2update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params=parms2update, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch * len(train_loader), eta_min = 1e-6)
    print(f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")
    print(len(train_dataset))
    print(len(valid_dataset))
    train(args, model, optimizer, scheduler)
