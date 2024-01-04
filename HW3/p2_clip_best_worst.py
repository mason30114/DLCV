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
from p2_model_lora import LORA_Transfomer
#from p2_model_plot import LORA_Transfomer
from p2_dataset import P2Dataset
from tokenizer import BPETokenizer
import loralib as lora
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
import clip
class CLIPScore:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def __call__(self, predictions, images_root):
        """
        Input:
            predictions: dict of str
            images_root: str
        Return:
            clip_score: float
        """
        total_score = 0.

        for img_name, pred_caption in predictions.items():
            image_path = os.path.join(images_root, f"{img_name}.jpg")
            image = Image.open(image_path).convert("RGB")

            total_score += self.getCLIPScore(image, pred_caption)
        return total_score / len(predictions)

    def getCLIPScore(self, image, caption):
        """
        This function computes CLIPScore based on the pseudocode in the slides.
        Input:
            image: PIL.Image
            caption: str
        Return:
            cilp_score: float
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([caption], truncate=True).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
        
        cos_sim = torch.nn.functional.cosine_similarity(image_features, text_features).item()
        return 2.5 * max(cos_sim, 0)
    

def inference(args, model):
    best_score = 0.0
    worst_score = 1.0
    #best_fn = []
    #worst_fn = []
    with torch.no_grad():
        for images, fn in tqdm(valid_loader):
            images = images.to(args.device)
            img_feat = model.encoder.forward_features(images)        
            current_word = torch.full((1, 1), 50256).long().to(args.device)
            pred = []

            for current_idx in range(0, args.max_len):
                output = model.decoder(img_feat, current_word)
                #attn.append(current_attn)
                #output = model.decoder(img_feat, current_word)
                output_pred = output.argmax(dim = 2)
                next_input = []
                current_pred = output_pred.squeeze(0)[current_idx].item()
                            
                if current_pred == 50256 or current_idx == args.max_len-1:  
                    output_w = tokenizer.decode(pred)
                    #tmp = []
                    #tmp.append(current_pred)
                    #pred_l.append(tmp)
                    break
                else:                  
                    pred.append(current_pred)
                    tmp = []
                    tmp.append(current_pred)
                    #pred_l.append(tmp)
                    for i in pred: 
                        next_input.append(i)
                    next_input.insert(0, 50256)
                    next_input = torch.LongTensor(next_input)            
                    current_word = next_input.to(torch.int64).unsqueeze(0).to(args.device)
                    current_idx += 1

            output_w = output_w.split('<e>')
            output_w = output_w[0]

            clip_model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
            clip_model.eval()
            total_score = 0.

            image = Image.open(os.path.join(args.val_img_root, fn[0])).convert("RGB")
            total_score += getCLIPScore(image, output_w, clip_model, clip_preprocess)

            if total_score > best_score:
                best_fn = fn
                best_score = total_score
                print('Best = {}'.format(best_score))
                print(best_fn)
            if total_score < worst_score:
                worst_fn = fn
                worst_score = total_score
                print('Worst = {}'.format(worst_score))
                print(worst_fn)
    print('Total Best = {}'.format(best_score))
    print('Total Best FN = {}'.format(best_fn))
    print('Total Worst = {}'.format(worst_score))
    print('Total Worst FN = {}'.format(worst_fn))
    return best_score, best_fn, worst_score, worst_fn

def getCLIPScore(image, caption, clip_model, clip_preprocess):
    """
    This function computes CLIPScore based on the pseudocode in the slides.
    Input:
        image: PIL.Image
        caption: str
    Return:
        cilp_score: float
    """
    image_input = clip_preprocess(image).unsqueeze(0).to(args.device)
    text_input = clip.tokenize([caption], truncate=True).to(args.device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        
    cos_sim = torch.nn.functional.cosine_similarity(image_features, text_features).item()
    return 2.5 * max(cos_sim, 0)   

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
    # Define validation dataset
    valid_dataset = P2Dataset(args.val_img_root, args.val_json_fn, transforms = transforms.Compose([
                                                                                  transforms.ToTensor(),
                                                                                  transforms.Resize((224, 224)),
                                                                                  transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
                                                                                  , sparse = True
                                                                                  , mode = 'test')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers = 0)

    model = LORA_Transfomer(args.encoder, args.decoder_checkpoint).to(args.device)     
    print(args.plot_type)
    model = model.to(args.device)
    load_checkpoint(args.p2_ckpt, model)
    for param in model.parameters():
        param.requires_grad = False  
    model.eval()

    #image = valid_dataset[0][0].unsqueeze(0).to(args.device)
    #fn = valid_dataset[0][1]
    best_score, best_fn, worst_score, worst_fn = inference(args, model)
    print('Total Best = {}'.format(best_score))
    print('Total Best FN = {}'.format(best_fn))
    print('Total Worst = {}'.format(worst_score))
    print('Total Worst FN = {}'.format(worst_fn))
    #print(output_w)
    #clip_model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
    #clip_model.eval()
    #total_score = 0.

    #image = Image.open(os.path.join(args.val_img_root, fn)).convert("RGB")
    #total_score += getCLIPScore(image, output_w)
    #print(total_score)
    #print(fn)
    #print(total_score / len(predictions))
