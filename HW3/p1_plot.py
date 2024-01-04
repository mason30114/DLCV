import os
import clip
import torch
import tqdm
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import pandas
from torchvision import datasets, transforms, models
from p1_dataset import P1Dataset
#import argparse
from p1_parser import arg_parse
@torch.no_grad()
def plot_top5(model, args, text_inputs): 
    for num in range(0, 3):
        model.eval()
        idx = random.randint(0, 2499)
        images, labels = test_dataset[idx] 
        image_input = images.to(args.device).unsqueeze(0)
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)          
        values, indices = similarity[0].topk(5)                                     # for plot usage  
        val = []    
        for v in values:
            v = v.item()
            val.append(v)
        color = ['r','b','g','y','m']
        x = ['','','','','']
        texts = []
        for i in indices:
            with open(args.id2label_fn) as f:
                label_name = json.load(f)
            for l, n in label_name.items():
                if int(l) == i.item():
                    texts.append("a photo of a {}".format(n)) 

        fig, ax = plt.subplots(1, 2, figsize = (9, 4))    
        width = 0.75 # the width of the bars 
        ind = np.arange(len(val))  # the x locations for the groups
        img_fn = test_dataset.get_fn(idx = idx)
        img = Image.open(img_fn).convert('RGB')
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[1].barh(ind, val, width, color=color, align='edge')
        ax[1].set_xlim(0,1)
        ax[1].set_yticks(ind+width/2)
        ax[1].set_yticklabels(x, minor=False)
        plt.xticks(np.linspace(0, 1.0, 6),('0','20','40','60', '80','100'))
        plt.margins(0,0.05)
        for bar, text in zip(ax[1].patches, texts):
            ax[1].text(0.01, bar.get_y()+bar.get_height()/2, text, color = 'black', ha = 'left', va = 'center')
        plt.savefig(os.path.join(args.plot_dir, 'plot_{}.png'.format(num)))


if __name__ == '__main__':
    args = arg_parse()

    # Define Dataset
    test_dataset = P1Dataset(args.img_path, transforms = transforms.Compose([transforms.ToTensor()]), mode = 'val')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 0)
    # Define model
    model, _ = clip.load('ViT-B/32', args.device)

    # Define text
    with open(args.id2label_fn) as f:
        label_name = json.load(f)
    class_labels = [n for l, n in label_name.items()]  
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_labels]).to(args.device)

    # Plot
    plot_top5(model, args, text_inputs)