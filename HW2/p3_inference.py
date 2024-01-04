import pandas
import sys
import torch
import torch.nn as nn
import torchvision
import os
from torchvision import datasets, transforms, models
from p3_model import NN_SVHN, NN_USPS
from p3_dataset import MNISTMDataset, USPSDataset, SVHNDataset
import argparse
import numpy as np
def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_hw2')
    parser.add_argument('--mnistm_image_dir', default = './hw2_data/digits/mnistm/data/')   
    parser.add_argument('--svhn_image_dir', default = './hw2_data/digits/svhn/data/')     
    parser.add_argument('--usps_image_dir', default = './hw2_data/digits/usps/data/')  
    parser.add_argument('--image_dir', default = sys.argv[1])  
    parser.add_argument('--out_fn', default = sys.argv[2])
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
    parser.add_argument('--svhn_ckpt', default = './p3_svhn_ckpt.pth')   
    parser.add_argument('--usps_ckpt', default = './p3_usps_ckpt.pth')   
    args = parser.parse_args()
    return args

@torch.no_grad()
def inference(args, model):
    print("Testing mode ...")
    model.eval()
    predict = []
    for images in test_loader:
        images = images.to(args.device)
        lambda_ = 0.0
        class_pred, domain_pred = model(images, lambda_)
        _, test_pred = torch.max(class_pred, 1) 
        for y in test_pred.cpu().numpy():
            predict.append(y)
 
    save_csv(predict, test_dataset, filepath = args.out_fn)
    print("Save succesfully!")  

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
  
def save_csv(prediction, test_dataset, filepath=sys.argv[2]):
    img_id = create_ids(test_dataset)
    assert len(img_id) == len(prediction) 
    dict = {
        "image_name": img_id,
        "label": prediction
        }
    pandas.DataFrame(dict).to_csv(filepath, index=False)
    

def create_ids(test_dataset):
    filenames = []
    for i in range(len(test_dataset)):
        tmp = test_dataset.filenames[i]
        tmp = os.path.split(tmp)
        filename_sh = tmp[1]
        filenames.append(filename_sh)
    return filenames


if __name__ == "__main__":

    args = arg_parse()
    image_dir_split = args.image_dir.split('/')
    dataset_type = ''
    for w in image_dir_split:
        if w == 'svhn':
            dataset_type = 'svhn'
            break
        elif w == 'usps':
            dataset_type = 'usps'
            break
        else:
            pass

    # Define test dataset
    if dataset_type == 'svhn':
        test_dataset = SVHNDataset(args.image_dir, transforms = transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean = [0.4413, 0.4458, 0.4715], std = [0.1169, 0.1206, 0.1042])]),
                                                   mode = 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8*args.batch_size, shuffle=False, num_workers = 0)
    elif dataset_type == 'usps':
        test_dataset = USPSDataset(args.image_dir, transforms = transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean = [0.2573, 0.2573, 0.2573], std = [0.3373, 0.3373, 0.3373])]),
                                                   mode = 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8*args.batch_size, shuffle=False, num_workers = 0)
    else:
        print("Invalid Dataset type!")

    if dataset_type == 'svhn':
        model = NN_SVHN().to(args.device)    
        load_checkpoint(args.svhn_ckpt, model)    
    else:
        model = NN_USPS().to(args.device)
        load_checkpoint(args.usps_ckpt, model)

    with torch.no_grad():
        inference(args, model)