import pandas
import sys
import torch
import torch.nn as nn
import torchvision
import os
from torchvision import datasets, transforms, models
from p3_model import NN_SVHN, NN_USPS
from p3_dataset import MNISTMDataset, USPSDataset, SVHNDataset
from p3_parser import arg_parse
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.manifold import TSNE

@torch.no_grad()
def inference(args, model):
    print("Testing mode ...")
    model.eval()
    all_class_pred = None
    all_class_labels = None
    all_domain_pred = None
    all_domain_labels = None

    # source plot 
    for source_images, source_class_labels in source_test_loader:
        source_images = source_images.to(args.device)
        source_class_labels = np.array(source_class_labels, int)
        source_class_labels = torch.Tensor(source_class_labels).type(torch.LongTensor)
        source_class_labels = source_class_labels.to(args.device)
        lambda_ = 0.0
        source_class_pred, source_domain_pred = model(source_images, lambda_)
        source_domain_labels = torch.zeros(len(source_images), dtype = int).to(args.device)        # source = 0  

        # append class result (source)
        if all_class_pred is None:
            all_class_pred = source_class_pred.cpu().numpy()
            all_class_labels = source_class_labels.cpu().numpy().flatten()
        else:
            source_class_pred = source_class_pred.cpu().numpy()
            source_class_labels = source_class_labels.cpu().numpy().flatten()
            all_class_pred = np.vstack((all_class_pred, source_class_pred))
            all_class_labels = np.concatenate((all_class_labels, source_class_labels))

        # append domain result (source)
        if all_domain_pred is None:
            all_domain_pred = source_domain_pred.cpu().numpy()
            all_domain_labels = source_domain_labels.cpu().numpy().flatten()
        else:
            source_domain_pred = source_domain_pred.cpu().numpy()
            source_domain_labels = source_domain_labels.cpu().numpy().flatten()
            all_domain_pred = np.vstack((all_domain_pred, source_domain_pred))
            all_domain_labels = np.concatenate((all_domain_labels, source_domain_labels))      

  
    for target_images, target_class_labels in target_test_loader:
        lambda_ = 0.0
        target_images = target_images.to(args.device)
        target_class_labels = np.array(target_class_labels, int)
        target_class_labels = torch.Tensor(target_class_labels).type(torch.LongTensor)
        target_class_labels = target_class_labels.to(args.device)
        target_class_pred, target_domain_pred = model(target_images, lambda_)

        target_domain_labels = torch.ones(len(target_images), dtype = int).to(args.device)         # target = 1    

        # append class result (target)
        target_class_pred = target_class_pred.cpu().numpy()
        target_class_labels = target_class_labels.cpu().numpy().flatten()
        all_class_pred = np.vstack((all_class_pred, target_class_pred))
        all_class_labels = np.concatenate((all_class_labels, target_class_labels))

        # append domain result (target)
        target_domain_pred = target_domain_pred.cpu().numpy()
        target_domain_labels = target_domain_labels.cpu().numpy().flatten()
        all_domain_pred = np.vstack((all_domain_pred, target_domain_pred))
        all_domain_labels = np.concatenate((all_domain_labels, target_domain_labels))      


    all_class_pred = all_class_pred.reshape(all_class_pred.shape[0], -1)
    all_domain_pred = all_domain_pred.reshape(all_domain_pred.shape[0], -1)

    # plot t-SNE (class)
    tsne = TSNE(n_components=2)
    d_x = tsne.fit_transform(all_class_pred)
    plt.figure()
    plt.title(f"Class t-SNE figure for {args.dataset_type}")
    plt.scatter(d_x[:, 0], d_x[:, 1], c=all_class_labels)
    plt.savefig(f"./P3_plots/Class_TSNE_{args.dataset_type}")

    # plot t-SNE (domain)
    tsne = TSNE(n_components=2)
    d_x = tsne.fit_transform(all_domain_pred)
    plt.figure()
    plt.title(f"Domain t-SNE figure for {args.dataset_type}")
    plt.scatter(d_x[:, 0], d_x[:, 1], c=all_domain_labels)
    plt.savefig(f"./P3_plots/Domain_TSNE_{args.dataset_type}")

    #print('Accuracy of the network on the test images:{:2f} %'.format(100 * correct / total))
    #return correct / total

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == "__main__":

    args = arg_parse()

    # Define source test dataset
    source_test_dataset = MNISTMDataset(args.mnistm_image_dir, transforms = transforms.Compose([
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean = [0.4631, 0.4666, 0.4195], std = [0.1979, 0.1845, 0.2083])]),
                                                                            mode = 'val')
    source_test_loader = torch.utils.data.DataLoader(source_test_dataset, batch_size=8*args.batch_size, shuffle=False, num_workers = 0)

    # Define target test dataset
    if args.dataset_type == 'svhn':
        target_test_dataset = SVHNDataset(args.svhn_image_dir, transforms = transforms.Compose([
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean = [0.4413, 0.4458, 0.4715], std = [0.1169, 0.1206, 0.1042])]),
                                                                            mode = 'val')
        target_test_loader = torch.utils.data.DataLoader(target_test_dataset, batch_size=8*args.batch_size, shuffle=False, num_workers = 0)
    elif args.dataset_type == 'usps':
        target_test_dataset = USPSDataset(args.usps_image_dir, transforms = transforms.Compose([
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean = [0.2573, 0.2573, 0.2573], std = [0.3373, 0.3373, 0.3373])]),
                                                                            mode = 'val')
        target_test_loader = torch.utils.data.DataLoader(target_test_dataset, batch_size=8*args.batch_size, shuffle=False, num_workers = 0)
    else:
        print("Invalid Dataset type!")

    if args.dataset_type == 'svhn':
        model = NN_SVHN().to(args.device)    
        load_checkpoint(args.svhn_ckpt, model)    
    else:
        model = NN_USPS().to(args.device)
        load_checkpoint(args.usps_ckpt, model)
    with torch.no_grad():
        inference(args, model)