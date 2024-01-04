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
from p3_parser import arg_parse
from p3_model import UNet, NN_USPS, NN_SVHN
from p3_dataset import MNISTMDataset, SVHNDataset, USPSDataset
from torchvision import datasets, transforms, models

def DANN_train(args, model, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    acc = 0.0
    iteration = 0
    total_len = min(len(source_train_loader), len(target_train_loader))

    for epoch in range(args.num_epoch):
        for batch_idx, ((source_images, source_class_labels), (target_images, target_labels)) in tqdm((enumerate(zip(source_train_loader, target_train_loader)))):
            model.train()
            source_images = source_images.to(args.device)
            source_class_labels = np.array(source_class_labels, int)
            source_class_labels = torch.Tensor(source_class_labels).type(torch.LongTensor)
            source_class_labels = source_class_labels.to(args.device)
            target_images = target_images.to(args.device)
            p = float(batch_idx + epoch * total_len) / args.num_epoch / total_len
            optimizer.param_groups[0]['lr'] = args.lr / (1.0 + 10 * p) ** 0.75
            lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
            optimizer.zero_grad()
            source_class_pred, source_domain_pred = model(source_images, lambda_)
            source_class_loss = criterion(source_class_pred, source_class_labels)
            source_domain_labels = torch.zeros(len(source_images), dtype = int).to(args.device)        # source = 0
            source_domain_loss = criterion(source_domain_pred, source_domain_labels)

            target_class_pred, target_domain_pred = model(target_images, lambda_)
            target_domain_labels = torch.ones(len(target_images), dtype = int).to(args.device)         # target = 1
            target_domain_loss = criterion(target_domain_pred, target_domain_labels)     

            loss = source_class_loss + source_domain_loss + target_domain_loss      
            loss.backward()
            optimizer.step()
            if iteration % 10 == 0 and batch_idx > 0:
                print('Train Epoch: {} Loss: {:.6f}  lr = {:.6f} Best Val Acc = {:.2f}%'.format(epoch, loss.item(), optimizer.param_groups[0]['lr'], 100 * best_acc))
            iteration += 1

        #scheduler.step()            
        with torch.no_grad():
            model.eval()
            acc = eval_n_save(args, model)
            if acc > best_acc:
                best_acc = acc
                print("Saving best model... Best Acc is: {:.3f}".format(best_acc))

        if args.dataset_type == 'svhn':
            save_checkpoint(os.path.join(args.ckpt_dir, 'svhn_dann3_{}_{:.2f}.pth'.format(epoch, 100 * best_acc)), model, optimizer)

        elif args.dataset_type == 'usps':
            save_checkpoint(os.path.join(args.ckpt_dir, 'usps_dann4_{}_{:.2f}.pth'.format(epoch, 100 * best_acc)), model, optimizer)

def Normal_train(args, model, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    acc = 0.0
    iteration = 0
    total_len = len(train_loader)

    for epoch in range(args.num_epoch):
        for batch_idx, (images, labels) in tqdm(enumerate(train_loader)):
            model.train()
            images = images.to(args.device)
            labels = np.array(labels, int)
            labels = torch.Tensor(labels).type(torch.LongTensor)
            labels = labels.to(args.device)
            p = float(batch_idx + epoch * total_len) / args.num_epoch / total_len
            optimizer.param_groups[0]['lr'] = args.lr / (1.0 + 10 * p) ** 0.75
            lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
            optimizer.zero_grad()
            class_pred, domain_pred = model(images, lambda_)
            loss = criterion(class_pred, labels)
            loss.backward()
            optimizer.step()
            if iteration % 10 == 0 and batch_idx > 0:
                print('Train Epoch: {} Loss: {:.6f}  Best Val Acc = {:.2f}%'.format(epoch, loss.item(), 100 * best_acc))               
            iteration += 1

        #scheduler.step()            
        with torch.no_grad():
            model.eval()
            acc = eval_n_save(args, model)
        if acc > best_acc:
            best_acc = acc
            print("Saving best model... Best Acc is: {:.3f}".format(best_acc))

        if args.train_type == 'Source':
            if args.dataset_type == 'svhn':
                save_checkpoint(os.path.join(args.ckpt_dir, 'svhn_source_{}_{:.2f}.pth'.format(epoch, 100 * best_acc)), model, optimizer)

            elif args.dataset_type == 'usps':
                save_checkpoint(os.path.join(args.ckpt_dir, 'usps_source_{}_{:.2f}.pth'.format(epoch, 100 * best_acc)), model, optimizer)

        elif args.train_type == 'Target':
            if args.dataset_type == 'svhn':
                save_checkpoint(os.path.join(args.ckpt_dir, 'svhn_target_{}_{:.2f}.pth'.format(epoch, 100 * best_acc)), model, optimizer)

            elif args.dataset_type == 'usps':
                save_checkpoint(os.path.join(args.ckpt_dir, 'usps_target_{}_{:.2f}.pth'.format(epoch, 100 * best_acc)), model, optimizer)

def eval_n_save(args, model):
    total_len = len(target_valid_loader.dataset)
    correct = 0
    for images, labels in target_valid_loader:
        images = images.to(args.device)
        labels = np.array(labels, int)
        labels = torch.Tensor(labels).type(torch.LongTensor)
        labels = labels.to(args.device)
        lambda_ = 0.0
        class_pred, domain_pred = model(images, lambda_)
        pred = class_pred.max(1, keepdim = True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
    
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total_len))
    return correct / total_len

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),}
    torch.save(state, checkpoint_path)

if __name__ == "__main__":

    args = arg_parse()

    # Define training dataset
    if args.train_type == 'DANN':
        source_train_dataset = MNISTMDataset(args.mnistm_image_dir, transforms = transforms.Compose([                                                   
                                                                                transforms.ToTensor(),
                                                                                transforms.Normalize(mean = [0.4631, 0.4666, 0.4195], std = [0.1979, 0.1845, 0.2083])]))
        source_train_loader = torch.utils.data.DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 0)
        if args.dataset_type == 'svhn':
            target_train_dataset = SVHNDataset(args.svhn_image_dir, transforms = transforms.Compose([
                                                                                 transforms.ToTensor(),
                                                                                 transforms.Normalize(mean = [0.4413, 0.4458, 0.4715], std = [0.1169, 0.1206, 0.1042])]))
            target_train_loader = torch.utils.data.DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 0)  
        elif args.dataset_type == 'usps':
            target_train_dataset = USPSDataset(args.usps_image_dir, transforms = transforms.Compose([
                                                                                transforms.ToTensor(),
                                                                                transforms.Normalize(mean = [0.2573, 0.2573, 0.2573], std = [0.3373, 0.3373, 0.3373])]))
            target_train_loader = torch.utils.data.DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 0)
        else:
            print("Invalid Dataset type!")

    elif args.train_type == 'Source':
        train_dataset = MNISTMDataset(args.mnistm_image_dir, transforms = transforms.Compose([                                                   
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize(mean = [0.4631, 0.4666, 0.4195], std = [0.1979, 0.1845, 0.2083])]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 0)
    elif args.train_type == 'Target':
        if args.dataset_type == 'svhn':
            train_dataset = SVHNDataset(args.svhn_image_dir, transforms = transforms.Compose([
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(mean = [0.4413, 0.4458, 0.4715], std = [0.1169, 0.1206, 0.1042])]))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 0)
        elif args.dataset_type == 'usps':
            train_dataset = USPSDataset(args.usps_image_dir, transforms = transforms.Compose([
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(mean = [0.2573, 0.2573, 0.2573], std = [0.3373, 0.3373, 0.3373])]))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 0)
    else:
        print("Invalid Dataset type!")

    # Define validation dataset
    if args.dataset_type == 'svhn':
        target_valid_dataset = SVHNDataset(args.svhn_image_dir, transforms = transforms.Compose([
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize(mean = [0.4413, 0.4458, 0.4715], std = [0.1169, 0.1206, 0.1042])]),
                                                                             mode = 'val')
        target_valid_loader = torch.utils.data.DataLoader(target_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 0)
    elif args.dataset_type == 'usps':
        target_valid_dataset = USPSDataset(args.usps_image_dir, transforms = transforms.Compose([
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize(mean = [0.2573, 0.2573, 0.2573], std = [0.3373, 0.3373, 0.3373])]),
                                                                             mode = 'val')
        target_valid_loader = torch.utils.data.DataLoader(target_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 0)
    else:
        print("Invalid Dataset type!")

    if args.dataset_type == 'svhn':
        model = NN_SVHN().to(args.device)        
    else:
        model = NN_USPS().to(args.device)          
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min = 1e-6)

    if args.train_type == 'DANN':
        DANN_train(args, model, optimizer, scheduler)
    else:
        Normal_train(args, model, optimizer, scheduler)
  

