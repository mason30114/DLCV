import torch
import torch.nn as nn
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
#from p1_test import eval_n_save, save_checkpoint, load_checkpoint
from p3_dataset import P3Dataset
from torchvision import datasets, transforms, models
from p3_model_A import FCN_32
from mean_iou_evaluate import mean_iou_score
def train(num_epoch, model):
    # create loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0015, weight_decay = 0.0003, momentum = 0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    best_miou = 0.0
    miou = 0.0
    val_loss = 0
    best_epoch = -1
    # start training
    iteration = 0
    scheduler = CosineAnnealingLR(optimizer, T_max=100, last_epoch=-1)
    for epoch in range(num_epoch):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(images)
            #labels = labels.squeeze(1)
            loss = criterion(output, labels)
            #pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #correct += pred.eq(labels.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            #acc = 100. * correct / len(train_loader.dataset)
            if iteration % 100 == 0 and batch_idx > 0:
                print('Train Epoch: {} Loss: {:.6f} Best mIOU = {} Best Epoch = {}'
                      .format(epoch, loss.item(), best_miou, best_epoch))
        
            iteration += 1
            #scheduler.step()

        # evaluate and save model
        scheduler.step()
        miou = eval_n_save(epoch, model)
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            print("Saving best model... Best Miou is: {:.3f}".format(miou))
            save_checkpoint(os.path.join('./p3_ckpt_model_A/best_{}_{:.2f}.pth'.format(epoch,miou)), model, optimizer)
        
def eval_n_save(epoch, model):
    # evaluate model
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    score = 0
    iou_score = 0.0
    total = len(valid_loader.dataset)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0015, weight_decay = 0.0003, momentum = 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with torch.no_grad():
        output_list = np.zeros((len(valid_loader.dataset), 512, 512))
        labels_list = np.zeros((len(valid_loader.dataset), 512, 512))
        cur = 0
        for images, labels in valid_loader:
            images, labels = images.cuda(), labels.cuda()
            output = torch.argmax(model(images), dim=1)
            #output = model(images)
            output = output.cpu()
            labels = labels.squeeze(1)
            labels = labels.cpu()
            #test_loss += criterion(output, labels).item()
            output_list[cur, :, :] = output[0]
            labels_list[cur, :, :] = labels[0]
            cur += 1
            #pred = output.max(1, keepdim = True)[1]
            #correct += pred.eq(labels.view_as(pred)).sum().item()
            #score_tmp = mean_iou_score(output, labels)
            #score += score_tmp
            #for i in range(pred.shape[0]):#pred.shape[0]=BATCH_SIZES
            #    iou_score += mean_iou_score(pred[i][0].cpu().numpy(), labels[i].cpu().numpy())#for each outputs in a batch,claculate the miou
        iou_score = mean_iou_score(output_list, labels_list)
    print("mIOU Score on Validaiton Sets =: {:.4f}".format(iou_score))

    # save model
    #os.makedirs('ckpt', exist_ok=True)
    #filename = 'ckpt/%i_%.4f.pth'%(epoch,(correct / total))
    #save_checkpoint(os.path.join('./log/model_best_{}.pth'.format(epoch)), model, optimizer)

    return iou_score

# Save the model

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

BATCH_SIZE = 8                             

# Create Dataset and Dataloader
train_dataset = P3Dataset('./hw1_data/p3_data/train', mode = 'train', 
transforms= transforms.Compose([
                                                                     #transforms.Resize((224)),    ###
                                                                     #transforms.RandomCrop((224, 224)),   ###
                                                                     #transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)), #
                                                                     #transforms.RandomCrop((224, 224), padding=4),
                                                                     #transforms.RandomCrop((32, 32), padding=4),
                                                                     #transforms.RandomHorizontalFlip(),
                                                                     #transforms.RandomRotation(degrees = (-30, 30)),                                                   #
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])]))



valid_dataset = P3Dataset('./hw1_data/p3_data/validation', mode = 'val', 
transforms= transforms.Compose([
                                                                   #transforms.Resize((24)),    ###
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])]))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 0)

model = FCN_32(n_class=7).cuda()
train(100, model)