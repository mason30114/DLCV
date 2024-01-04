import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
#from p1_test import eval_n_save, save_checkpoint, load_checkpoint
from p1_dataset import P1Dataset
from torchvision import datasets, transforms, models
from p1_model_B import get_model_pre

def train(num_epoch, model):
    # create loss function and optimizer
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0007, weight_decay = 0.0003, momentum = 0.9) #lr = 0.0007 wd = 0.0003: 69%
    #optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.train()
    best_acc = 0.0
    correct = 0
    # start training
    iteration = 0
    total = len(train_loader.dataset)
    scheduler = CosineAnnealingLR(optimizer, T_max=15, last_epoch=-1)
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = CosineAnnealingLR(optimizer, T_max=30, last_epoch=-1)
    for epoch in range(num_epoch):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(1), labels.cuda(1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            #acc = 100. * correct / len(train_loader.dataset)
            if iteration % 100 == 0 and batch_idx > 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Best Val Acc = {:.2f}%'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), 100 * best_acc))
                #print("Epoch:{} Train Acc: {:3f} ({}/{}) | Loss:{:.4f} | Best Val Acc:{:4f}".format(epoch, 100.* correct/(batch_idx * len(images)),int(correct), batch_idx * len(images),loss.item(), best_acc))
            iteration += 1
        #optimizer.step()
            #scheduler.step()

        # evaluate and save model
        scheduler.step()
        acc = eval_n_save(epoch, model)
        if acc > best_acc:
            best_acc = acc
            print("Saving best model... Best Acc is: {:.3f}".format(best_acc))
            save_checkpoint(os.path.join('./ckpt_model_B/best_{}_{:.2f}.pth'.format(epoch, 100 * best_acc)), model, optimizer)
        '''if acc > best_acc:
            best_acc = acc
            print("Saving best model... Best Acc is: {:.3f}".format(best_acc))
            save_checkpoint(, model, optimizer) '''
        
def eval_n_save(epoch, model):
    # evaluate model
    model.eval()
    total = len(valid_loader.dataset)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    test_loss = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0007, weight_decay = 0.0003, momentum = 0.9)
    #optimizer = torch.optim.Adam(model.parameters())
    for images, labels in valid_loader:
        images, labels = images.cuda(1), labels.cuda(1)
        output = model(images)
        test_loss += criterion(output, labels).item()
        pred = output.max(1, keepdim = True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    # save model
    #os.makedirs('ckpt', exist_ok=True)
    #filename = 'ckpt/%i_%.4f.pth'%(epoch,(correct / total))
    #save_checkpoint(os.path.join('./log/model_best_{}.pth'.format(epoch)), model, optimizer)

    return correct / total

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

BATCH_SIZE = 32                             

# Create Dataset and Dataloader
train_dataset = P1Dataset('./hw1_data/p1_data/train_50', transforms= transforms.Compose([
                                                                     transforms.Resize((224)),    ###
                                                                     #transforms.RandomCrop((224, 224)),   ###
                                                                     #transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)), #
                                                                     transforms.RandomCrop((224, 224), padding=4),
                                                                     #transforms.RandomCrop((32, 32), padding=4),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.RandomRotation(degrees = (-30, 30)),                                                   #
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean = [0.5077, 0.4813, 0.4312], std = [0.2000, 0.1986, 0.2034])]))


valid_dataset = P1Dataset('./hw1_data/p1_data/val_50', transforms= transforms.Compose([
                                                                   transforms.Resize((224)),    ###
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize(mean = [0.5077, 0.4813, 0.4312], std = [0.2000, 0.1986, 0.2034])]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 0)

model = get_model_pre().cuda(1)
train(15, model)