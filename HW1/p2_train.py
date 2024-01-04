import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from p2_model import Resnet_MLP, RandomApply
from p2_dataset import OfficeDataset
def train(num_epoch, model):
    # create loss function and optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0007, weight_decay = 0.0003, momentum = 0.9) #lr = 0.0007 wd = 0.0003: 69%
    optimizer = torch.optim.Adam(model.parameters(), lr = 4e-4)
    #optimizer_c = torch.optim.Adam(classifier.parameters(), lr = 3e-4)
    criterion = nn.CrossEntropyLoss()
    #backbone.train()
    model.train()
    correct = 0
    best_acc = 0.0
    # start training
    iteration = 0
    total = len(train_loader.dataset)
    #scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.75)
    #scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
    #scheduler_c = StepLR(optimizer_c, step_size=40, gamma=0.5)
    #scheduler = CosineAnnealingLR(optimizer, T_max=30, last_epoch=-1)
    for epoch in range(num_epoch):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            #optimizer_b.zero_grad()
            optimizer.zero_grad()
            #output = backbone(images)
            output = model(images)
            loss = criterion(output, labels)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()
            #optimizer_c.step()

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
        #scheduler_c.step()       
        acc = eval_n_save(epoch, model)
        if acc > best_acc:
            best_acc = acc
            print("Saving best model... Best Acc is: {:.3f}".format(best_acc))
            save_checkpoint(os.path.join('./p2_checkpoint/Resnet_MLP_best_{}_{:.2f}_C_Cos.pth'.format(epoch, 100 * best_acc)), model, optimizer)
            #save_checkpoint(os.path.join('./drive/MyDrive/classifier_best_{}_{:.2f}.pth'.format(epoch, 100 * best_acc)), classifier, optimizer_c)
        '''if acc > best_acc:
            best_acc = acc
            print("Saving best model... Best Acc is: {:.3f}".format(best_acc))
            save_checkpoint(, model, optimizer) '''
# Valid the network
def eval_n_save(epoch, model):
    # evaluate model
    #backbone.eval()
    model.eval()
    total = len(valid_loader.dataset)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    test_loss = 0
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0007, weight_decay = 0.0003, momentum = 0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr = 4e-4)
    #optimizer_c = torch.optim.Adam(classifier.parameters(), lr = 5e-4)
    for images, labels in valid_loader:
        images, labels = images.cuda(), labels.cuda()
        #output = backbone(images)
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

BATCH_SIZE = 128                        # 400:50%    300:53%    200:53%  128:56%  64:57%(UNSTABLE)  32:58%(UNSTABLE)





# Create Dataset and Dataloader
train_dataset = OfficeDataset('./hw1_data/p2_data/office/train', transforms= transforms.Compose([
                                                                     transforms.Resize((128, 128)),    ###
                                                                     #transforms.RandomCrop((224, 224)),   ###
                                                                     #transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)), #
                                                                     transforms.RandomCrop((128, 128)),
                                                                     RandomApply(transforms.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
                                                                     RandomApply(transforms.ColorJitter(0.7, 0.7, 0.7, 0.2), p=0.1),
                                                                     #transforms.CenterCrop((128, 128)),
                                                                     #transforms.RandomCrop((32, 32), padding=4),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.RandomRotation(degrees = (-30, 30)),                                                   #
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))


valid_dataset = OfficeDataset('./hw1_data/p2_data/office/val', transforms= transforms.Compose([
                                                                   transforms.Resize((128, 128)),    ###
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 0)

#backbone = models.resnet50(pretrained=False).cuda()
#backbone.load_state_dict(torch.load('./drive/MyDrive/pre_checkpoint/pretrain_model_SSL.pt'), strict=False)
#classifier = MLP().cuda()
model = Resnet_MLP().cuda()
model.mode_setting('C')
#model = model.cuda()
#model.load_state_dict(torch.load('./p2_best_checkpoint/Resnet_MLP_best_C.pth'), strict=False)
train(300, model)