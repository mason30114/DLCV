import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from p2_dataset import MiniDataset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

def load_checkpoint(path, model):
    state = torch.load(path)
    model.load_state_dict(state)
    #optimizer.load_state_dict(state['optimizer'])

resnet = models.resnet50(weights = None)

'''Reference: https://github.com/lucidrains/byol-pytorch'''
learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool',
    use_momentum=False,
).cuda()

opt = torch.optim.Adam(learner.parameters(), lr=1e-4)

scheduler = CosineAnnealingLR(opt, T_max=500, last_epoch=-1)
pre_dataset = MiniDataset('./hw1_data/p2_data/mini/train', transforms= transforms.Compose([
                                                                     transforms.Resize((128, 128)),
                                                                     #transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)), #
                                                                     #transforms.RandomCrop((128, 128)),
                                                                     #RandomApply(transforms.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
                                                                     #RandomApply(transforms.ColorJitter(0.7, 0.7, 0.7, 0.2), p=0.1),
                                                                     #transforms.CenterCrop((128, 128)),
                                                                     #transforms.RandomHorizontalFlip(),
                                                                     #transforms.RandomRotation(degrees = (-30, 30)),                                                   #
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))

#def sample_unlabelled_images():
#train_set, valid_set = random_split(pre_dataset, [0.9, 0.1])
num_train_images = int(0.9 * len(pre_dataset))
train_set, valid_set = torch.utils.data.random_split(pre_dataset, [num_train_images, len(pre_dataset) - num_train_images])
train_loader = DataLoader(train_set, batch_size = 128, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size = 256, shuffle=False)
total_va_loss = 0.0
best_va_loss = 100.0
for epoch in range(500):
    print("=====Epoch {}=====".format(epoch))
    #images = sample_unlabelled_images()
    train_loss = []
    total_train_loss = 0.0
    for data in tqdm(train_loader):
        images = data.cuda()
        loss = learner(images)
        #print("Train Loss:{:.4f}".format(loss.item()))
        train_loss.append(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        #learner.update_moving_average() # update moving average of target encoder
    scheduler.step()
    total_train_loss = sum(train_loss) / len(train_loss)
    print("Train Loss:{:.4f}".format(total_train_loss))

    va_loss = []
    for data in valid_loader:
        images = data.cuda()
        with torch.no_grad():
            loss = learner(images).item()
        #print("Val Loss:{:.4f}".format(va_loss.item()))
        va_loss.append(loss)

    total_va_loss = sum(va_loss) / len(va_loss)
    print("Val Loss:{:.4f}".format(total_va_loss))
    if total_va_loss < best_va_loss:
        best_va_loss = total_va_loss
        torch.save(resnet.state_dict(), './p2_pre_checkpoint/Epoch_{}_{:.4f}.pt'.format(epoch, total_va_loss))
        print('Saved model... Best Val Loss = {:.4f}'.format(best_va_loss))

torch.save(resnet.state_dict(), './p2_pre_checkpoint/Epoch_Last.pt')
print('Saved model... Last')
    # save your improved network
    #if epoch % 10 == 0:
    #    torch.save(resnet.state_dict(), './pre_checkpoint/Epoch{}_net.pt'.format(epoch))