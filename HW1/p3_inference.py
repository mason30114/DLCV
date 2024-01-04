import pandas
import sys
import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
from torchvision import datasets, transforms, models
from p3_model_B import FCN8s, DeepLabV3_resnet101
from p3_model_A import FCN_32
#from p1_model_A import get_model
from p3_dataset import P3Dataset
from mean_iou_evaluate import mean_iou_score
#import collections
#from collections import OrderedDict
def inference(checkpoint_path, model, test_loader, test_dataset, setting):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    #criterion = nn.CrossEntropyLoss()
    model.eval() 
    #test_loss = 0
    #correct = 0
    #total = len(valid_loader.dataset)
    predict = []
    #iou_score = 0.0
    prediction = None

    with torch.no_grad():
        #va_loss = 0
        #all_preds = []
        #all_gt = []
        for images in test_loader:
            images = images.cuda()
            out = model(images)#['out']
            if setting == 'A':
                pred = out.argmax(dim=1)
            else:
                pred = out['out'].argmax(dim=1)
            if prediction is None:
                prediction = pred.cpu().numpy() 
            else: 
                prediction = np.concatenate((prediction, pred.cpu().numpy()), axis = 0)
            #pred = pred.cpu().numpy().astype(np.int64)
            #labels = labels.cpu().numpy().astype(np.int64)
            #all_preds.append(pred)
            #all_gt.append(labels)

        #va_loss /= len(valid_loader)
        #mIoU = mean_iou_score(np.concatenate(
        #    all_preds, axis=0), np.concatenate(all_gt, axis=0))



    #print("mIOU Score on Validaiton Sets =: {:.4f}".format(mIoU))
    return prediction
    # write and save file
    #save_csv(predict, test_dataset, filepath = sys.argv[2] + 'pred.csv')

# change predict img(512, 512) to RGB(512, 512, 3)
def channel_to_mask(pred, n_class):                                 #???????????????
    label_colors = np.array([[0, 255, 255], #Urban
                         [255, 255, 0], #Agriculture
                         [255, 0, 255], #Rangeland
                         [0, 255, 0], #Forest
                         [0, 0, 255], #Water
                         [255, 255, 255], #Barren
                         [0, 0, 0]]) #Unknown

    mask = np.zeros((512, 512, 3), np.uint8)
    for c in range(n_class):
        #print(pred)
        check = np.where(pred[:,:] == c)
        #print(check)
        mask[check[0], check[1], :] = label_colors[c]
        #print(mask.shape)
    return mask
    
test_path = sys.argv[1]
output_path = sys.argv[2]

test_dataset = P3Dataset(test_path, mode = 'test', 
transforms= transforms.Compose([
                                                                   #transforms.Resize((24)),    ###
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])]))


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers = 0)

setting = 'B'
#model = FCN_32(n_class=7).cuda()
model = DeepLabV3_resnet101().cuda()
pred = inference('p3_final_ckpt.pth', model, test_loader, test_dataset, setting)
#print(pred)
trans = transforms.ToPILImage()

for i in range(pred.shape[0]):

    if os.path.exists(os.path.join(output_path)) == False:
        print('folder not found!')
        #os.mkdir(os.path.join(out_mask_dir))

    path = os.path.join(output_path, test_dataset.get_path(i).split('/')[-1].split('_')[0]+'_mask.png')
    
    mask = channel_to_mask(pred[i], 7)
    mask = trans(mask)
    
    mask.save(os.path.join(path))
print('saved! total {} masks!'.format(pred.shape[0]))
#inference('./ckpt_model_A/best_37_55.52.pth', model, valid_loader, valid_dataset)