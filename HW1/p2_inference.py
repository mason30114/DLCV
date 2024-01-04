import pandas
import sys
import torch
import torch.nn as nn
import torchvision
import os
from torchvision import datasets, transforms, models
from p2_model import Resnet_MLP
#from p1_model_A import get_model
from p2_dataset import OfficeDataset
def inference(checkpoint_path, model, test_loader, test_dataset):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    #criterion = nn.CrossEntropyLoss()
    model.eval() 
    #test_loss = 0
    #correct = 0
    #total = len(valid_loader.dataset)
    predict = []
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for images in test_loader:
            images = images.cuda()
            output = model(images)
            #pred = output.max(1, keepdim = True)[1]
            #correct += pred.eq(labels.view_as(pred)).sum().item()

            # inference 
            _, test_pred = torch.max(output, 1) # get the index of the class with the highest probability
            for y in test_pred.cpu().numpy():
                predict.append(y)
        #print('Accuracy of the network on the test images:{:.4f}'.format(correct / total))
    # write and save file
    save_csv(predict, test_dataset, filepath = sys.argv[3])
    print('Save successfully!')

'''Reference: https://github.com/itsalicelee/DLCV-Fall-2021/blob/master/hw1/src/test.py'''
def save_csv(prediction, test_dataset, filepath=sys.argv[3]):
    id_list, img_id = create_ids(test_dataset)
    assert len(img_id) == len(prediction) 
    dict = {
        "id": id_list,
        "filename": img_id,
        "label": prediction
        }
    pandas.DataFrame(dict).to_csv(filepath, index=False)
    

def create_ids(test_dataset):
    filenames = []
    id_list = []
    cnt = 0
    for i in range(len(test_dataset)):
        tmp = test_dataset.filenames[i]
        tmp = os.path.split(tmp)
        filename_sh = tmp[1]
        filenames.append(filename_sh)
        id_list.append(cnt)
        cnt = cnt + 1
    return id_list, filenames

image_csv = sys.argv[1]
folder_path = sys.argv[2]
out_csv = sys.argv[3]

test_dataset = OfficeDataset(sys.argv[2], transforms= transforms.Compose([
                                                                   transforms.Resize((128, 128)),    ###
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]),
                                                                   mode = 'test')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers = 0)

model = Resnet_MLP().cuda()
model.mode_setting('Infer')
#model = get_model().cuda()
inference('p2_final_ckpt.pth', model, test_loader, test_dataset)
