from sklearn import datasets
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
import sklearn.model_selection as skms
from toolbox.dataset import BirdDataset
import matplotlib.pyplot as plt

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomOrder([T.RandomCrop((375, 375)),
                                        T.RandomHorizontalFlip(),
                                        T.RandomVerticalFlip()
                                        ]))
    transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(transforms)

ROOT_DIR_DATA = "D:/Adrien/cours/Master2/Thesis/Flying_birds_detection/Birds_detection/dataset/dataset/"

# instantiate dataset objects
ds = BirdDataset(ROOT_DIR_DATA, get_transform(train=True))
ds_test = BirdDataset(ROOT_DIR_DATA, get_transform(train=False))
# image, target = ds.__getitem__(0)
# print(image.size())
# show([image])
# print(target)

# set hyper-parameters
params = {'batch_size': 24, 'num_workers': 4}
num_epochs = 100
num_classes = 2
num_coord = 4

# model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torchvision.models.resnet50(pretrained=True).to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes + num_coord)

# instantiate data loaders
# split the dataset in train and test set
indices = torch.randperm(len(ds)).tolist()
dataset = torch.utils.data.Subset(ds, indices[:-50])
dataset_test = torch.utils.data.Subset(ds_test, indices[-50:])

def collate_fn(batch):
    return tuple(zip(*batch))
# define training and validation data loaders
data_loader_training = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=collate_fn, **params)
data_loader_test = torch.utils.data.DataLoader(dataset_test, shuffle=True, collate_fn=collate_fn, **params)

# For Training
# if __name__ == '__main__':
#     images,targets = next(iter(data_loader_training))
#     images = list(image for image in images)
#     targets = [{k: v for k, v in t.items()} for t in targets]
#     print(targets)


# instantiate optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

import sys

# loop over epochs
if __name__ == '__main__':
    for epoch in range(num_epochs):
    # train the model
        model.train()
        train_loss = list()
        train_acc = list()
        for batch in data_loader_training:
            x, y = batch
            print(x)
            print(y)
            print("ok")
            sys.exit("Error message")
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            # predict bird species
            y_pred = model(x)
            # calculate the loss
            loss = F.cross_entropy(y_pred, y)
            # backprop & update weights
            loss.backward()
            optimizer.step()
            # calculate the accuracy
            acc = skms.accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])
            
            train_loss.append(loss.item())
            train_acc.append(acc)
                    
        # validate the model
        model.eval()
        val_loss = list()
        val_acc = list()
        with torch.no_grad():
            for batch in data_loader_test:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                # predict bird species
                y_pred = model(x)
                
                # calculate the loss
                loss = F.cross_entropy(y_pred, y)
                # calculate the accuracy
                acc = skms.accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])
            val_loss.append(loss.item())
            val_acc.append(acc)
        # adjust the learning rate
        scheduler.step()

# # test the model
# true = list()
# pred = list()
# with torch.no_grad():
#     for batch in test_loader:
#         x, y = batch
#         x = x.to(DEVICE)
#         y = y.to(DEVICE)
#         y_pred = model(x)
#         true.extend([val.item() for val in y])
#         pred.extend([val.item() for val in y_pred.argmax(dim=-1)])
# # calculate the accuracy 
# test_accuracy = skms.accuracy_score(true, pred)
# print('Test accuracy: {:.3f}'.format(test_accuracy)