import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T

import torch.nn as nn
import torch

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from toolbox.dataset import BirdDataset

# load a pre-trained model on ImageNet for classification and return only the features
backbone = torchvision.models.resnet50(pretrained=True)


# # FasterRCNN needs to know the number of
# # output channels in a backbone. For resnet50, it's 2048
# # so we need to add it here
backbone.out_channels = 2048

# # let's make the RPN generate 5 x 3 anchors per spatial
# # location, with 5 different sizes and 3 different aspect
# # ratios. We have a Tuple[Tuple[int]] because each feature
# # map could potentially have different sizes and
# # aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

# # let's define what are the feature maps that we will
# # use to perform the region of interest cropping, as well as
# # the size of the crop after rescaling.
# # if your backbone returns a Tensor, featmap_names is expected to
# # be [0]. More generally, the backbone should return an
# # OrderedDict[Tensor], and in featmap_names you can choose which
# # feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# # put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

import torchvision.transforms as T

# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
#     # if train:
#     #     transforms.append(T.RandomHorizontalFlip(0.5))
#     if train:
#         transforms.append(A.Compose([A.Flip(0.5)], bbox_params={'format': 'pascal_voc'}))
#     return T.Compose(transforms)

def get_transform(train):
    return A.Compose([ A.RandomCrop(width=500, height=500),
                    A.Flip(0.5),
                    ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}) 


ROOT_DIR_DATA = "D:/Adrien/cours/Master2/Thesis/Flying_birds_detection/Birds_detection/dataset/dataset"
# ROOT_DIR_DATA = "D:\Adrien\cours\Master2\Thesis\Flying_birds_detection\Birds_detection\dataset\dataset"

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

# Test
# For Training
if __name__ == '__main__':
    #########
    # First test & visualisation
    ###########
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    images, targets = next(iter(data_loader_training))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)
    print(images[2].size())
    sample = images[2].permute(1,2,0).cpu().numpy()
    print(sample)
    print(sample.shape)
    print("Box", boxes)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    for box in boxes:
        cv2.rectangle(sample,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255, 0, 0), 2)
        
    ax.set_axis_off()
    ax.imshow((sample * 255).astype(np.uint8))
    plt.show()

    ############
    # Training #
    ############
    # model.to(device)
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # lr_scheduler = None

    # num_epochs = 2
    # itr = 1
    # epoch_loss = 0

    # for epoch in range(num_epochs):
    #     epoch_loss = 0
    #     iteration = 0
        
    #     for images, targets, image_ids in data_loader_training:
            
    #         images = list(image.to(device) for image in images)
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #         loss_dict = model(images, targets)

    #         losses = sum(loss for loss in loss_dict.values())
    #         loss_value = losses.item()

    #         epoch_loss += loss_value

    #         optimizer.zero_grad()
    #         losses.backward()
    #         optimizer.step()

    #         if itr % 50 == 0:
    #             print(f"Iteration #{itr} loss: {loss_value}")

    #         itr += 1
    #         iteration += 1
        
    #     # update the learning rate
    #     if lr_scheduler is not None:
    #         lr_scheduler.step()

    #     print(f"Epoch #{epoch} loss: {epoch_loss/iteration}")

    ########
    # Test #
    ########