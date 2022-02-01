import torchvision
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T

plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

dog1 = read_image('C:/Users/Scorpion/Desktop/dog1.jpg')
dog1 = read_image('C:/Users/Scorpion/Desktop/dog2.jpg')
dog1 = read_image('C:/Users/Scorpion/Desktop/fourmi.jpg')
dog1 = read_image('C:/Users/Scorpion/Desktop/zebre.jpg')
show([dog1])


import torch.nn as nn

transforms = torch.nn.Sequential(
    T.Resize([256, ]),
    T.CenterCrop(224),
    T.ConvertImageDtype(torch.float),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dog1 = dog1.to(device)
print(dog1.size())

transformed_dog1 = transforms(dog1)
show([transformed_dog1])
print(transformed_dog1.size())
transformed_dog1 = transformed_dog1[None, ...]
print(transformed_dog1.size())

model_conv = torchvision.models.resnet18(pretrained=True)
inputs = transformed_dog1
# batch = torch.stack([dog1, dog2]).to(device)
# pytorch (and most other DL toolboxes) expects a batch of images as an input 
# output = model(data[None, ...])  
outputs = model_conv(inputs).argmax(dim=1)
print(outputs)

import json

with open('imagenet_class_index.json', 'r') as labels_file:
    labels = json.load(labels_file)
print(labels[str(outputs.item())])

# for i, (pred, pred_scripted) in enumerate(zip(res, res_scripted)):
#     assert pred == pred_scripted
#     print(f"Prediction for Dog {i + 1}: {labels[str(pred.item())]}")