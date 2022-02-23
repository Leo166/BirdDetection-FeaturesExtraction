import torchvision
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
import json

# Load ResNet18 to do first classification on any type of images.
# There is still no transfer learning or training
# Inputs: 224x224 images



plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

# read_image -> Tensor[image_channels, image_height, image_width]
# OpenCv frame -> numpy[image_height, image_width, image_channels]
# Pil -> numpy[image_height, image_width, image_channels]
dog1 = read_image('images/dog1.jpg')
# dog1 = read_image('images/dog2.jpg')
# dog1 = read_image('images/fourmi.jpg')
dog2 = read_image('images/zebre.jpg')
# print(dog2)
# show([dog2])


import torch.nn as nn

transforms = torch.nn.Sequential(
    T.Resize([256,]),
    T.CenterCrop(224),
    T.ConvertImageDtype(torch.float), # convert to this type and scale if need it [0,255] -> [0,1]
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)



def prediction(dog1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dog1 = dog1.to(device)
    print(dog1.size())
    show([dog1])
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
    model_conv.eval()
    outputs = model_conv(inputs)
    print(outputs[0, 207])
    print(outputs.argmax(dim=1))
    outputs = outputs.argmax(dim=1)

    with open('labels_ImageNet/imagenet_class_index.json', 'r') as labels_file:
        labels = json.load(labels_file)
    print(labels[str(outputs.item())])

prediction(dog1)
# prediction(dog2)


# for i, (pred, pred_scripted) in enumerate(zip(res, res_scripted)):
#     assert pred == pred_scripted
#     print(f"Prediction for Dog {i + 1}: {labels[str(pred.item())]}")