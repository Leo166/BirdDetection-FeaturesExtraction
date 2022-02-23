import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import xml.etree.ElementTree as ET

def load_image(image_ref, label_ref):

    image = Image.open(image_ref)
    # image.thumbnail(size, Image.ANTIALIAS)
    # summarize some details about the image
    print("Image Format: ", image.format)
    print("Image Mode: ", image.mode)
    print("Image Size: ", image.size)

    # Get the labels

    document = ET.parse(label_ref)
    root = document.getroot()

    all_box = []

    for item in root.findall(".//object/bndbox"):
        xmin = float(item.find('xmin').text)
        xmax = float(item.find('xmax').text)
        ymin = float(item.find('ymin').text)
        ymax = float(item.find('ymax').text)

        box = [xmin, xmax, ymin, ymax]
        all_box.append(box)

    size = (image.size[0], 600)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    
    # Create a Rectangle patch
    for i in all_box:
        anchor = (i[0], i[2])
        width = i[1]-i[0]
        height = i[3]-i[2]
        rect = patches.Rectangle(anchor, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    image.thumbnail(size, Image.ANTIALIAS)
    print("Image Size After: ", image.size)
    ax2.imshow(image)
    plt.show()
    return

root_dir = "D:/Adrien/cours/Master2/Thesis/Flying_birds_detection/Birds_detection/dataset/dataset"
# image_ref = "../dataset/AUbirds/train/621.jpg"
image_ref = "D:/Adrien/cours/Master2/Thesis/Flying_birds_detection/Birds_detection/dataset/dataset/all_images/549.jpg"
# label_ref = "../dataset/AUbirds/train/621.xml"
label_ref = "D:/Adrien/cours/Master2/Thesis/Flying_birds_detection/Birds_detection/dataset/dataset/all_labels/549.xml"
load_image(image_ref, label_ref)

# import os
# imgs = list(sorted(os.listdir(os.path.join(root_dir, "all_images"))))
# for i in imgs:
#     image_ref = os.path.join(root_dir, "all_images", i)
#     label_ref = os.path.join(root_dir, "all_labels", i.split(".")[0]+".xml")
#     load_image(image_ref, label_ref)

from torchvision.io import read_image
import torchvision.transforms as T
import cv2 as cv

# def show(imgs):
#     fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = T.ToPILImage()(img.to('cpu'))
#         axs[0, i].imshow(np.asarray(img), cmap="gray")
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#     plt.show()


# image = cv.imread(image_ref,0)
# ret,thresh1 = cv.threshold(image,127,255,cv.THRESH_BINARY)

# plt.imshow(thresh1,'gray',vmin=0,vmax=255)
# plt.show()