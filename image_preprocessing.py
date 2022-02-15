from re import I
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
from PIL import Image
from pathlib import Path
import imagesize
import xml.etree.ElementTree as ET

import torchvision
from torchvision.io import read_image


# Get the Image Resolutions
# root = "D:/Adrien/cours/Master2/Mémoire/Flying_birds_detection/Birds_detection/dataset/AUbirds/train/"
# imgs = [img.name for img in Path(root).iterdir() if img.suffix == ".jpg"]
# img_meta = {}
# for f in imgs: img_meta[str(f)] = imagesize.get(root+f)

# # # Convert it to Dataframe and compute aspect ratio
# img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', inplace=False)
# img_meta_df[["Width", "Height"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index=img_meta_df.index)
# img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)

# print(f'Total Nr of Images in the dataset: {len(img_meta_df)}')
# print(img_meta_df.head())
# print(img_meta_df)


# Scale the bounding boxes with the image transformation
def get_coordinates(image_ref, label_ref):
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
    return all_box

def transform_coordinate(coord, orig_size, target_size):
    """
    https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box/49468149
    coord = list of list of 4 coordinates
    """
    y_ = orig_size[1]
    x_ = orig_size[0]

    x_scale = target_size / x_
    y_scale = target_size / y_

    transformed_coord = []
    for i in coord:
        x = int(np.round(i[0] * x_scale))
        y = int(np.round(i[2] * y_scale))
        xmax = int(np.round(i[1] * x_scale))
        ymax = int(np.round(i[3] * y_scale))
        transformed_coord.append([x, xmax, y, ymax])
    return transformed_coord

def show_image(image_ref, all_box):
    # image = read_image(image_ref) # give a tensor, difficult to manimulate sauf for CNN
    image = Image.open(image_ref) # (height, width, channels)
    # summarize some details about the image
    print("Image Format: ", image.format)
    print("Image Mode: ", image.mode)
    print("Image Size: ", image.size)
    orig_size = image.size

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    
    # Create bounding boxes on original image
    for i in all_box:
        anchor = (i[0], i[2])
        width = i[1]-i[0]
        height = i[3]-i[2]
        rect = patches.Rectangle(anchor, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)

    size = (224, 224)
    image = image.resize(size,Image.ANTIALIAS)
    print(image.size)
    ax2.imshow(image)

    # Create bounding boxes on transformed image
    all_box = transform_coordinate(all_box, orig_size, 224)
    for i in all_box:
        anchor = (i[0], i[2])
        width = i[1]-i[0]
        height = i[3]-i[2]
        rect = patches.Rectangle(anchor, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)

    plt.show()
    return

# image_ref = "../dataset/AUbirds/test/13.jpg"
# label_ref = "../dataset/AUbirds/test/13.xml"
# coordinate_boxes = get_coordinates(image_ref, label_ref)
# show_image(image_ref, coordinate_boxes)

# Get the mean and standard deviation of the image dataset
# root = "D:/Adrien/cours/Master2/Mémoire/Flying_birds_detection/Birds_detection/dataset/AUbirds/train/"
# imgs = [img.name for img in Path(root).iterdir() if img.suffix == ".jpg"]
# img_meanR = []
# img_meanG = []
# img_meanB = []
# img_stdR = []
# img_stdG = []
# img_stdB = []
# for i in imgs:
#     linktoimage = "../dataset/AUbirds/train/" + i
#     image = cv2.imread(linktoimage) # BGR order
#     valR = np.reshape(image[:,:,2]/255, -1)
#     valG = np.reshape(image[:,:,1]/255, -1)
#     valB = np.reshape(image[:,:,0]/255, -1)
#     img_meanR.append(np.mean(valR))
#     img_meanG.append(np.mean(valG))
#     img_meanB.append(np.mean(valB))
#     img_stdR.append(np.std(valR))
#     img_stdG.append(np.std(valG))
#     img_stdB.append(np.std(valB))

# print("Mean", [np.mean(img_meanB), np.mean(img_meanG), np.mean(img_meanR)])
# print("Standard Deviation", [np.mean(img_stdB), np.mean(img_stdG), np.mean(img_stdR)])
# Mean [0.5976994951230858, 0.5536538457122057, 0.5083647087239678]
# Standard Deviation [0.08991753724992092, 0.08064956781380776, 0.075949670959734]