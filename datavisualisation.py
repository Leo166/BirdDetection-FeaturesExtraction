import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import xml.etree.ElementTree as ET

def load_image(image_ref, label_ref):
    image = Image.open(image_ref)
    # summarize some details about the image
    print(image.format)
    print(image.mode)
    print(image.size)

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



    fig, ax = plt.subplots()
    ax.imshow(image)

    # Create a Rectangle patch
    for i in all_box:
        anchor = (i[0], i[2])
        width = i[1]-i[0]
        height = i[3]-i[2]
        rect = patches.Rectangle(anchor, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    return

image_ref = "../dataset/AUbirds/test/104.jpg"
label_ref = "../dataset/AUbirds/test/104.xml"
load_image(image_ref, label_ref)