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


# import os
# os.getcwd()
# count_jpg = 501
# count_xml = 501
# collection = "../dataset/AUbirds/test/"
# for i, filename in enumerate(os.listdir(collection)):
#     filesuffix = filename.split(".")
#     if filesuffix[1] =="jpg":
#         os.rename("../dataset/AUbirds/test/" + filename, "../dataset/AUbirds/all_labelled_data/" + str(count_jpg) + ".jpg")
#         count_jpg += 1
#     elif filesuffix[1] =="xml":
#         os.rename("../dataset/AUbirds/test/" + filename, "../dataset/AUbirds/all_labelled_data/" + str(count_xml) + ".xml")
#         count_xml += 1

# import os
# os.getcwd()
# # count_jpg = 488
# # count_xml = 488
# collection = "../dataset/AUbirds/test_order/"
# for i, filename in enumerate(os.listdir(collection)):
#     filesuffix = filename.split(".")
#     if filesuffix[1] =="xml":
#         os.rename("../dataset/AUbirds/test_order/" + filename, "../dataset/AUbirds/all_labels/" + filename)
#         # count_xml += 1
#     # elif filesuffix[1] =="jpg":
#     #     os.rename("../dataset/AUbirds/test/" + filename, "../dataset/AUbirds/test_order/" + str(count_jpg) + ".jpg")
#     #     count_jpg += 1
    

print(list(sorted(os.listdir(os.path.join("../dataset/dataset/", "all_images")))))