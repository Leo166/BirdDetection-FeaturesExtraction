import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
import xml.etree.ElementTree as ET
# from torch.utils.data import Dataset, DataLoader

# Manage multiple versions of python with pip
# py -3.8 -m pip install package
#https://stackoverflow.com/questions/2812520/dealing-with-multiple-python-versions-and-pip

# Inspired by torchvision example: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

class BirdDataset(torch.utils.data.Dataset):
    """Class to charecterize the bird dataset"""

    def __init__(self, root_dir, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root_dir
        self.transforms = transforms
        
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "all_images"))))  # list of all image names - jpg
        self.boxes = list(sorted(os.listdir(os.path.join(root_dir, "all_labels")))) # list of all image names - xml
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """Loads and returns a sample from the dataset at the given index idx"""
        # load images and boxes
        img_path = os.path.join(self.root, "all_images", self.imgs[idx])
        box_path = os.path.join(self.root, "all_labels", self.boxes[idx])
        img = Image.open(img_path).convert("RGB")

        # get boxes for each bird
        document = ET.parse(box_path)
        root = document.getroot()
        boxes = []
        for item in root.findall(".//object/bndbox"):
            xmin = float(item.find('xmin').text)
            xmax = float(item.find('xmax').text)
            ymin = float(item.find('ymin').text)
            ymax = float(item.find('ymax').text)

            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
        num_objs = len(boxes)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # only one class : a bird

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target