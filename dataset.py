import torch
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader

# Manage multiple versions of python with pip
# py -3.8 -m pip install package
#https://stackoverflow.com/questions/2812520/dealing-with-multiple-python-versions-and-pip


# class BirdDataset(Dataset):
#     """Class to charecterize the bird dataset"""

#     def __init__(self, root_dir, samples_name, transform=None):
#         """
#         Args:
#             samples_name (list of int): list of number corresponding to the name of images
#             root_dir (string): Directory with all the images
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         # intiate
    
#     def __len__(self):
#         """Returns the size of the dataset"""
#         return 0

#     def __getitem__(self, idx):
#         """Loads and returns a sample from the dataset at the given index idx"""
#         return image, target