import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

# Image section based on temporal differentiation. We assume that the bird is a moving object 
# through different frames. Based on motion-detection using OpenCv.
# Inputs: pair of images.
# Outputs: -bounding box around the object of interest (on the original image?). 
#          -Object of interest (not the original images) to classify with a NN. Then if the object of interest isn't
#           a bird, ignore it.

def temporal_differentiation(image1, image2):

    return 