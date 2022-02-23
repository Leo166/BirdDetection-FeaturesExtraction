import cv2
import imutils
import glob
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
import json

# Source
# https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# https://levelup.gitconnected.com/build-your-own-motion-detector-using-webcam-and-opencv-in-python-ff5bdb78a55e

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

# Image section based on temporal differentiation. We assume that the bird is a moving object 
# through different frames. Based on motion-detection using OpenCv.
# Inputs: pair of images.
# Outputs: -bounding box around the object of interest (on the original image?). 
#          -Object of interest (not the original images) to classify with a NN. Then if the object of interest isn't
#           a bird, ignore it.
def temporal_differentiation(ref_frame, next_frame):

    # ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    # ref_gray = cv2.GaussianBlur(ref_gray, (21,21), 0)
    
    # next_gray = next_frame
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    # next_gray = cv2.GaussianBlur(next_gray, (21,21), 0) 

    # compute the absolute difference between the current frame and reference frame
    frameDelta = cv2.absdiff(ref_frame, next_gray)
    # cv2.imshow('Difference frame', frameDelta)
    # frameDelta = next_gray
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.adaptiveThreshold(frameDelta,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow('threshold frame', thresh)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # cv2.imshow('Dilate frame', thresh)
    (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts) #get all the contours

    return cnts

transforms = torch.nn.Sequential(
    T.Resize([256, ]),
    T.CenterCrop(224),
    T.ConvertImageDtype(torch.float),
)

def prediction_model(input):
    input = T.ToTensor()(input)
    # input = torch.from_numpy(np.moveaxis(input, -1, 0))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = input.to(device)
    print("Taille input", input.size())
    transformed_input = transforms(input)
    print("Taille input", transformed_input.size())
    # cv2.imshow("Input", transformed_input.cpu().detach().numpy())
    transformed_input = transformed_input[None, ...]

    model_conv = torchvision.models.resnet18(pretrained=True)
    model_conv.eval()
    output = model_conv(transformed_input)
    output = output.argmax(dim=1)

    with open('labels_ImageNet/imagenet_class_index.json', 'r') as labels_file:
        labels = json.load(labels_file)
    return labels[str(output.item())]

# vid = cv2.VideoCapture("../dataset/videos/A003_02011713_C021.mp4")
vid = cv2.VideoCapture("../dataset/Personal_videos/videos/VID_20211105_120528.mp4")
first_frame = None
last_coords = None
while vid.isOpened():
    ret, frame = vid.read()
    print(ret)
    if ret == True:
        frame = cv2.resize(frame, (1280,500))
        if first_frame is None:
            first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # first_frame = cv2.GaussianBlur(first_frame, (21,21), 0)
            continue
        
        # Object(s) of interest on each frameq
        roi = temporal_differentiation(first_frame, frame)
        # first_frame = frame
        # cv2.imshow('img1',first_frame)

        # loop over the contours
        coords = []
        for c in roi:
            # if the contour is too small, ignore it
            # print(cv2.contourArea(c))
            if cv2.contourArea(c) < 50:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            coords.append((x, y, w, h))
            # If not moving skip
            if last_coords is not None:
                if (x, y, w, h) in last_coords:
                    continue

            # Get the specific roi to put to the NN to determine if it is a bird
            crop_frame = frame[y:y+h, x:x+w]
            # print(crop_frame.shape)
            # print(np.moveaxis(crop_frame, 2, 0).shape)
            # print(T.ToTensor()(crop_frame).size())
            itsabird = prediction_model(crop_frame)
            print(itsabird)
            # cv2.imshow("cropped", crop_frame)
            # show([T.ToTensor()(crop_frame)])

            # Visualize the roi and the result
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # print(coords)
        last_coords = coords

        cv2.imshow('img',frame)
        # # Press Q on keyboard to  exit
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    else:
        break
