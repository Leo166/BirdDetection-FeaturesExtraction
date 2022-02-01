#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 23:04:28 2021

@author: gducci
"""
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np


def list_frames(framenames):
    simlist = glob.glob(framenames)
    return simlist

# frames = list_frames("../dataset/AUbirds/test/34*.jpg")
frames = ["../dataset/AUbirds/test/104.jpg"]
# vid = cv2.VideoCapture("../dataset/videos/A003_02011701_C010.mp4")
# ret, frames = vid.read()
frames.sort()
# initializing the picture
im = cv2.imread("../dataset/AUbirds/test/104.jpg")
cv2.imshow('img',im)
cv2.waitKey(100)


coordinates_center = np.zeros([len(frames), 2])
pool_counter = 0
count = 0
for i in range(len(frames)):
    print("ok")
    print("Frame ", i)
    im = cv2.imread(frames[i])
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = 40
    ret,thresh_img = cv2.threshold(gray, thresh, 500, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0

    # TO DO: Filter out some contours
    
    for cnt in contours:
        # here it finds all the contours
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        roi=im[y:y+h,x:x+w]
        area_rect = w*h
        if area_rect > 4500 and area_rect< 100000:
            print("passed")
            print("x: ", x)
            print("y: ", y)
            print("w: ", w)
            print("h: ", h)
            cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
            
            coordinates_center[i, :] = [(x+w/2),(y+h/2)]
            print("coordinates: ", coordinates_center)
            cv2.imshow('img',im)
            print("ok")
            cv2.waitKey(120)
            count = count+1
            #else:
                #cv2.imshow('img',im)
                #cv2.waitKey(120)
                #count = count+1

print("done")
#cv2.imwrite(str(idx) +'foto'+ '.jpg', im)

fig, ax = plt.subplots()
img = cv2.imread("../dataset/AUbirds/test/134.jpg")
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.scatter(coordinates_center[0,0], coordinates_center[0,1])
plt.show()
