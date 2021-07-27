# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:45:43 2021
newE
struggles with white powder
@author: chomi
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# image = cv.imread('C:/Users/chomi/Desktop/all_small/A0159-0009.png')
# image = cv.imread('C:/Users/chomi/Desktop/est.jpg')
def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation = inter)
    return resized

crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
for filename in os.listdir('C:/Users/chomi/Desktop/all_crop/'):
    if(filename.endswith('.png')):
        print(filename)
        image = cv.imread('C:/Users/chomi/Desktop/all_crop/' + filename)
        temp = image_resize(image, width = 400)
        cv.imshow('og', temp)
        pixs = image.reshape((-1,3))
        pixs = np.float32(pixs)
        
        k = 2
        _,labels, (centers) = cv.kmeans(pixs, k, None, crit, 10, cv.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        labels_ = labels.flatten()
        segim = centers[labels_.flatten()]
        segim1 = segim.reshape(image.shape)
        tempp = image_resize(segim1, width = 400)
        cv.imshow('a',tempp)
        
        k = 3
        pixs = image.reshape((-1,3))
        pixs = np.float32(pixs)
        _,labels, (centers) = cv.kmeans(pixs, k, None, crit, 10, cv.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        labels_ = labels.flatten()
        segim = centers[labels_.flatten()]
        segim2 = segim.reshape(image.shape)
        temppp = image_resize(segim2, width = 400)
        cv.imshow('ab',temppp)
        
        k = 4
        pixs = image.reshape((-1,3))
        pixs = np.float32(pixs)
        _,labels, (centers) = cv.kmeans(pixs, k, None, crit, 10, cv.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        labels_ = labels.flatten()
        segim = centers[labels_.flatten()]
        segim3 = segim.reshape(image.shape)
        tempppp = image_resize(segim3, width = 400)
        cv.imshow('abc', tempppp)
        
        key = cv.waitKey(0)
        if(key == ord('0')):
            break
        elif(key == ord('1')):
            cv.imwrite('C:/Users/chomi/Desktop/maskss/' + filename, segim1)
        elif(key == ord('2')):
            cv.imwrite('C:/Users/chomi/Desktop/maskss/' + filename, segim2)
        elif(key == ord('3')):
            cv.imwrite('C:/Users/chomi/Desktop/maskss/' + filename, segim3)
        elif(key == ord('m')):
            continue
            
        cv.destroyAllWindows()
    
cv.destroyAllWindows()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    