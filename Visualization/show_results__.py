# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 01:01:40 2020

@author: Eric Bianchi
"""

import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm   
import os
import numpy

def spectrum_score(logits, targets):
    length = len(logits)
    difference = logits - targets
    abs_val = numpy.absolute(difference)
    sum_val = sum(abs_val)
    norm = sum_val / length
    spectrum_score = norm

    return spectrum_score
def generate_images(model, image_path, name, destination_mask, destination_overlays):
    
    if not os.path.exists(destination_mask): # if it doesn't exist already
        os.makedirs(destination_mask)  
        
    if not os.path.exists(destination_overlays): # if it doesn't exist already
        os.makedirs(destination_overlays)  
 
    image = cv2.imread(image_path)
    # assumes that the image is png...
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('show', image)
    # cv2.waitKey(0)
    image = cv2.resize(image, (512,512))
    img = image.transpose(2,0,1)

    imgC, imgW, imgH = img.shape

    img = img.reshape(1,imgC,imgW,imgH) # BCWH
    
    
    with torch.no_grad():
        mask_pred = model(torch.from_numpy(img).type(torch.cuda.FloatTensor))
        
    # color mapping corresponding to classes
    # ---------------------------------------------------------------------
    # 0 = background
    # 1 = fair (RED)
    # 2 = poor (GREEN)
    # 3 = severe (YELLOW)
    # BGR
    # ---------------------------------------------------------------------
    import numpy as np
    mapping = {0:np.array([0,0,0], dtype=np.uint8), 1:np.array([0,0,128], dtype=np.uint8), 
               2:np.array([0,128,0], dtype=np.uint8), 3:np.array([0,128,128], dtype=np.uint8)}
    
    y_pred_tensor = mask_pred
    pred = torch.argmax(y_pred_tensor, dim=1)
    y_pred = pred.data.cpu().numpy()
    yy = y_pred.ravel()
    import numpy as np
    height, width, channels = image.shape
    mask = np.zeros((height, width, channels), dtype=np.uint8)
    
    color = mapping[0]   
    
    for k in mapping:
        # Get all indices for current class
        idx = (pred==torch.tensor(k, dtype=torch.uint8))
        idx_np = (y_pred==k)[0]
        # color = mapping[k]
        mask[idx_np] = (mapping[k])
        
    # cv2.imshow('show', mask)
    # cv2.waitKey(0)
    image = img[0,...].transpose(1,2,0)
    # cv2.imshow('show', image)
    # cv2.waitKey(0)
    # overlay the mask on the image using the alpha combination blending
    overlay = cv2.addWeighted(image, 1, mask, 0.75, 0)

    
    # overlays
    cv2.imwrite(destination_mask+'/'+name, mask)
    cv2.imwrite(destination_overlays+'/'+name, overlay)
