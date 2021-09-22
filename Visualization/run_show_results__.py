# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:48:39 2020

@author: Eric Bianchi
"""

import os 
from show_results__ import*
from tqdm import tqdm   
import torch

# Load the trained model, you could possibly change the device from cpu to gpu if 
# you have your gpu configured.
model = torch.load(f'./stored_weights/l1_loss/weights_27.pt', map_location=torch.device('cuda'))

# Set the model to evaluate mode
model.eval()

source_image_dir = './corrosion_progression/'
destination_mask = './predicted_masks_l1_loss_corrosion_progression/'
destination_overlays = './combined_overlays_l1_loss_corrosion_progression/'

from PIL import Image
import os
import glob
import numpy as np

for image_name in tqdm(os.listdir(source_image_dir)):
    print(image_name)
    image_path = source_image_dir + image_name
    generate_images(model, image_path, image_name, destination_mask, destination_overlays)
    
    

'''
def crop(im, height, width):
    # im = Image.open(infile)
    imgwidth, imgheight = im.size
    rows = np.int(imgheight/height)
    cols = np.int(imgwidth/width)
    im_list = []
    for i in range(rows):
        for j in range(cols):
            # print (i,j)
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            im_list.append(im.crop(box))
    return im_list


for image_name in tqdm(os.listdir(source_image_dir)):
    print(image_name)
    image_path = source_image_dir + image_name
    image_name, ext =  image_name.split('.')  
    im = Image.open(image_path)
    imgwidth, imgheight = im.size

    height = np.int(imgheight/3)
    width = np.int(imgwidth/3)
    start_num = 0
    imList = crop(im, height, width)
    i = 0 
    for image in imList:
        image.save(source_image_dir + image_name + '_mini_' +str(i)+'.'+ext)
        i = i + 1
'''