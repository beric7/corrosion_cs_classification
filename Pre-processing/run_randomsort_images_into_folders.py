# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:38:49 2020

@author: Eric Bianchi
"""
import sys
from image_utils import random_sort_images

# blackAndWhite(source_image_folder, destination):
source_mask = './mask_512/'
source_image = './image_512/'
destination_mask_test = './Test/Mask/'
destination_image_test = './Test/Image/'
destination_mask_train = './Train/Mask/'
destination_image_train = './Train/Image/'
percentage = 0.1

# random_sort_images(source_image_folder, destination, seed=10, percentage=0.1)
random_sort_images(source_mask, 
                   source_image, 
                   destination_mask_test, 
                   destination_image_test, 
                   destination_mask_train, 
                   destination_image_train,
                   percentage=percentage)
