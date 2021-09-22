# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:38:49 2020

@author: Eric Bianchi
"""
import sys

from image_utils import rescale

# rescale(source_image_folder, destination, dimension):
dimension = 512
source = './Train_renamed/images/'
destination = './Train_renamed/images_512/'
rescale(source, destination, dimension)
