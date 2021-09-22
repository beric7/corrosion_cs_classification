# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:24:31 2021

@author: Admin
"""
from image_utils import rename_segmentation

source = './Test/Images/'
source_image_folder = './images/'
source_mask_folder = './VOC_440_Corrosion_Dataset_review/SegmentationClassPNG/'
source_json_folder = './JSON/'
source_ohev_folder = './VOC_440_Corrosion_Dataset_review/SegmentationClass/'

test_or_train = './Test_renamed/'

image_destination = '/images/'
mask_destination = '/masks/'
json_destination = '/json/'
ohev_destination = './ohev/'

rename_segmentation(source, source_image_folder, source_mask_folder, source_json_folder, source_ohev_folder,
                        test_or_train, image_destination, mask_destination, json_destination, ohev_destination)