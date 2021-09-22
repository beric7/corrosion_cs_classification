# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:23:10 2020

@author: Eric Bianchi
"""

#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys
import shutil

import imgviz
import numpy as np

import labelme
import cv2
from PIL import Image

# labelme2voc.py input_dir "D://DATA/Datasets/Original_Dataset/Train/Train_Images/" ^
# output_dir "D://DATA/Datasets/Original_Dataset/Train/Train_Images/segmentation_corrosion/Masked_Patched_Corrosion_png_v2/" ^
# --labels "labels_segmentation.txt"

def createMasks(input_dir, output_dir, label_txt_file, imviz=True):

    ###########################################################################
    # removing and creating directories
    if osp.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        print("Output directory removed:", output_dir)
    os.makedirs(output_dir)
    os.makedirs(osp.join(output_dir, "SegmentationClass"))
    os.makedirs(osp.join(output_dir, "SegmentationClassPNG"))
    ###########################################################################
    
    ###########################################################################
    # reading the text file to determine classes and background
    print("Creating dataset:", output_dir)
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(label_txt_file).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)
    ###########################################################################

    ###########################################################################
    # Creating masks from jsons
    for filename in glob.glob(osp.join(input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        if imviz:
            out_img_file = osp.join(output_dir, "JPEGImages", base + ".jpg")
        out_lbl_file = osp.join(
            output_dir, "SegmentationClass", base + ".npy"
        )
        # Changed from .png to .jpg
        out_png_file = osp.join(
            output_dir, "SegmentationClassPNG", base + ".png"
        )
        # imagee = cv2.imread('D://DATA/Datasets/ZED_background_blur/Black Camera DSCF140.JPG')
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        
        # im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # newImage = imagee - im_rgb        
        # cv2.imwrite('D://DATA/Datasets/ZED_background_blur/save.jpg', newImage)
        
        if imviz:
            imgviz.io.imsave(out_img_file, img)

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        labelme.utils.lblsave(out_png_file, lbl)

        np.save(out_lbl_file, lbl)
    ###########################################################################
