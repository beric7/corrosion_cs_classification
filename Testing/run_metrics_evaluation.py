# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:52:06 2021

@author: Admin
"""
import torch
from metric_evaluation import plot_confusion_matrix, iterate_data, spectrum_score_norm, spectrum_score


data_dir = './DSCN0411/'
batchsize = 1

model = torch.load(f'./stored_weights/var_aug_wbatch_2_resnet50/var_aug_wbatch_2_resnet50_weights_18.pt', map_location=torch.device('cuda'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()   # Set model to evaluate mode
##############################################################################

iOU, f1, confm_sum, y_pred = iterate_data(model, data_dir)

plot_confusion_matrix(confm_sum, target_names=['Background', 'Fair', 'Poor', 'Severe'], normalize=True, 
                      title='Confusion Matrix')

import matplotlib.pyplot as plt
import numpy as np
heatmap_im = y_pred.reshape(512,512)
plt.imshow(heatmap_im, cmap='jet', interpolation='nearest')
plt.clim(0,3)
plt.colorbar()
plt.show()


'''
ss = spectrum_score(confm_sum)
ss_norm = spectrum_score_norm(confm_sum)
'''