import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
import numpy as np
from numpy import ndarray
from skimage import measure
import cv2
from ImageSet import CUBDataSet
import os.path as osp
import os
import random



def DDT_fit(model,data_loader,batch_size):
    model.eval()

    descriptors = np.zeros((1, 2048))
    descriptors_tensor = torch.zeros((1,2048)).cuda()
    count_images = 0
    for images,_ in data_loader:
        count_images +=1  
        images = images.cuda()

        for name, module in model.named_children():
            if name == "avgpool":
                break
            images = module(images)
    
        output = images.view(-1 , 2048, images.shape[2] * images.shape[3])
        output = output.permute(1,0,2)
        output = output.reshape(2048,-1)
        output = output.transpose(0, 1)
        #descriptors = np.vstack((descriptors, output.detach().cpu().numpy().copy()))
        descriptors_tensor = torch.vstack((descriptors_tensor,output.detach()))
        del output
        count = batch_size * count_images
        print("success deal with {} images".format(count))
        if count % 64 == 0:
            descriptors = np.vstack((descriptors, descriptors_tensor.detach().cpu().numpy().copy()))
            descriptors_tensor = torch.zeros((1,2048)).cuda()

    if count % 64 != 0:
        descriptors = np.vstack((descriptors, descriptors_tensor.detach().cpu().numpy().copy()))

    descriptors = descriptors[1:]
    descriptors_mean = sum(descriptors)/len(descriptors)
    descriptors_mean_tensor = torch.FloatTensor(descriptors_mean)
    pca = PCA(n_components=1)
    pca.fit(descriptors)
    trans_vec = pca.components_[0]

    return trans_vec, descriptors_mean_tensor

