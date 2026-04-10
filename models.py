import cv2, os, glob, random, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from skimage.filters import median
from skimage.morphology import disk
from skimage.filters import sobel, laplace
from scipy.ndimage import uniform_filter
from concurrent.futures import ThreadPoolExecutor


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import utils as ut
import dataprocess as dp

import segmentation_models_pytorch as smp


#---------------------------------------Pre-train Model--------------------------------------------------------------------------#
def build_model(model_name, in_channels, n_classes):
    if model_name == "unet_resnet34": #working
        return smp.Unet(encoder_name="resnet34",encoder_weights="imagenet",in_channels=in_channels,classes=n_classes)

    elif model_name == "unet_densenet121": 
        return smp.Unet(encoder_name="densenet121",encoder_weights="imagenet",in_channels=in_channels,classes=n_classes)

    elif model_name == "unet_effb7": #working
        return smp.Unet(encoder_name="efficientnet-b7",encoder_weights="imagenet",in_channels=in_channels,classes=n_classes)

    elif model_name == "unet_mobilenetv2": #working
        return smp.Unet(encoder_name="mobilenet_v2",encoder_weights="imagenet",in_channels=in_channels,classes=n_classes)
    

    elif model_name == "unetpp_effb7": #working
        return smp.UnetPlusPlus(encoder_name="efficientnet-b7",encoder_weights="imagenet",in_channels=in_channels,classes=n_classes)
        

def replace_activation_with_dropout(module, p=0.2):
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.SiLU)): # Logic: Catch ReLU, LeakyReLU, and SiLU (used in EfficientNet)
            setattr(module, name, nn.Sequential(child, nn.Dropout2d(p)))
        else:
            replace_activation_with_dropout(child, p)
