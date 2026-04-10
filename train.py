#!/usr/bin/env python
# coding: utf-8

import argparse
import cv2, os, glob, random, time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter
from scipy.ndimage import sobel
from joblib import Parallel, delayed

from skimage.morphology import disk
from skimage.filters.rank import median
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from ptflops import get_model_complexity_info

import utils as ut
import models as md
import dataprocess as dp
import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, required=True)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--augmentation', type=str)
parser.add_argument('--attention', type=str)
parser.add_argument('--pretrain', type=str)
parser.add_argument('--dropout', type=str)

args = parser.parse_args()
channels = args.in_channels
loss_type = args.loss_type
Augmentation = args.augmentation
Attention = args.attention
Pretrain = args.pretrain
Dropout = args.dropout
print(f'Ch: {channels}, Loss: {loss_type}, Aug: {Augmentation}, Drop: {Dropout}, Pretrain: {Pretrain}')
if Attention == 'yes':
    attention_layers=[0,1] # On decoder only
else:
    attention_layers=None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 1024**3, "GB")

# Conditional parameters
batch_size = 128
n_classes = 2           # 0=background, 1-8 classes
if Pretrain == 'yes':
    save_model_path = f'./new_save_model/UPPEffi_RLRP{Pretrain}M_{channels}C_{loss_type}L_{Augmentation}_Drop{Dropout}'
else:
    save_model_path = f'./save_model/MRLRP_{channels}C_{loss_type}L_{Augmentation}_Attn{Attention}'

path1 = './ETCI_dataset/train/train/'
path2 = './ETCI_dataset/New1/data/test/'
train_samples_all = dp.load_dataset(path1)
test_samples = dp.load_dataset(path2)
train_samples_all, val_samples = dp.split_by_event(train_samples_all, split_ratio=1.0 - 0.12) #val_split = 0.01
stats = dp.compute_train_stats(train_samples_all) # Compute train stats (do it before augmentation but after train val split)

VV_train = np.stack([s["vv"] for s in train_samples_all], axis=0)       # shape: (N, H, W)
VH_train = np.stack([s["vh"] for s in train_samples_all], axis=0)       # shape: (N, H, W)
Label_train = np.stack([s["label"] for s in train_samples_all], axis=0) # shape: (N, H, W)
events = [s["event"] for s in train_samples_all]
#Check imbalance status
flood_idx_list, non_flood_idx_list, hard_neg_idx_list, bin_indices = ut.analyze_flood_dataset(VV_train, VH_train, Label_train,
                                                                                              n_workers=8, dark_pixel_thr=5000,
                                                                                              smooth_thr=0.6, edge_thr=1000)
print(VV_train.shape, VH_train.shape, Label_train.shape)

print(f'Augmentation is applied?-----------------------{Augmentation}-------------------------------')
current_counts = {b: len(bin_indices[b]) for b in bin_indices}
augmented_indices, bin_probs = dp.prepare_augmentation_indices(bin_indices, len(non_flood_idx_list), current_counts)
print("Bin probs for augmentation:", bin_probs)
print("Total augmented flood images to generate:", len(augmented_indices))
train_sample_aug = dp.augment_dataset(VV_train, VH_train, Label_train, events, augmented_indices, augmentation = Augmentation)
VV_train_aug = np.stack([s["vv"] for s in train_sample_aug], axis=0)       # shape: (N, H, W)
print("Final dataset shape:", VV_train_aug.shape)

train_dataset = dp.FloodSARSegDataset(train_sample_aug, stats)
val_dataset = dp.FloodSARSegDataset(val_samples, stats)
test_dataset  = dp.FloodSARSegDataset(test_samples, stats)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

for v in ["model","optimizer","x","labels"]: 
    globals().pop(v, None)
torch.cuda.empty_cache()
# ------------------------------Parameters---------------------------------------------#
warmup_epochs=10 # 10 for vit
num_epochs = 4000  # adjust as needed
selectSC = 'RLRP' #RLRP
lr = 1e-3
if Pretrain == 'yes':
    model_name = "unetpp_effb7"   # change here or pass via CLI Null unet_resnet34, unet_effb7, unet_mobilenetv2, unetpp_effb7
    model = md.build_model(model_name=model_name, in_channels=3, n_classes=n_classes).to(device)
    if Dropout== 'yes':
        print('Dropout is gonaa be applied.........')
        md.replace_activation_with_dropout(model.decoder, p=0.2)

else:
    print('No pretrained model applied')
    model = md.GeneralizedResidualUNet(in_channels=channels, out_channels=n_classes, filters=[32, 64, 128, 256], use_attention='custom',
                                   attention_layers=attention_layers).to(device) #attention_layers=[0,1]
    
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
weights = None#torch.tensor([1.0, 5.0]).to(device)
loss_fn = ut.MultiLoss(n_classes=n_classes, weight=weights, ce_weight_in_combo=0.5, focal_gamma=2.0, device='cuda') #focal_gamma=0--> CE weight = torch.tensor([0.2, 1.0])
if selectSC == 'COS':scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
if selectSC == 'RLRP':scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, min_lr=1e-6, verbose=True)


#----------Train--------------#
scaler = GradScaler()
best_val_acc, counter = 0, 0
for epoch in range(num_epochs):
    model.train()
    start = time.time()
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        sar, labels = batch['sar'].to(device), batch['label'].to(device)
        if channels == 2:
            sar = sar[:,0:2]
        if Pretrain == 'yes':
            sar = sar[:, [0, 1, 2], :, :]
        optimizer.zero_grad()
        with autocast():
            outputs = model(sar)
            if epoch==0 and batch_idx==0:
                print(f'Training started with input size {sar.shape}, output size {labels.shape}, {labels.dtype}, pred is {outputs.shape}, {outputs.dtype}')
            loss = loss_fn(outputs, labels, loss_name=loss_type)  # choose 'ce', 'dice', 'combo', 'focal'
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {time.time()-start}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    #------------------------------------------Check validation----------------------------------------------------#
    if (epoch) % 10 == 0:
        val_label, val_eval = ut.test_evaluation(val_loader, model, channels=channels, Pretrain = Pretrain, device=device)
        y_true_t = torch.tensor(val_label, device=device)
        y_pred_t = torch.tensor(val_eval, device=device)
        metrics = ut.evaluate_segmentation(y_pred_t, y_true_t, n_classes=2)
        val_mean_iou = metrics['mean_iou']
        if selectSC == 'RLRP': scheduler.step(val_mean_iou)
        print(f"Mean F1: {metrics['mean_f1']:.3f} and Mean IoU: {val_mean_iou:.3f}")
        if val_mean_iou>best_val_acc:
            best_val_acc = val_mean_iou
            counter = 0
            torch.save(model.state_dict(), save_model_path+'_Best.pth')
            print(f"--> Best model updated and saved at epoch {epoch}!")
        else:
            counter = counter + 1
        if counter >10:
            print(".----------->Early stopping triggered! Training stopped<------------.")
            break
    # --- Warmup LR for first few epochs ---
    if selectSC == 'COS':
        if epoch < warmup_epochs:
            scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * scale # this will increase the LR till the first warmup_epochs number of epochs
        else:
            scheduler.step() # this will reduce the lr till the end
    if (epoch) % 100 == 0:
        val_label, val_eval = ut.test_evaluation(test_loader, model, channels=channels, Pretrain = Pretrain, device=device)
        y_true_t = torch.tensor(val_label, device=device)
        y_pred_t = torch.tensor(val_eval, device=device)
        metrics = ut.evaluate_segmentation(y_pred_t, y_true_t, n_classes=2)
        val_mean_iou_test = metrics['mean_iou']
        print(f"For test set, Mean F1: {metrics['mean_f1']:.3f} and Mean IoU: {val_mean_iou_test:.3f}")
        
torch.save(model.state_dict(), save_model_path+'_Final.pth')
