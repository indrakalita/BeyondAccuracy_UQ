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

def show_images(images, max_cols=5, figsize=(18, 6)):
    n = len(images)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols  # automatic number of rows
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img) # RGB
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def analyze_flood_dataset(VV, VH, Label, n_workers=8, dark_pixel_thr=5000, smooth_thr=0.6, edge_thr=1000):
    N, H, W = VV.shape
    total_pixels = H * W
    
    #10th percentile pixel value across all VV and VH images. Any pixel below these thresholds is likely to be water (or looks like it)
    global_vv_thr = np.percentile(VV, 10)
    global_vh_thr = np.percentile(VH, 10)
    
    flood_idx_list, non_flood_idx_list, hard_neg_idx_list = [], [], []
    bin_indices = {'0_10': [], '10_30': [], '30_50': [], '50_up': []}
    
    def analyze_one(i):
        vv, vh, lbl = VV[i], VH[i], Label[i]
        flood_pixels = np.count_nonzero(lbl)
        flood_ratio = flood_pixels / total_pixels

        # Classify flood ratio bin
        if flood_pixels == 0: flood_bin = "background"
        elif flood_ratio <= 0.10: flood_bin = "0_10"
        elif flood_ratio <= 0.30: flood_bin = "10_30"
        elif flood_ratio <= 0.50: flood_bin = "30_50"
        else: flood_bin = "50_up"

        # Hard negative detection (only for pure non-flood)
        is_hard = False
        if flood_pixels == 0: # if no flood then check the dark pixels
            dark_mask = (vv < global_vv_thr) | (vh < global_vh_thr)
            dark_pixels = dark_mask.sum() # count the number of dark pixels in the image
    
            edges = sobel(vv.astype(float)) #Sobel detects image edges.
            edge_count = np.sum(edges > edges.mean()) #edge_count is small → image is smooth like water
            smooth_score = 1 - (edge_count / (H * W)) #If smooth_score > 0.6 → 60% or more area is smooth
    
            #5000 / (256x256) ≈ 0.076 (7.6%): dark region > 7%--> image has a significant area that looks like flood
            
            if dark_pixels > 5000 and (smooth_score > 0.6 or edge_count < 1000): 
                is_hard = True
        has_flood = flood_pixels > 0
        return i, has_flood, flood_bin, is_hard #i, has_flood, bin, hard_img?

    # Parallel execution
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for idx, has_flood, fbin, is_hard in ex.map(analyze_one, range(N)):
            if has_flood:
                flood_idx_list.append(idx)
                bin_indices[fbin].append(idx)
            else:
                non_flood_idx_list.append(idx)
            if is_hard:
                hard_neg_idx_list.append(idx)
    
    print("Total flood images:", len(flood_idx_list))
    print("Total non-flood images:", len(non_flood_idx_list))
    print("Flood bins:")
    for b, l in bin_indices.items():
        print(f"{b}: {len(l)}")
    print("Hard negatives:", len(hard_neg_idx_list))
    
    return flood_idx_list, non_flood_idx_list, hard_neg_idx_list, bin_indices

#---------Loss class------------#
class MultiLoss:
    def __init__(self, n_classes, weight=None, ce_weight_in_combo=0.5, focal_gamma=2.0, device='cuda'):
        """
        n_classes: number of segmentation classes
        weight: tensor of class weights for CE or Focal loss
        ce_weight_in_combo: weight of CE in Combo Loss (CE + Dice)
        focal_gamma: gamma parameter for Focal Loss
        ignore_index=0--> We have ignored the background to get the loss
        """
        self.n_classes = n_classes
        self.device = device
        self.ce_weight_in_combo = ce_weight_in_combo
        self.focal_gamma = focal_gamma
        
        if weight is not None:
            self.weight = weight.to(device)
        else:
            self.weight = None
        
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=self.weight)

    # --- Dice Loss ---
    def dice_loss(self, pred, target, smooth=1e-6, ignore_background=True):
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.n_classes).permute(0,3,1,2).float()
        
        if ignore_background == True:
            # Focus only on the flood class (index 1) for a more sensitive signal
            intersection = (pred_soft[:, 1:] * target_onehot[:, 1:]).sum(dim=(2,3))
            union = pred_soft[:, 1:].sum(dim=(2,3)) + target_onehot[:, 1:].sum(dim=(2,3))
        else:
            intersection = (pred_soft * target_onehot).sum(dim=(2,3))
            union = pred_soft.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    # --- Focal Loss ---
    def focal_loss(self, pred, target):
        logpt = -F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(logpt)
        loss = ((1 - pt)**self.focal_gamma) * (-logpt)
        return loss.mean()

    def focal_tversky(self, pred, target, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-7):
        #Higher α → more penalty for predicting flood where it’s actually non-flood. (to get high precision)
        #Higher β → more penalty for missing actual flood pixels. (to get high recall)
        
        prob = F.softmax(pred, dim=1)
        prob = torch.clamp(prob, min=eps, max=1-eps)
        
        target_onehot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
        
        # Ignore background (class 0) - focus on flood class only
        prob = prob[:, 1:, ...]      # (B, 1, H, W)
        target_onehot = target_onehot[:, 1:, ...]  # (B, 1, H, W)
        
        prob_flat = prob.view(prob.size(0), prob.size(1), -1)
        target_flat = target_onehot.view(target_onehot.size(0), target_onehot.size(1), -1)
        
        TP = (prob_flat * target_flat).sum(dim=2)
        FP = (prob_flat * (1 - target_flat)).sum(dim=2)
        FN = ((1 - prob_flat) * target_flat).sum(dim=2)
        
        tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
        focal_base = torch.clamp(1 - tversky, min=eps, max=1-eps)
        return (focal_base ** gamma).mean()  # Now scalar per batch


    # --- Combo Loss (CE + Dice) ---
    def combo_loss(self, pred, target, ignore_background=True):
        ce = self.ce_loss_fn(pred, target)
        dice = self.dice_loss(pred, target, ignore_background=ignore_background)
        return self.ce_weight_in_combo * ce + (1 - self.ce_weight_in_combo) * dice

    def ce_focal_tversky_loss(self, pred, target, lam=0.5):
        ce = self.ce_loss_fn(pred, target)
        ft = self.focal_tversky(pred, target)
        return lam * ce + (1 - lam) * ft

    # --- Forward function ---
    def __call__(self, pred, target, loss_name='ce', ignore_background=True): #pred: (B,C,H,W) logits, target: (B,H,W) class indices
        """
        loss_name: 'ce', 'dice', 'focal', 'ftv', 'cedice', 'ceftv'
        """
        if loss_name == 'ce':
            return self.ce_loss_fn(pred, target)
        elif loss_name == 'dice':
            return self.dice_loss(pred, target, ignore_background=ignore_background)
        elif loss_name == 'cedice':
            return self.combo_loss(pred, target, ignore_background=ignore_background)
        elif loss_name == 'focal':
            return self.focal_loss(pred, target)
        elif loss_name == 'ftv':          # <-- minimal new line
            return self.focal_tversky(pred, target)
        elif loss_name =='ceftv':
            return self.ce_focal_tversky_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss name: {loss_name}")

#-------------------#Compute PDE-based regularization loss on predictions#-------------------#
def pde_regularizer(pred_probs, input_image = 'None', method='tv', anisotropic_k=0.1, edge_sensitivity=10.0):
    if method == 'tv':
        dx = pred_probs[:, :, 1:, :] - pred_probs[:, :, :-1, :]
        dy = pred_probs[:, :, :, 1:] - pred_probs[:, :, :, :-1]
        loss = (dx.abs().mean() + dy.abs().mean())
        return loss

    elif method == 'laplacian':
        laplace_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=pred_probs.dtype, device=pred_probs.device)
        laplace_kernel = laplace_kernel.view(1,1,3,3)
        # apply to each channel separately
        loss = 0
        for c in range(pred_probs.shape[1]):
            pred_c = pred_probs[:,c:c+1,:,:]
            conv = F.conv2d(pred_c, laplace_kernel, padding=1)
            loss += (conv**2).mean()
        return loss / pred_probs.shape[1]

    elif method == 'anisotropic':
        # simple approximation: weight smoothing less at edges
        dx = pred_probs[:, :, 1:, :] - pred_probs[:, :, :-1, :]
        dy = pred_probs[:, :, :, 1:] - pred_probs[:, :, :, :-1]
        c_x = torch.exp(-(dx**2)/anisotropic_k)
        c_y = torch.exp(-(dy**2)/anisotropic_k)
        loss = (c_x*dx**2).mean() + (c_y*dy**2).mean()
        return loss
    elif method == 'edge_tv':
        """
        pred_probs: The Softmax output of your model (B, C, H, W)
        input_image: The structural input channel (e.g., your Median3 VV channel) (B, 1, H, W)
        """
        #print("NaN in input_image:", torch.isnan(input_image).any())
        #print("NaN in pred_probs:", torch.isnan(pred_probs).any())
        #print("Inf in input_image:", torch.isinf(input_image).any())
        #print("Inf in pred_probs:", torch.isinf(pred_probs).any())

        # 1. Calculate gradients of the PREDICTION (The flood map)
        # We focus on channel 1 (Flood)
        pred_flood = pred_probs
        
        dp_dx = pred_flood[:, :, :, 1:] - pred_flood[:, :, :, :-1]
        dp_dy = pred_flood[:, :, 1:, :] - pred_flood[:, :, :-1, :]
    
        # 2. Calculate gradients of the INPUT IMAGE (The SAR data)
        # We ensure input is the same size as the gradients by slicing
        img_dx = input_image[:, :, :, 1:] - input_image[:, :, :, :-1]
        img_dy = input_image[:, :, 1:, :] - input_image[:, :, :-1, :]
    
        # 3. Calculate Edge Weights
        # If input gradient is HIGH (edge), weight is LOW (don't smooth).
        # If input gradient is LOW (flat water/land), weight is HIGH (smooth aggressively).
        weight_y = torch.exp(-edge_sensitivity * img_dy)
        weight_x = torch.exp(-edge_sensitivity * img_dx)
        #weight_x = torch.exp(-edge_sensitivity * torch.clamp(torch.abs(img_dx), max=50))
        #weight_y = torch.exp(-edge_sensitivity * torch.clamp(torch.abs(img_dy), max=50))
        #weight_x = torch.clamp(weight_x, 1e-6, 1e6)
        #weight_y = torch.clamp(weight_y, 1e-6, 1e6)
    
        # 4. Apply Weighted TV
        # Using L1 here is often safer for edge-aware logic than L2
        loss_x = (weight_x * torch.abs(dp_dx)).mean()
        loss_y = (weight_y * torch.abs(dp_dy)).mean()
    
        return loss_x + loss_y

    else:
        raise ValueError(f"Unknown method: {method}")

#---------------Evaluation-----------------------
def test_evaluation(test_loader, model, channels=4, Pretrain = 'no', device=None):
    all_labels, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            sar, labels = batch['sar'].to(device), batch['label'].to(device)
            if channels ==2: sar = sar[:,0:2]
            if Pretrain == 'yes': sar = sar[:, [0, 1, 2], :, :]
            outputs = model(sar)
            preds = torch.argmax(outputs, dim=1)  # (B,H,W)
            for i in range(labels.size(0)):
                all_labels.append(labels[i].cpu().numpy()), all_preds.append(preds[i].cpu().numpy())
    return np.stack(all_labels), np.stack(all_preds)

def evaluate_segmentation(preds, labels, n_classes=9):
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Compute confusion matrix on GPU
    k = (labels >= 0) & (labels < n_classes)
    cm = torch.bincount(
        n_classes * labels[k].to(torch.int64) + preds[k].to(torch.int64),
        minlength=n_classes ** 2
    ).reshape(n_classes, n_classes)

    # True positives, false positives, false negatives
    TP = cm.diag()
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # Precision, Recall, F1
    precision = TP / (TP + FP + 1e-7)
    recall    = TP / (TP + FN + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)

    # IoU
    iou = TP / (TP + FP + FN + 1e-7)

    overall_acc = TP.sum() / cm.sum()

    return {"precision_per_class": precision.cpu().numpy(), "recall_per_class": recall.cpu().numpy(), "f1_per_class": f1.cpu().numpy(),
            "iou_per_class": iou.cpu().numpy(), "mean_precision": precision.mean().item(), "mean_recall": recall.mean().item(), 
            "mean_f1": f1.mean().item(), "mean_iou": iou.mean().item(), "overall_accuracy": overall_acc.item(),}
#----------The visual function---------------#
def visual(images_dict, bar_images=None):
    if bar_images is None:
        bar_images = []

    n = len(images_dict)
    plt.figure(figsize=(5*n, 5))

    for i, (name, img) in enumerate(images_dict.items(), 1):
        plt.subplot(1, n, i)

        # Convert torch tensors if passed
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()

        # Convert (C,H,W) → (H,W,C) if needed
        if img.ndim == 3 and img.shape[0] in [1,3]:
            img = np.transpose(img, (1, 2, 0))

        # Choose colormap
        if name in bar_images:
            plt.imshow(img, cmap='Reds')
            plt.colorbar()
        else:
            if img.ndim == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)

        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
