import torch
import numpy as np
import torch.nn.functional as F

def adjust_learning_rate(optimizer, initial_lr, iter, total_iter, power=0.9):
    lr = initial_lr * (1 - iter / total_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def compute_iou(outputs, targets):
    smooth = 1e-6
    preds = torch.argmax(outputs, dim=1)
    
    ious = []
    for cls in range(1, 21): 
        pred_inds = preds == cls
        target_inds = targets == cls
        
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        
        if union.item() > 0:
            iou = (intersection + smooth) / (union + smooth)
            ious.append(iou.item())
    
    return np.mean(ious) if ious else 0


def calculate_unsupervised_loss(outputs, pseudo_labels, conf_mask):
    if conf_mask.sum() > 0:
        outputs = outputs.permute(0, 2, 3, 1)
        masked_outputs = outputs[conf_mask].view(-1, outputs.size(3))
        masked_labels = pseudo_labels[conf_mask].view(-1)
        unsup_loss = F.cross_entropy(
            masked_outputs, 
            masked_labels, 
            ignore_index=-1
        )
    else:
        unsup_loss = torch.tensor(0.0).to(device)
    
    return unsup_loss