import torch
import numpy as np
import torch.nn.functional as F

def adjust_learning_rate(optimizer, initial_lr, iter, total_iter, power=0.9):
    lr = initial_lr * (1 - iter / total_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def compute_iou(outputs, targets, num_classes):
    smooth = 1e-6
    preds = torch.argmax(outputs, dim=1)
    
    ious = []
    for cls in range(1, num_classes):
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
        unsup_loss = torch.tensor(0.0).to(outputs.device)
    
    return unsup_loss

def reco_loss_func(rep, label, mask, prob=None, temp=0.5, num_queries=256, num_negatives=256):
    B, C, H, W = rep.shape
    rep = rep.reshape(B, C, -1)  
    label = label.reshape(B, -1)  
    mask = mask.reshape(B, -1)  
    
    # Get unique classes in the batch
    classes = torch.unique(label)
    classes = classes[classes != 255]  
    
    if len(classes) == 0:
        return torch.tensor(0.0, device=rep.device, requires_grad=True)
    
    # Compute class representations (means)
    r_c_plus_k = {} 
    for c in classes:
        c_mask = (label == c) & mask 
        if c_mask.sum() == 0:
            continue
        
        class_pixels = []
        for b in range(B):
            b_mask = c_mask[b] 
            if b_mask.sum() > 0:
                b_rep = rep[b, :, b_mask] 
                class_pixels.append(b_rep)
        
        if len(class_pixels) > 0:
            all_pixels = torch.cat(class_pixels, dim=1)  
            r_c_plus_k[c.item()] = all_pixels.mean(dim=1) 
    
    # TODO: Implement contrastive loss computation 
    
    return torch.tensor(0.0, device=rep.device, requires_grad=True)
