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

