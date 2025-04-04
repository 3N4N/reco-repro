import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm
import numpy as np

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.cityscapes_data_loader import CityscapesLoader
from data.pascal_data_loader import PascalVOCLoader
from network.mean_ts import TeacherModel
from train_utils import adjust_learning_rate, calculate_unsupervised_loss, compute_iou


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised semantic segmentation')
    
    parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'cityscapes'],
                        help='Dataset to use (pascal or cityscapes)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--num-labeled', type=int, default=5,
                        help='Number of labeled examples')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2.5e-3,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay for SGD optimizer')
    parser.add_argument('--power', type=float, default=0.9,
                        help='Power for polynomial decay of learning rate')
    parser.add_argument('--iterations', type=int, default=40000,
                        help='Total number of training iterations')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum number of epochs')
    
    # Model arguments
    parser.add_argument('--ema-decay', type=float, default=0.99,
                        help='EMA decay rate for teacher model')
    parser.add_argument('--conf-thresh', type=float, default=0.95,
                        help='Confidence threshold for pseudo-labels')
    parser.add_argument('--unsup-weight', type=float, default=0.5,
                        help='Weight for unsupervised loss')
    
    # Logging and checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--val-interval', type=int, default=2000,
                        help='Validation interval (iterations)')
    parser.add_argument('--save-interval', type=int, default=10000,
                        help='Checkpoint saving interval (iterations)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=1)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.dataset == 'pascal':
        loader = PascalVOCLoader(
            data_path=args.data_path,
            num_labeled=args.num_labeled,
            seed=args.seed
        )
        num_classes = 21  # 20 classes + background
        
    elif args.dataset == 'cityscapes':
        loader = CityscapesLoader(
            data_path=args.data_path,
            num_labeled=args.num_labeled,
            seed=args.seed
        )
        num_classes = 19
    
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    train_labeled_loader, train_unlabeled_loader, val_loader = loader.create_loaders(use_unlabeled=True)
    
    student_model = fcn_resnet50(pretrained=True)
    student_model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    student_model = student_model.to(device)
    
    teacher_model = TeacherModel(student_model, ema_decay=args.ema_decay).to(device)
    
    optimizer = optim.SGD(
        student_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_iou = 0
    total_iterations = 0
    
    pbar = tqdm(total=args.iterations)
    
    for epoch in range(args.max_epochs):
        # print(f"Epoch {epoch+1}/{args.max_epochs}")
        
        labeled_iter = iter(train_labeled_loader)
        unlabeled_iter = iter(train_unlabeled_loader)
        
        for labeled_batch in train_labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(train_unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            
            labeled_img = labeled_batch[0].float().to(device)
            labeled_mask = labeled_batch[1].long().to(device)
            unlabeled_img = unlabeled_batch[0].float().to(device)
            
            current_lr = adjust_learning_rate(
                optimizer, args.lr, total_iterations, args.iterations, args.power
            )
            
            student_model.train()
            student_labeled_output = student_model(labeled_img)['out']
            
            supervised_loss = criterion(student_labeled_output, labeled_mask)
            
            pseudo_labels, conf_mask, confidence = teacher_model.generate_pseudo_labels(
                unlabeled_img, confidence_threshold=args.conf_thresh
            )
            
            student_unlabeled_output = student_model(unlabeled_img)['out']
            
            unsupervised_loss = calculate_unsupervised_loss(
                student_unlabeled_output, pseudo_labels, conf_mask
            )
            
            conf_ratio = conf_mask.float().mean().item()
            
            eta = conf_ratio if conf_ratio > 0 else 0.1
            total_loss = supervised_loss + args.unsup_weight * eta * unsupervised_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            teacher_model.update_weights(student_model)
            
            total_iterations += 1
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch+1}/{args.max_epochs}, Iter: {total_iterations}/{args.iterations}, "
                f"Loss: {total_loss.item():.4f}, Sup: {supervised_loss.item():.4f}, "
                f"Unsup: {unsupervised_loss.item():.4f}, LR: {current_lr:.6f}, Conf: {conf_ratio:.2f}"
            )
            
            if total_iterations % args.val_interval == 0:
                student_model.eval()
                val_running_loss = 0
                val_iou = 0
                num_batches = 0
                
                with torch.no_grad():
                    for idx, img_mask in enumerate(val_loader):
                        img = img_mask[0].float().to(device)
                        mask = img_mask[1].long().to(device)
                        
                        y_pred = student_model(img)['out']
                        loss = criterion(y_pred, mask)
                        
                        batch_iou = compute_iou(y_pred, mask, num_classes)
                        val_iou += batch_iou
                        val_running_loss += loss.item()
                        num_batches += 1
                    
                    val_loss = val_running_loss / num_batches
                    mean_iou = val_iou / num_batches
                
                print("-"*50)
                print(f"Epoch: {epoch+1}/{args.max_epochs}, Iteration: {total_iterations}/{args.iterations}")
                print(f"Validation Loss: {val_loss:.4f}, Mean IoU: {mean_iou:.4f}")
                print("-"*50)
                
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    torch.save(
                        student_model.state_dict(), 
                        os.path.join(args.checkpoint_dir, "best_student_model.pth")
                    )
            
            if total_iterations % args.save_interval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'iteration': total_iterations,
                    'student_model_state_dict': student_model.state_dict(),
                    'teacher_model_state_dict': teacher_model.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                }, os.path.join(args.checkpoint_dir, f"checkpoint_iter_{total_iterations}.pth"))
            
            if total_iterations >= args.iterations:
                break
        
        if total_iterations >= args.iterations:
            print(f"Reached {args.iterations} iterations. Stopping training.")
            break
    
    pbar.close()
    print(f"Training completed! Best validation IoU: {best_iou:.4f}")
    torch.save(
        student_model.state_dict(), 
        os.path.join(args.checkpoint_dir, "final_student_model.pth")
    )


if __name__ == '__main__':
    main()
