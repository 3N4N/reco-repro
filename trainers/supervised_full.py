#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
from torch import optim, nn
from tqdm import tqdm
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet101
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.cityscapes_data_loader import CityscapesDataset, CityscapesLoader
from data.pascal_data_loader import PascalVOCDataset, PascalVOCLoader
from train_utils import adjust_learning_rate, calculate_unsupervised_loss, compute_iou



def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training Script')
    parser.add_argument('--dataset', type=str, required=True, choices=['pascal', 'cityscapes'],
                        help='Dataset to train on (pascal or cityscapes)')
    parser.add_argument('--data-path', type=str, required=True, 
                        help='Path to dataset')
    parser.add_argument('--model', type=str, default='fcn_resnet50', 
                        choices=['fcn_resnet50', 'deeplabv3'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override default batch size')
    parser.add_argument('--lr', type=float, default=2.5e-3,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--total-iterations', type=int, default=40000,
                        help='Total training iterations')
    parser.add_argument('--val-interval', type=int, default=2000,
                        help='Validation interval (iterations)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-labeled', type=int, default=None,
                        help='Number of labeled samples (for semi-supervised learning)')
    parser.add_argument('--gpu', type=int, default=1)
    return parser.parse_args()

def create_model(args):
    num_classes = 21 if args.dataset == 'pascal' else 19
    
    if args.model == 'fcn_resnet50':
        model = fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif args.model == 'deeplabv3':
        model = deeplabv3_resnet101(pretrained=True)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    return model

def main():
    args = parse_args()
    print(f"Training on {args.dataset} dataset with {args.model} model")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.dataset == 'pascal':
        loader = PascalVOCLoader(
            data_path=args.data_path,
            num_labeled=args.num_labeled,
            seed=args.seed
        )
        if args.batch_size is not None:
            loader.batch_size = args.batch_size
        num_classes = 21
    else:  # cityscapes
        loader = CityscapesLoader(
            data_path=args.data_path,
            num_labeled=args.num_labeled,
            seed=args.seed
        )
        if args.batch_size is not None:
            loader.batch_size = args.batch_size
        num_classes = 19
    
    train_loader, val_loader = loader.create_loaders(use_unlabeled=False)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    model = create_model(args)
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  
    
    val_interval = args.val_interval
    best_iou = 0
    total_iterations = 0
    
    pbar = tqdm(total=args.total_iterations)
    
    for epoch in range(args.epochs):
        # print(f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (img, mask) in enumerate(train_loader):
            model.train()
            
            img = img.to(device)
            mask = mask.long().to(device)
            
            current_lr = adjust_learning_rate(optimizer, args.lr, total_iterations, args.total_iterations)
            
            outputs = model(img)['out']
            optimizer.zero_grad()
            
            loss = criterion(outputs, mask)
            
            loss.backward()
            optimizer.step()
            
            total_iterations += 1
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch+1}/{args.epochs}, Iter: {total_iterations}/{args.total_iterations}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
            
            # Validation
            if total_iterations % val_interval == 0:
                model.eval()
                val_running_loss = 0
                val_iou = 0
                num_batches = 0
                
                with torch.no_grad():
                    for idx, (img, mask) in enumerate(val_loader):
                        img = img.to(device)
                        mask = mask.long().to(device)
                        
                        outputs = model(img)['out']
                        loss = criterion(outputs, mask)
                        
                        batch_iou = compute_iou(outputs, mask, num_classes)
                        val_iou += batch_iou
                        val_running_loss += loss.item()
                        num_batches += 1
                    
                    val_loss = val_running_loss / num_batches
                    mean_iou = val_iou / num_batches
                
                print("-"*50)
                print(f"Epoch: {epoch+1}/{args.epochs}, Iteration: {total_iterations}/{args.total_iterations}")
                print(f"Validation Loss: {val_loss:.4f}, Mean IoU: {mean_iou:.4f}")
                print("-"*50)
                
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.model}_best.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Saved best model to {checkpoint_path}")
            
            # Save checkpoint periodically
            if total_iterations % 10000 == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.model}_iter_{total_iterations}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'iteration': total_iterations,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'iou': best_iou,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            if total_iterations >= args.total_iterations:
                break
        
        if total_iterations >= args.total_iterations:
            print(f"Reached {args.total_iterations} iterations. Stopping training.")
            break
    
    pbar.close()
    print(f"Training completed! Best validation IoU: {best_iou:.4f}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.model}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

if __name__ == "__main__":
    main()
