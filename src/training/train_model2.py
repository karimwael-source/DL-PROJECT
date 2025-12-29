"""
Training Script for Model 2: EfficientNet-B0 + Transformer
===========================================================

Two-stage training strategy for efficient keyframe detection.

Stage 1 (Epochs 1-10):
- Freeze EfficientNet-B0 backbone
- Train Transformer + Dual Attention layers
- Learning rate: 1e-4
- Goal: Learn temporal modeling with fixed visual features

Stage 2 (Epochs 11-30):
- Unfreeze last 2 EfficientNet blocks
- Fine-tune end-to-end
- Learning rate: 1e-5 (10× lower)
- Gradient clipping: max norm 1.0
- Goal: Adapt visual features to keyframe detection

Usage:
------
```bash
python src/training/train_model2.py \\
    --video_dir data/tvsum/videos \\
    --h5_path data/tvsum/tvsum.h5 \\
    --batch_size 4 \\
    --epochs_stage1 10 \\
    --epochs_stage2 20 \\
    --lr_stage1 1e-4 \\
    --lr_stage2 1e-5 \\
    --save_dir checkpoints/model2 \\
    --log_dir logs/model2
```

Author: Deep Learning Project
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.model2 import create_model2, count_parameters
from src.data.dataset import create_dataloaders


class RankingLoss(nn.Module):
    """
    Spearman's rank correlation loss for keyframe importance scoring.
    
    Better than MSE for ranking tasks because it:
    - Focuses on correct ordering rather than absolute values
    - More robust to label noise
    - Aligns better with human perception of importance
    
    Formula:
    --------
    1. Convert predictions and targets to ranks
    2. Compute Spearman correlation: ρ = cov(rank_pred, rank_target) / (σ_pred * σ_target)
    3. Loss = 1 - ρ (maximize correlation = minimize loss)
    """
    
    def __init__(self):
        super(RankingLoss, self).__init__()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, num_frames) predicted importance scores
            targets: (batch_size, num_frames) ground truth scores
        
        Returns:
            loss: Scalar ranking loss (1 - Spearman correlation)
        """
        batch_size = predictions.size(0)
        
        total_loss = 0
        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]
            
            # Convert to ranks (argsort twice gives ranks)
            pred_rank = pred.argsort().argsort().float()
            target_rank = target.argsort().argsort().float()
            
            # Compute means
            pred_mean = pred_rank.mean()
            target_mean = target_rank.mean()
            
            # Compute Spearman correlation
            numerator = ((pred_rank - pred_mean) * (target_rank - target_mean)).sum()
            denominator = torch.sqrt(
                ((pred_rank - pred_mean) ** 2).sum() * 
                ((target_rank - target_mean) ** 2).sum()
            )
            
            correlation = numerator / (denominator + 1e-8)
            
            # Loss = 1 - correlation (we want to maximize correlation)
            loss = 1 - correlation
            total_loss += loss
        
        return total_loss / batch_size


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, global_step):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (frames, scores) in enumerate(pbar):
        frames = frames.to(device)
        scores = scores.to(device)
        
        # Forward pass
        predictions = model(frames)
        loss = criterion(predictions, scores)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Log to TensorBoard
        if writer is not None and global_step[0] % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), global_step[0])
            writer.add_scalar('Train/AvgLoss', avg_loss, global_step[0])
        
        global_step[0] += 1
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch, writer):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for frames, scores in pbar:
            frames = frames.to(device)
            scores = scores.to(device)
            
            # Forward pass
            predictions = model(frames)
            loss = criterion(predictions, scores)
            
            total_loss += loss.item()
            avg_loss = total_loss / len(all_predictions) if all_predictions else loss.item()
            
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Store for metrics calculation
            all_predictions.append(predictions.cpu())
            all_targets.append(scores.cpu())
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute additional metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # MSE
    mse = ((all_predictions - all_targets) ** 2).mean().item()
    
    # Spearman correlation (average across batch)
    from scipy.stats import spearmanr
    correlations = []
    for i in range(len(all_predictions)):
        corr, _ = spearmanr(all_predictions[i].numpy(), all_targets[i].numpy())
        if not np.isnan(corr):
            correlations.append(corr)
    
    avg_spearman = np.mean(correlations) if correlations else 0.0
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/MSE', mse, epoch)
        writer.add_scalar('Val/Spearman', avg_spearman, epoch)
    
    return avg_loss, mse, avg_spearman


def save_checkpoint(model, optimizer, epoch, loss, save_path, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model to {best_path}")


def train_model2(args):
    """Main training function for Model 2."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  TRAINING MODEL 2: EfficientNet-B0 + Transformer")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        video_dir=args.video_dir,
        h5_path=args.h5_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating Model 2...")
    model = create_model2(freeze_efficientnet=True)
    model = model.to(device)
    
    # Count parameters
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable (Stage 1): {params['trainable']:,}")
    print(f"Frozen: {params['frozen']:,}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard logs: {log_dir}")
    
    # Loss and optimizer
    criterion = RankingLoss()
    
    # Global step counter
    global_step = [0]
    
    # Best validation loss
    best_val_loss = float('inf')
    
    # ========================================================================
    # STAGE 1: Train with frozen EfficientNet
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  STAGE 1: Training Transformer + Attention (EfficientNet frozen)")
    print(f"{'='*70}")
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_stage1,
        weight_decay=args.weight_decay
    )
    
    for epoch in range(1, args.epochs_stage1 + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs_stage1} (Stage 1) ---")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                 device, epoch, writer, global_step)
        
        # Validate
        val_loss, val_mse, val_spearman = validate(model, val_loader, criterion, 
                                                    device, epoch, writer)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val MSE: {val_mse:.4f} | Val Spearman: {val_spearman:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, 
                       is_best=(val_loss < best_val_loss))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # ========================================================================
    # STAGE 2: Fine-tune with unfrozen EfficientNet
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  STAGE 2: Fine-tuning end-to-end (EfficientNet unfrozen)")
    print(f"{'='*70}")
    
    # Unfreeze EfficientNet
    model.unfreeze_efficientnet(unfreeze_last_n_blocks=2)
    params_stage2 = count_parameters(model)
    print(f"Trainable (Stage 2): {params_stage2['trainable']:,}")
    
    # New optimizer with lower learning rate
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_stage2,
        weight_decay=args.weight_decay
    )
    
    for epoch in range(args.epochs_stage1 + 1, args.epochs_stage1 + args.epochs_stage2 + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs_stage1 + args.epochs_stage2} (Stage 2) ---")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                 device, epoch, writer, global_step)
        
        # Validate
        val_loss, val_mse, val_spearman = validate(model, val_loader, criterion, 
                                                    device, epoch, writer)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val MSE: {val_mse:.4f} | Val Spearman: {val_spearman:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, 
                       is_best=(val_loss < best_val_loss))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # Final evaluation on test set
    print(f"\n{'='*70}")
    print(f"  FINAL EVALUATION ON TEST SET")
    print(f"{'='*70}")
    
    test_loss, test_mse, test_spearman = validate(model, test_loader, criterion, 
                                                   device, "Test", writer)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test Spearman: {test_spearman:.4f}")
    
    writer.close()
    
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {args.log_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 2 for Keyframe Detection")
    
    # Data
    parser.add_argument('--video_dir', type=str, default='data/tvsum/videos',
                       help='Directory containing video files')
    parser.add_argument('--h5_path', type=str, default='data/tvsum/tvsum.h5',
                       help='Path to TVSum annotations file')
    
    # Training Stage 1
    parser.add_argument('--epochs_stage1', type=int, default=10,
                       help='Number of epochs for stage 1 (frozen EfficientNet)')
    parser.add_argument('--lr_stage1', type=float, default=1e-4,
                       help='Learning rate for stage 1')
    
    # Training Stage 2
    parser.add_argument('--epochs_stage2', type=int, default=20,
                       help='Number of epochs for stage 2 (unfrozen EfficientNet)')
    parser.add_argument('--lr_stage2', type=float, default=1e-5,
                       help='Learning rate for stage 2 (lower for fine-tuning)')
    
    # Optimization
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='checkpoints/model2',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/model2',
                       help='Directory for TensorBoard logs')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Train the model
    train_model2(args)
