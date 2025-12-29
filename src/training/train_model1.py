import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime

from src.models.model1 import create_model
from src.data.dataset import create_dataloaders


class RankingLoss(nn.Module):
    """
    Spearman's rank correlation loss for keyframe importance scoring.
    Better than MSE for ranking tasks.
    """
    def __init__(self):
        super(RankingLoss, self).__init__()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, num_frames) predicted importance scores
            targets: (batch_size, num_frames) ground truth scores
        """
        batch_size = predictions.size(0)
        
        total_loss = 0
        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]
            
            # Convert to ranks
            pred_rank = pred.argsort().argsort().float()
            target_rank = target.argsort().argsort().float()
            
            # Spearman correlation
            pred_mean = pred_rank.mean()
            target_mean = target_rank.mean()
            
            numerator = ((pred_rank - pred_mean) * (target_rank - target_mean)).sum()
            denominator = torch.sqrt(
                ((pred_rank - pred_mean) ** 2).sum() * 
                ((target_rank - target_mean) ** 2).sum()
            )
            
            correlation = numerator / (denominator + 1e-8)
            
            # Convert correlation to loss (maximize correlation = minimize negative)
            total_loss += (1 - correlation)
        
        return total_loss / batch_size


class Trainer:
    """
    Training manager with 2-stage fine-tuning strategy.
    
    Stage 1 (5-10 epochs): Freeze ResNet, train Transformer + Attention
    Stage 2 (10-20 epochs): Unfreeze ResNet last block, fine-tune end-to-end
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        save_dir='checkpoints',
        log_dir='logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Loss functions
        self.ranking_loss = RankingLoss()
        self.mse_loss = nn.MSELoss()
        
        # Metrics
        self.best_val_loss = float('inf')
        
    def train_epoch(self, optimizer, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_ranking_loss = 0
        total_mse_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, (frames, scores, _) in enumerate(pbar):
            frames = frames.to(self.device)  # (B, T, 3, H, W)
            scores = scores.to(self.device)  # (B, T)
            
            # Forward pass
            predictions = self.model(frames)  # (B, T)
            
            # Calculate losses
            ranking_loss = self.ranking_loss(predictions, scores)
            mse_loss = self.mse_loss(predictions, scores)
            
            # Combined loss (ranking is primary, MSE is secondary)
            loss = ranking_loss + 0.1 * mse_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_ranking_loss += ranking_loss.item()
            total_mse_loss += mse_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rank': f'{ranking_loss.item():.4f}',
                'mse': f'{mse_loss.item():.4f}'
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_ranking_loss = total_ranking_loss / len(self.train_loader)
        avg_mse_loss = total_mse_loss / len(self.train_loader)
        
        return avg_loss, avg_ranking_loss, avg_mse_loss
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        total_ranking_loss = 0
        total_mse_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for frames, scores, _ in pbar:
                frames = frames.to(self.device)
                scores = scores.to(self.device)
                
                # Forward pass
                predictions = self.model(frames)
                
                # Calculate losses
                ranking_loss = self.ranking_loss(predictions, scores)
                mse_loss = self.mse_loss(predictions, scores)
                loss = ranking_loss + 0.1 * mse_loss
                
                # Update metrics
                total_loss += loss.item()
                total_ranking_loss += ranking_loss.item()
                total_mse_loss += mse_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'rank': f'{ranking_loss.item():.4f}',
                    'mse': f'{mse_loss.item():.4f}'
                })
        
        # Calculate average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_ranking_loss = total_ranking_loss / len(self.val_loader)
        avg_mse_loss = total_mse_loss / len(self.val_loader)
        
        return avg_loss, avg_ranking_loss, avg_mse_loss
    
    def save_checkpoint(self, epoch, optimizer, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, num_epochs_stage1=10, num_epochs_stage2=20, lr_stage1=1e-4, lr_stage2=1e-5):
        """
        Two-stage training strategy.
        
        Stage 1: Freeze ResNet, train Transformer + Attention
        Stage 2: Unfreeze ResNet last block, fine-tune end-to-end
        """
        print("\n" + "="*60)
        print("🔥 STAGE 1: Training Transformer + Attention (ResNet Frozen)")
        print("="*60 + "\n")
        
        # Stage 1 optimizer (only trainable parameters)
        optimizer_stage1 = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr_stage1,
            weight_decay=1e-4
        )
        
        scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage1,
            T_max=num_epochs_stage1
        )
        
        # Stage 1 training
        for epoch in range(1, num_epochs_stage1 + 1):
            train_loss, train_rank, train_mse = self.train_epoch(optimizer_stage1, epoch)
            val_loss, val_rank, val_mse = self.validate(epoch)
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('RankingLoss/train', train_rank, epoch)
            self.writer.add_scalar('RankingLoss/val', val_rank, epoch)
            self.writer.add_scalar('MSELoss/train', train_mse, epoch)
            self.writer.add_scalar('MSELoss/val', val_mse, epoch)
            
            print(f"\nEpoch {epoch}/{num_epochs_stage1}")
            print(f"Train - Loss: {train_loss:.4f}, Rank: {train_rank:.4f}, MSE: {train_mse:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Rank: {val_rank:.4f}, MSE: {val_mse:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, optimizer_stage1, is_best)
            
            scheduler_stage1.step()
        
        print("\n" + "="*60)
        print("🔥 STAGE 2: Fine-tuning End-to-End (ResNet Unfrozen)")
        print("="*60 + "\n")
        
        # Unfreeze last ResNet block
        self.model.unfreeze_resnet(unfreeze_last_n_blocks=1)
        
        # Stage 2 optimizer (all parameters)
        optimizer_stage2 = optim.AdamW(
            self.model.parameters(),
            lr=lr_stage2,
            weight_decay=1e-4
        )
        
        scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage2,
            T_max=num_epochs_stage2
        )
        
        # Stage 2 training
        for epoch in range(num_epochs_stage1 + 1, num_epochs_stage1 + num_epochs_stage2 + 1):
            train_loss, train_rank, train_mse = self.train_epoch(optimizer_stage2, epoch)
            val_loss, val_rank, val_mse = self.validate(epoch)
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('RankingLoss/train', train_rank, epoch)
            self.writer.add_scalar('RankingLoss/val', val_rank, epoch)
            self.writer.add_scalar('MSELoss/train', train_mse, epoch)
            self.writer.add_scalar('MSELoss/val', val_mse, epoch)
            
            print(f"\nEpoch {epoch}/{num_epochs_stage1 + num_epochs_stage2}")
            print(f"Train - Loss: {train_loss:.4f}, Rank: {train_rank:.4f}, MSE: {train_mse:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Rank: {val_rank:.4f}, MSE: {val_mse:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, optimizer_stage2, is_best)
            
            scheduler_stage2.step()
        
        print("\n" + "="*60)
        print("✓ Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train keyframe detection model')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to video directory')
    parser.add_argument('--h5_path', type=str, required=True, help='Path to TVSum h5 file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--epochs_stage1', type=int, default=10, help='Epochs for stage 1')
    parser.add_argument('--epochs_stage2', type=int, default=20, help='Epochs for stage 2')
    parser.add_argument('--lr_stage1', type=float, default=1e-4, help='Learning rate stage 1')
    parser.add_argument('--lr_stage2', type=float, default=1e-5, help='Learning rate stage 2')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.video_dir,
        args.h5_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = create_model(freeze_resnet=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (Stage 1): {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    # Train
    trainer.train(
        num_epochs_stage1=args.epochs_stage1,
        num_epochs_stage2=args.epochs_stage2,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2
    )


if __name__ == "__main__":
    main()
