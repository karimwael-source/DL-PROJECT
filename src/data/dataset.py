import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import h5py
import os
from torchvision import transforms


class TVSumDataset(Dataset):
    """
    TVSum dataset loader for keyframe detection.
    
    Dataset structure:
    - Videos: 50 videos, each ~30 seconds
    - Annotations: Importance scores per frame (from multiple annotators)
    - Sampling: 2 FPS (60 frames per video)
    """
    def __init__(
        self,
        video_dir,
        h5_path,
        split='train',
        num_frames=60,
        img_size=224,
        transform=None
    ):
        """
        Args:
            video_dir: Path to directory containing video files
            h5_path: Path to TVSum h5 file with annotations
            split: 'train', 'val', or 'test'
            num_frames: Number of frames to sample (default: 60 for 2 FPS)
            img_size: Size to resize frames to (default: 224 for ResNet)
            transform: Optional torchvision transforms
        """
        self.video_dir = video_dir
        self.h5_path = h5_path
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        
        # Default transform for ResNet
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Load dataset
        self.video_list = []
        self.load_data()
        
    def load_data(self):
        """Load video information and annotations from h5 file."""
        with h5py.File(self.h5_path, 'r') as f:
            for video_name in f.keys():
                video_data = f[video_name]
                
                # Get annotations (importance scores)
                # Shape: (num_annotators, num_frames)
                user_summary = video_data['user_summary'][...]
                
                # Average across annotators
                gt_scores = np.mean(user_summary, axis=0)
                
                # Normalize to [0, 1]
                gt_scores = (gt_scores - gt_scores.min()) / (gt_scores.max() - gt_scores.min() + 1e-8)
                
                self.video_list.append({
                    'video_name': video_name,
                    'gt_scores': gt_scores,
                    'n_frames': len(gt_scores)
                })
        
        # Split dataset (80% train, 10% val, 10% test)
        np.random.seed(42)
        indices = np.random.permutation(len(self.video_list))
        
        n_train = int(0.8 * len(self.video_list))
        n_val = int(0.1 * len(self.video_list))
        
        if self.split == 'train':
            indices = indices[:n_train]
        elif self.split == 'val':
            indices = indices[n_train:n_train + n_val]
        else:  # test
            indices = indices[n_train + n_val:]
        
        self.video_list = [self.video_list[i] for i in indices]
        
        print(f"✓ Loaded {len(self.video_list)} videos for {self.split} split")
    
    def __len__(self):
        return len(self.video_list)
    
    def load_video_frames(self, video_path):
        """
        Load and sample frames from video at 2 FPS.
        
        Args:
            video_path: Path to video file
        Returns:
            frames: (num_frames, 3, H, W) tensor
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample at 2 FPS
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame can't be read, duplicate last frame
                if len(frames) > 0:
                    frame = frames[-1]
                else:
                    # Create black frame
                    frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transform
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
        
        cap.release()
        
        # Stack frames: (num_frames, 3, H, W)
        frames = torch.stack(frames)
        
        return frames
    
    def resample_scores(self, gt_scores):
        """
        Resample ground truth scores to match num_frames.
        
        Args:
            gt_scores: Original importance scores (variable length)
        Returns:
            resampled_scores: Scores resampled to num_frames
        """
        if len(gt_scores) == self.num_frames:
            return gt_scores
        
        # Linear interpolation
        original_indices = np.linspace(0, len(gt_scores) - 1, len(gt_scores))
        target_indices = np.linspace(0, len(gt_scores) - 1, self.num_frames)
        
        resampled_scores = np.interp(target_indices, original_indices, gt_scores)
        
        return resampled_scores
    
    def __getitem__(self, idx):
        """
        Get video frames and importance scores.
        
        Returns:
            frames: (num_frames, 3, H, W) tensor
            scores: (num_frames,) tensor of importance scores
            video_name: str
        """
        video_info = self.video_list[idx]
        video_name = video_info['video_name']
        
        # Construct video path
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        
        # Alternative extensions if .mp4 doesn't exist
        if not os.path.exists(video_path):
            for ext in ['.avi', '.webm', '.mkv', '.mov']:
                alt_path = os.path.join(self.video_dir, f"{video_name}{ext}")
                if os.path.exists(alt_path):
                    video_path = alt_path
                    break
        
        # Load frames
        frames = self.load_video_frames(video_path)
        
        # Get and resample ground truth scores
        gt_scores = video_info['gt_scores']
        gt_scores = self.resample_scores(gt_scores)
        gt_scores = torch.tensor(gt_scores, dtype=torch.float32)
        
        return frames, gt_scores, video_name


def create_dataloaders(video_dir, h5_path, batch_size=4, num_workers=2):
    """
    Create train, val, and test dataloaders.
    
    Args:
        video_dir: Path to video directory
        h5_path: Path to TVSum h5 file
        batch_size: Batch size
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = TVSumDataset(video_dir, h5_path, split='train')
    val_dataset = TVSumDataset(video_dir, h5_path, split='val')
    test_dataset = TVSumDataset(video_dir, h5_path, split='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    video_dir = "path/to/tvsum/videos"
    h5_path = "path/to/tvsum/tvsum.h5"
    
    # Create dataset
    dataset = TVSumDataset(video_dir, h5_path, split='train')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    if len(dataset) > 0:
        frames, scores, video_name = dataset[0]
        
        print(f"\nVideo: {video_name}")
        print(f"Frames shape: {frames.shape}")
        print(f"Scores shape: {scores.shape}")
        print(f"Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
