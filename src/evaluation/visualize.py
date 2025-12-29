import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm

from src.models.model1 import create_model
from src.data.dataset import TVSumDataset


def plot_importance_curve(video_name, predicted_scores, gt_scores, save_path=None):
    """
    Plot importance curve: predicted vs ground truth scores.
    
    Args:
        video_name: Name of the video
        predicted_scores: Predicted importance scores (num_frames,)
        gt_scores: Ground truth scores (num_frames,)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    frames = np.arange(len(predicted_scores))
    
    # Plot curves
    plt.plot(frames, predicted_scores, 'b-', linewidth=2, label='Predicted', marker='o', markersize=3)
    plt.plot(frames, gt_scores, 'r--', linewidth=2, label='Ground Truth', marker='x', markersize=3)
    
    # Highlight keyframes (top-k by predicted scores)
    k = int(0.15 * len(predicted_scores))  # Top 15% as keyframes
    keyframe_indices = np.argsort(predicted_scores)[-k:]
    plt.scatter(keyframe_indices, predicted_scores[keyframe_indices], 
                color='gold', s=100, marker='*', zorder=5, label='Predicted Keyframes')
    
    # Styling
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.title(f'Importance Curve: {video_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_keyframes(video_path, keyframe_indices, save_dir=None):
    """
    Visualize detected keyframes from a video.
    
    Args:
        video_path: Path to video file
        keyframe_indices: Indices of keyframes
        save_dir: Directory to save keyframe images
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Extract keyframes
    keyframes = []
    for idx in keyframe_indices:
        # Map to actual frame index (if sampled at 2 FPS)
        actual_idx = int(idx * (total_frames / 60))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keyframes.append(frame_rgb)
            
            # Save individual keyframe
            if save_dir:
                save_path = os.path.join(save_dir, f'keyframe_{idx:03d}.jpg')
                cv2.imwrite(save_path, frame)
    
    cap.release()
    
    # Plot keyframes in a grid
    if keyframes:
        n_frames = len(keyframes)
        cols = min(5, n_frames)
        rows = (n_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (ax, frame) in enumerate(zip(axes.flat, keyframes)):
            ax.imshow(frame)
            ax.set_title(f'Frame {keyframe_indices[i]}', fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(keyframes), len(axes.flat)):
            axes.flat[i].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            grid_path = os.path.join(save_dir, 'keyframes_grid.jpg')
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved keyframe grid to {grid_path}")
        else:
            plt.show()
        
        plt.close()


def evaluate_model(model, dataset, device, save_dir='visualizations', num_videos=5):
    """
    Evaluate model and visualize results.
    
    Args:
        model: Trained model
        dataset: Test dataset
        device: Computation device
        save_dir: Directory to save visualizations
        num_videos: Number of videos to visualize
    """
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Randomly select videos
    video_indices = np.random.choice(len(dataset), min(num_videos, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in tqdm(video_indices, desc="Visualizing videos"):
            frames, gt_scores, video_name = dataset[idx]
            
            # Add batch dimension
            frames = frames.unsqueeze(0).to(device)  # (1, T, 3, H, W)
            
            # Predict
            predicted_scores = model(frames)  # (1, T)
            predicted_scores = predicted_scores.squeeze(0).cpu().numpy()
            gt_scores = gt_scores.numpy()
            
            # Create video-specific directory
            video_dir = os.path.join(save_dir, video_name)
            os.makedirs(video_dir, exist_ok=True)
            
            # Plot importance curve
            curve_path = os.path.join(video_dir, 'importance_curve.jpg')
            plot_importance_curve(video_name, predicted_scores, gt_scores, save_path=curve_path)
            
            # Detect keyframes (top 15%)
            k = int(0.15 * len(predicted_scores))
            keyframe_indices = np.argsort(predicted_scores)[-k:]
            keyframe_indices = np.sort(keyframe_indices)
            
            # Visualize keyframes
            video_path = os.path.join(dataset.video_dir, f"{video_name}.mp4")
            visualize_keyframes(video_path, keyframe_indices, save_dir=video_dir)
            
            print(f"\n✓ Processed {video_name}")
            print(f"  Detected {len(keyframe_indices)} keyframes: {keyframe_indices.tolist()}")


def calculate_metrics(predicted_scores, gt_scores):
    """
    Calculate evaluation metrics.
    
    Args:
        predicted_scores: (N,) predicted importance scores
        gt_scores: (N,) ground truth scores
    
    Returns:
        metrics: dict of metric values
    """
    from scipy.stats import spearmanr, kendalltau
    
    # Spearman's rank correlation
    spearman_corr, _ = spearmanr(predicted_scores, gt_scores)
    
    # Kendall's tau
    kendall_corr, _ = kendalltau(predicted_scores, gt_scores)
    
    # MSE
    mse = np.mean((predicted_scores - gt_scores) ** 2)
    
    # Precision@K (top-K overlap)
    k = int(0.15 * len(predicted_scores))
    pred_top_k = set(np.argsort(predicted_scores)[-k:])
    gt_top_k = set(np.argsort(gt_scores)[-k:])
    precision_at_k = len(pred_top_k & gt_top_k) / k
    
    return {
        'spearman': spearman_corr,
        'kendall': kendall_corr,
        'mse': mse,
        'precision@15': precision_at_k
    }


def evaluate_full_dataset(model, dataset, device):
    """
    Evaluate model on full dataset and print metrics.
    
    Args:
        model: Trained model
        dataset: Test dataset
        device: Computation device
    """
    model.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            frames, gt_scores, video_name = dataset[idx]
            
            # Add batch dimension
            frames = frames.unsqueeze(0).to(device)
            
            # Predict
            predicted_scores = model(frames).squeeze(0).cpu().numpy()
            gt_scores = gt_scores.numpy()
            
            # Calculate metrics
            metrics = calculate_metrics(predicted_scores, gt_scores)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    for key, value in avg_metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("="*50 + "\n")
    
    return avg_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize keyframe detection results')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to video directory')
    parser.add_argument('--h5_path', type=str, required=True, help='Path to TVSum h5 file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--num_videos', type=int, default=5, help='Number of videos to visualize')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = create_model(freeze_resnet=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = TVSumDataset(args.video_dir, args.h5_path, split='test')
    
    # Evaluate and visualize
    print("\nEvaluating model...")
    avg_metrics = evaluate_full_dataset(model, test_dataset, device)
    
    print("\nGenerating visualizations...")
    evaluate_model(model, test_dataset, device, save_dir=args.save_dir, num_videos=args.num_videos)
    
    print(f"\n✓ All visualizations saved to: {args.save_dir}")
