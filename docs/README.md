# Keyframe Detection using ResNet50 + Transformer + Dual Temporal Attention

Deep learning model for automatic keyframe detection in videos using TVSum dataset.

## 🏗️ Architecture

**Technique 2: CNN + Transformer with Dual Temporal Attention**

```
Input (30s video, 2 FPS → 60 frames)
    ↓
ResNet50 (pretrained, feature extraction) → 2048-dim features
    ↓
Feature Projection → 512-dim
    ↓
Positional Encoding (temporal order)
    ↓
Transformer Encoder (3 layers, 8 heads)
    ↓
Dual Temporal Attention
    ├─ Local Attention (short-range dependencies)
    └─ Global Attention (long-range dependencies)
    ↓
Fusion Layer
    ↓
Importance Scorer → Frame importance scores [0, 1]
```

## 🎯 Key Features

- **Pretrained ResNet50**: Transfer learning from ImageNet
- **Dual Temporal Attention**: Captures both local and global temporal patterns
- **Two-Stage Training**: 
  - Stage 1: Freeze ResNet, train Transformer (10 epochs)
  - Stage 2: Unfreeze ResNet, fine-tune end-to-end (20 epochs)
- **Ranking Loss**: Spearman's correlation for importance ranking
- **Importance Curve Visualization**: Plot predicted vs ground truth scores

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 📂 Dataset Preparation

Download TVSum dataset:
```bash
# Download TVSum videos and annotations
# Expected structure:
# tvsum/
#   ├── videos/
#   │   ├── video1.mp4
#   │   ├── video2.mp4
#   │   └── ...
#   └── tvsum.h5
```

## 🚀 Training

```bash
python train.py \
    --video_dir /path/to/tvsum/videos \
    --h5_path /path/to/tvsum/tvsum.h5 \
    --batch_size 4 \
    --epochs_stage1 10 \
    --epochs_stage2 20 \
    --lr_stage1 1e-4 \
    --lr_stage2 1e-5 \
    --save_dir checkpoints \
    --log_dir logs
```

### Training Strategy

**Stage 1 (Epochs 1-10):**
- Freeze ResNet50 backbone
- Train Transformer + Dual Attention
- Learning rate: 1e-4
- Optimizer: AdamW with weight decay

**Stage 2 (Epochs 11-30):**
- Unfreeze last ResNet block (layer4)
- Fine-tune end-to-end
- Learning rate: 1e-5 (lower to avoid catastrophic forgetting)
- Gradient clipping: max norm 1.0

## 📊 Visualization

Generate importance curves and keyframe visualizations:

```bash
python visualize.py \
    --video_dir /path/to/tvsum/videos \
    --h5_path /path/to/tvsum/tvsum.h5 \
    --checkpoint checkpoints/best_model.pth \
    --save_dir visualizations \
    --num_videos 5
```

Output:
- `visualizations/{video_name}/importance_curve.jpg` - Importance score plot
- `visualizations/{video_name}/keyframes_grid.jpg` - Detected keyframes grid
- `visualizations/{video_name}/keyframe_*.jpg` - Individual keyframe images

## 📈 Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

Metrics tracked:
- Training/Validation Loss
- Ranking Loss (Spearman)
- MSE Loss
- Learning Rate

## 🧪 Testing & Evaluation

Test the model on the test set:

```python
from model import create_model
from dataset import TVSumDataset
from visualize import evaluate_full_dataset
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(freeze_resnet=False)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Load test dataset
test_dataset = TVSumDataset(video_dir, h5_path, split='test')

# Evaluate
metrics = evaluate_full_dataset(model, test_dataset, device)
```

**Evaluation Metrics:**
- Spearman's Rank Correlation
- Kendall's Tau
- Mean Squared Error (MSE)
- Precision@15% (top-15% keyframe overlap)

## 🎓 Model Details

**Parameters:**
- Total: ~26M parameters
- Trainable (Stage 1): ~4M parameters
- Trainable (Stage 2): ~14M parameters

**Hyperparameters:**
- Feature dimension: 512
- Transformer layers: 3
- Attention heads: 8
- Feedforward dimension: 2048
- Dropout: 0.1
- Batch size: 4
- Frames per video: 60 (2 FPS sampling)
- Input size: 224×224 RGB

## 💡 Key Design Choices

1. **ResNet50 over ResNet101**: Balance between performance and speed (works on Google Colab)
2. **Dual Attention**: Captures both local scene transitions and global video context
3. **2 FPS sampling**: 60 frames for 30s video, sufficient temporal resolution
4. **Ranking loss**: Better than MSE for importance scoring tasks
5. **Two-stage training**: Prevents overfitting and training instability

## 🔥 Advantages Over Model 1

- **Temporal modeling**: Transformer captures long-range dependencies
- **Dual attention**: Both local and global patterns
- **Pretrained features**: Transfer learning from ImageNet
- **Importance curve**: Smooth, interpretable output
- **Scalable**: Works on variable-length videos

## 📝 Usage Example

```python
from model import create_model
import torch

# Create model
model = create_model(freeze_resnet=False)
model.eval()

# Input: 1 video, 60 frames, 224×224 RGB
frames = torch.randn(1, 60, 3, 224, 224)

# Predict importance scores
with torch.no_grad():
    importance_scores = model(frames)  # Shape: (1, 60)

# Select top-15% as keyframes
k = int(0.15 * 60)
keyframe_indices = importance_scores.argsort(descending=True)[:k]

print(f"Detected keyframes: {keyframe_indices.tolist()}")
```

## 🐛 Troubleshooting

**Out of Memory (OOM):**
- Reduce batch size: `--batch_size 2`
- Use mixed precision training (add in train.py)
- Sample fewer frames: modify `num_frames` in dataset.py

**Training not converging:**
- Check learning rates
- Ensure data normalization is correct
- Verify ResNet is properly frozen in Stage 1

**Video loading errors:**
- Check video codec (use H.264)
- Install ffmpeg: `conda install ffmpeg`
- Try different video extensions in dataset.py

## 📚 References

- TVSum Dataset: [Yale Song et al., CVPR 2015]
- Transformer: [Vaswani et al., NeurIPS 2017]
- ResNet: [He et al., CVPR 2016]
- Video Summarization: Survey papers on keyframe detection

## 🙋 Questions?

For project-specific questions, refer to your project documentation or contact your supervisor.

## ✅ Checklist for Submission

- [ ] Train model for 30 epochs (10 + 20)
- [ ] Save best model checkpoint
- [ ] Generate importance curves for test videos
- [ ] Calculate evaluation metrics
- [ ] Compare with Model 1 (technique comparison)
- [ ] Prepare presentation with visualizations
- [ ] Document model architecture and training strategy

---

**Author**: Your Name  
**Date**: December 2025  
**Course**: Deep Learning Project
