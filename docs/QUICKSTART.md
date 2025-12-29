# 🚀 Quick Start Guide

## ✅ Step 1: Test Model (No Dataset Needed)

Test if the model architecture works:

```bash
python test_model.py
```

This will:
- Create the model
- Test forward pass with dummy data
- Verify keyframe selection
- Test ResNet unfreezing

**Expected output**: "ALL TESTS PASSED - MODEL WORKS PERFECTLY!"

---

## 📥 Step 2: Get TVSum Dataset

You need to download TVSum dataset manually:

### Option A: Official Source
```bash
# Clone the TVSum repo
git clone https://github.com/yalesong/tvsum

# Or download directly from:
# https://github.com/yalesong/tvsum/tree/master/data
```

### Option B: Alternative Sources
- Search for "TVSum dataset download" 
- Look for `ydata-tvsum50-v1_1.zip`
- Extract to get `ydata-tvsum50.mat` or `tvsum.h5`

### Expected Structure:
```
E:/
└── DL_project_finalized/
    ├── model.py
    ├── dataset.py
    ├── train.py
    ├── ...
    └── data/              # Create this folder
        ├── tvsum/
        │   ├── videos/    # Put video files here
        │   │   ├── video_1.mp4
        │   │   ├── video_2.mp4
        │   │   └── ...
        │   └── tvsum.h5   # Annotation file
```

---

## 🧪 Step 3: Test Dataset Loading

Once you have the dataset:

```bash
python test_dataset.py
```

This will verify:
- Videos can be loaded
- Frames are sampled correctly (2 FPS = 60 frames)
- Annotations are loaded properly

---

## 🏋️ Step 4: Start Training

```bash
python train.py \
    --video_dir E:/DL_project_finalized/data/tvsum/videos \
    --h5_path E:/DL_project_finalized/data/tvsum/tvsum.h5 \
    --batch_size 4 \
    --epochs_stage1 10 \
    --epochs_stage2 20
```

**For Google Colab** (if dataset is on Google Drive):
```python
from google.colab import drive
drive.mount('/content/drive')

!python train.py \
    --video_dir /content/drive/MyDrive/tvsum/videos \
    --h5_path /content/drive/MyDrive/tvsum/tvsum.h5 \
    --batch_size 2
```

---

## 📊 Step 5: Monitor Training

Open TensorBoard:
```bash
tensorboard --logdir logs
```

Then open: http://localhost:6006

---

## 🎨 Step 6: Visualize Results

After training completes:

```bash
python visualize.py \
    --video_dir E:/DL_project_finalized/data/tvsum/videos \
    --h5_path E:/DL_project_finalized/data/tvsum/tvsum.h5 \
    --checkpoint checkpoints/best_model.pth \
    --num_videos 5
```

Check `visualizations/` folder for:
- Importance curve plots
- Keyframe grids
- Individual keyframes

---

## ⚡ Quick Check (Before Full Training)

If you want to test everything quickly:

1. **Install packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test model** (no dataset needed):
   ```bash
   python test_model.py
   ```

3. **Get dataset** (manual download)

4. **Test dataset loading**:
   ```bash
   python test_dataset.py
   ```

5. **Start training!**

---

## ❓ FAQ

**Q: Do I need an API key?**  
A: No! The data loads from local files, no API needed.

**Q: Where to download TVSum?**  
A: Search "TVSum dataset" or check GitHub: https://github.com/yalesong/tvsum

**Q: Can I use a different dataset?**  
A: Yes, but you need to modify `dataset.py` to match your format.

**Q: How long does training take?**  
A: ~2-3 hours on GPU (Colab), 10+ hours on CPU (not recommended).

**Q: Out of memory error?**  
A: Reduce `--batch_size 2` or `--batch_size 1`

**Q: Can I test with 1 video only?**  
A: Yes! Modify the dataset split in `dataset.py` or create a custom test script.

---

## 🎯 Summary

1. ✅ Test model → `python test_model.py`
2. 📥 Download TVSum dataset manually
3. 🧪 Test dataset → `python test_dataset.py`
4. 🏋️ Train → `python train.py --video_dir ... --h5_path ...`
5. 📊 Monitor → `tensorboard --logdir logs`
6. 🎨 Visualize → `python visualize.py --checkpoint ...`

**No API needed - everything runs locally!**
