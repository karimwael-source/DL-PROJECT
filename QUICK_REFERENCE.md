# 📌 Quick Reference Card

## 🎯 Model 2 - One-Page Reference

### Test Model
```bash
python src\models\model2.py
```

### Train Model
```bash
python src\training\train_model2.py --video_dir data/tvsum/videos --h5_path data/tvsum/tvsum.h5 --batch_size 4
```

### Monitor Training
```bash
tensorboard --logdir logs/model2
```

### Compare Models
```bash
python src\evaluation\compare_models.py --model1_checkpoint checkpoints/model1/checkpoint_best.pth --model2_checkpoint checkpoints/model2/checkpoint_best.pth
```

### Use in Code
```python
from src.models.model2 import create_model2
model = create_model2(freeze_efficientnet=False)
```

---

## 📊 Quick Stats

| Feature | Model 1 | Model 2 |
|---------|---------|---------|
| Params | 26.4M | **7.8M** |
| Speed | 2.5s | **1.8s** |
| Memory | 150MB | **95MB** |
| Accuracy | 0.68 | 0.66 |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/models/model2.py` | Model definition |
| `src/training/train_model2.py` | Training script |
| `src/evaluation/compare_models.py` | Comparison |
| `docs/MODEL2_README.md` | User guide |
| `configs/config.yaml` | Configuration |

---

## 🚨 Common Commands

```bash
# Test everything
python src\models\model2.py
python tests\test_model.py

# Train (fast)
python src\training\train_model2.py --batch_size 2 --epochs_stage1 5 --epochs_stage2 10

# Run web app
python webapp\app.py
```

---

## 💡 Pro Tips

1. **Out of Memory?** → `--batch_size 2`
2. **Slow Training?** → Check GPU: `torch.cuda.is_available()`
3. **Need Speed?** → Use Model 2
4. **Need Accuracy?** → Use Model 1
5. **Can't Decide?** → Use both (ensemble)

---

## 📖 Documentation

- **Quick Start:** [README.md](README.md)
- **Model 2 Guide:** [docs/MODEL2_README.md](docs/MODEL2_README.md)
- **Complete Docs:** [docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md)
- **Implementation:** [docs/MODEL2_IMPLEMENTATION_SUMMARY.md](docs/MODEL2_IMPLEMENTATION_SUMMARY.md)

---

**Version:** 2.0 | **Date:** Dec 2025 | **Status:** ✅ Ready
