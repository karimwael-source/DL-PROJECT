# 🎬 Enhanced UI - Quick Start Guide

## 🚀 Start the Application

```bash
python webapp/app.py
```

**Open in browser:** http://localhost:5000

---

## ✨ New Features at a Glance

### 1. 🎯 Choose Your Model

**Model 1: ResNet50** 🚀
- 26.4M parameters
- High accuracy
- Best for: Critical applications

**Model 2: EfficientNet-B0** ⚡  
- 7.8M parameters (70% fewer!)
- 28% faster
- Best for: Quick processing

**How to use:** Click on the model card to select

---

### 2. 🎚️ Custom Keyframe Count

Select **1 to 60** keyframes using the slider

**Examples:**
- `3 frames` → Quick thumbnail
- `9 frames` → Balanced summary
- `15 frames` → Detailed coverage
- `30 frames` → Frame-by-frame analysis

**How to use:** Drag the slider to your desired number

---

### 3. 📤 Upload Video

**Three ways:**
1. **Drag & Drop** → Drop video on upload area
2. **File Browser** → Click "Choose Video File"
3. **Demo** → Click "Try Demo Video"

**Supported formats:** MP4, AVI, MOV, WEBM (max 500MB)

---

### 4. 🎨 View Results

**Dashboard shows:**
- ✅ Model used
- ⏱️ Video duration
- 📊 FPS and frame count
- 🎯 Number of keyframes detected
- 📈 Importance curve graph
- 🖼️ Keyframe gallery with scores

---

## 📖 Full Documentation

- **Detailed Guide:** [UI_ENHANCEMENT_GUIDE.md](UI_ENHANCEMENT_GUIDE.md)
- **Features Summary:** [UI_FEATURES_SUMMARY.md](UI_FEATURES_SUMMARY.md)
- **Technical Details:** [UI_MODIFICATION_SUMMARY.md](UI_MODIFICATION_SUMMARY.md)

---

## 💡 Pro Tips

**For Fast Processing:**
- Use Model 2
- Set keyframes to 9-15
- Keep videos under 30 seconds

**For Best Quality:**
- Use Model 1
- Set keyframes to 15-30
- Ensure good video quality

**For Batch Work:**
- Use Model 2 for speed
- Process one at a time
- Consistent keyframe count

---

## 🆘 Need Help?

**Troubleshooting:**
1. Check [UI_ENHANCEMENT_GUIDE.md](UI_ENHANCEMENT_GUIDE.md) - Troubleshooting section
2. Review [VERIFICATION_TEST_REPORT.md](../VERIFICATION_TEST_REPORT.md)
3. Check server terminal for errors

---

**Enjoy your enhanced keyframe detection! 🎬✨**
