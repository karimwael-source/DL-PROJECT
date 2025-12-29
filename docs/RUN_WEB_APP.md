# 🌐 Web Application - Visual Testing Guide

## 🎯 What This Does

Beautiful web interface to visually test your keyframe detection model:
- Upload any video
- See real-time processing
- View importance curve plot
- Display detected keyframes in a grid
- Try demo video (no upload needed)

## 🚀 Quick Start

### 1. Install Flask
```bash
pip install flask
```

### 2. Run the Web App
```bash
python app.py
```

### 3. Open Browser
Go to: **http://localhost:5000**

## 📱 How to Use

### Option A: Upload Your Video
1. Drag & drop video OR click "Choose Video File"
2. Click "🚀 Detect Keyframes"
3. Wait for processing (10-30 seconds)
4. See results:
   - Video info (duration, FPS, frames)
   - Importance curve graph
   - Keyframe grid with scores

### Option B: Try Demo Video
1. Click "🎨 Try Demo Video"
2. Instantly see how it works with synthetic video
3. No upload needed!

## 🎨 Features

### ✨ Beautiful UI
- Modern gradient design
- Responsive layout
- Drag & drop support
- Loading animations
- Smooth transitions

### 📊 Visualizations
- **Importance Curve**: Line plot showing frame importance over time
- **Keyframe Grid**: Visual gallery of detected keyframes
- **Scores**: Importance score for each keyframe
- **Timestamps**: Time position of each keyframe

### 🎯 Information Display
- Video duration
- FPS (frames per second)
- Total frames analyzed
- Number of keyframes detected

## 🔧 Technical Details

### Architecture
```
Frontend (HTML/CSS/JS)
    ↓ Upload Video
Backend (Flask/Python)
    ↓ Extract 60 frames (2 FPS)
Model (ResNet + Transformer)
    ↓ Predict importance scores
Visualization
    ↓ Plot + Keyframes
Frontend Display
```

### Processing Steps
1. **Upload**: User uploads video
2. **Extract**: Sample 60 frames uniformly
3. **Transform**: Resize to 224×224, normalize
4. **Predict**: Run through model → importance scores
5. **Select**: Top 15% frames as keyframes
6. **Visualize**: Create plot + encode images
7. **Display**: Show results in browser

## 🎥 Supported Formats

- MP4 (recommended)
- AVI
- MOV
- WEBM
- MKV

## ⚠️ Tips

### For Best Results:
- Use videos 10-60 seconds long
- Higher quality = better results
- Diverse scenes work best
- File size: < 500MB

### If Processing is Slow:
- First run downloads ResNet weights (~100MB)
- CPU processing: 20-40 seconds
- GPU processing: 5-10 seconds

## 🐛 Troubleshooting

**Server won't start:**
```bash
# Install Flask
pip install flask
```

**"Cannot open video" error:**
- Check video format
- Install codecs: `conda install ffmpeg`

**Out of memory:**
- Close other programs
- Use shorter videos
- Model runs on CPU by default

**Port 5000 in use:**
```python
# Edit app.py, change last line:
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📸 Screenshots

### Main Page
- Upload area with drag & drop
- Process and Demo buttons
- Clean, modern interface

### Results Page
- Video statistics
- Importance curve graph
- Keyframe gallery with scores

## 🎓 For Your Project

Perfect for:
- **Demonstration**: Show your professor how it works
- **Testing**: Quickly test different videos
- **Debugging**: Visually verify model output
- **Presentation**: Live demo during project defense

## 🔥 Advanced Usage

### Custom Video Path
```python
# Test with specific video
results = predict_keyframes('path/to/video.mp4')
```

### Save Results
Results auto-display in browser, but you can also:
- Right-click plot → Save image
- Right-click keyframe → Save image

### API Endpoint
Use programmatically:
```python
import requests

files = {'video': open('video.mp4', 'rb')}
response = requests.post('http://localhost:5000/process', files=files)
data = response.json()

print(f"Detected {data['num_keyframes']} keyframes")
```

## 🚦 Status Indicators

- 🔵 **Blue button**: Ready to process
- ⏳ **Spinner**: Processing...
- ✅ **Green box**: Success!
- ❌ **Red box**: Error occurred

## 💡 Next Steps

1. Run: `python app.py`
2. Open: http://localhost:5000
3. Upload a video or try demo
4. See your model in action!

---

**No dataset needed** - works with any video you upload! 🎉
