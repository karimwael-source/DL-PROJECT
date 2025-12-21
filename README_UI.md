# 🎨 AI Keyframe Detection - UI Guide

## ✨ Two Beautiful Interfaces Available

### **New Stunning UI** (Default - Recommended)
- 🌐 **Access at:** http://localhost:5000
- **Features:**
  - 3D rotating rings background
  - Floating particle animations
  - Dark elegant glassmorphism theme
  - Smooth transitions and hover effects
  - Professional gradient colors
  - Mobile responsive design

### **Original UI** (Alternative)
- 🌐 **Access at:** http://localhost:5000/old
- **Features:**
  - Clean and simple interface
  - Purple/pink gradient theme
  - Card-based layout

---

## 🚀 How to Start the Server

### **Option 1: Double-click the batch file**
```
start_server.bat
```

### **Option 2: Run from terminal**
```powershell
cd E:\DL_project_finalized
E:\python.exe app.py
```

### **Option 3: Python command**
```bash
python app.py
```

---

## 📝 Usage Instructions

1. **Start the server** using any method above
2. **Open your browser** to http://localhost:5000
3. **Click "Start Detection"** button
4. **Upload a video** or try the demo
5. **View results:**
   - Video statistics (duration, FPS, frames)
   - Importance curve plot
   - Detected keyframes gallery

---

## 🎬 Features

### **Upload Your Own Video**
- Drag & drop or click to browse
- Supports: MP4, AVI, MOV, WEBM
- Max 30 seconds recommended for best results

### **Try Demo Video**
- Click "✨ Try Demo Video" button
- Uses synthetic video with color transitions
- See the AI in action instantly

### **View Results**
- **Importance Curve:** Visual graph showing frame importance
- **Keyframes Grid:** Top 15% most important frames
- **Frame Details:** Frame number, timestamp, and importance score

---

## 🎨 Animations Included

- 🔄 3D ring rotation (perspective effect)
- ✨ Floating particles background
- 💫 Shimmer effects on cards
- 🌊 Background pulse animation
- 📍 Badge pulse effect
- 🎯 Button hover glow
- 🎪 Card pop-in animations
- 🔮 Loading spinner with dual rings
- 🎭 Glassmorphism with backdrop blur

---

## 📁 Project Structure

```
E:\DL_project_finalized\
│
├── app.py                  # Flask application
├── start_server.bat        # Quick launcher
│
├── templates/
│   ├── index_new.html     # New stunning UI ⭐
│   └── index.html         # Original UI
│
├── static/
│   ├── style.css          # All animations & styles
│   └── outputs/           # Generated visualizations
│
├── model.py               # AI model architecture
├── dataset.py             # Video processing
└── data/                  # TVSum dataset
```

---

## 🐛 Troubleshooting

### Server won't start?
```powershell
# Check if port 5000 is available
netstat -ano | findstr :5000

# Kill any process using port 5000
Get-Process -Id <PID> | Stop-Process -Force
```

### Can't access the page?
1. Make sure server shows: "Running on http://127.0.0.1:5000"
2. Try: http://127.0.0.1:5000 or http://localhost:5000
3. Check firewall isn't blocking port 5000

### Model takes too long to load?
- Model loads only when processing first video (lazy loading)
- Initial page appears instantly
- First video processing takes ~30-60 seconds

---

## 💡 Tips

- **Best video length:** 10-30 seconds
- **Optimal format:** MP4 with H.264 codec
- **Frame count:** System extracts 60 frames uniformly
- **Keyframe selection:** Top 15% most important frames
- **Use demo first:** Test with synthetic video before uploading

---

## 🎯 Current Status

✅ **Server:** Working perfectly  
✅ **UI:** Both interfaces functional  
✅ **Model:** Loads on-demand  
✅ **Processing:** Successfully tested  
✅ **Animations:** All effects active  
✅ **Mobile:** Responsive design  

---

## 🌟 What Makes This Special

1. **Dual Temporal Attention** - AI analyzes both local and global patterns
2. **ResNet50 + Transformer** - State-of-the-art architecture
3. **Pre-trained Weights** - Transfer learning from ImageNet
4. **Real-time Processing** - Get results in seconds
5. **Beautiful Visualization** - Professional charts and galleries
6. **3D Animations** - Stunning visual effects

---

**Enjoy your AI-powered keyframe detection! 🎬✨**
