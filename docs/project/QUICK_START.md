# 🚀 Quick Start Guide

## ✅ **FIXED: Server Now Works Perfectly!**

### **The Problem:**
PyTorch import takes 10-15 seconds, and you were pressing CTRL+C during startup (KeyboardInterrupt).

### **The Solution:**
New launcher with progress indicator shows each loading step.

---

## 🎯 **How to Start the Server:**

### **Method 1: Double-Click (Easiest)**
```
📁 Double-click: start_server.bat
```
- Wait for all 4 steps to complete (✓ marks)
- Server will show "SERVER READY!"
- Open browser to http://localhost:5000

### **Method 2: PowerShell**
```powershell
cd E:\DL_project_finalized
.\start_server.ps1
```

### **Method 3: Direct Python**
```powershell
cd E:\DL_project_finalized
E:\python.exe app_launcher.py
```

---

## ⏳ **Loading Steps (DO NOT INTERRUPT!):**

```
[1/4] Loading Python modules... ✓       (1 second)
[2/4] Importing PyTorch...      ✓       (10-15 seconds) ⚠️ WAIT!
[3/4] Loading Flask...          ✓       (2 seconds)
[4/4] Initializing Flask app... ✓       (1 second)

🌐 SERVER READY!
```

**Total time: ~15-20 seconds**

---

## 🌐 **Access the Application:**

Once you see "SERVER READY!", open your browser:

**Primary URL:**
```
http://localhost:5000
```

**Alternative URLs:**
```
http://127.0.0.1:5000
http://10.21.3.145:5000
```

---

## 🎨 **Two Interfaces Available:**

### **New Stunning UI** (Default)
- http://localhost:5000
- 3D rotating rings background
- Floating particles animation
- Glassmorphism design
- Dark elegant theme

### **Original UI** (Backup)
- http://localhost:5000/old
- Simple clean design
- Purple gradient theme

---

## ✨ **Features:**

1. **Click "Start Detection"** button
2. **Upload your video** or click "Try Demo Video"
3. **View results:**
   - Video statistics (Duration, FPS, Total Frames, Keyframes)
   - Importance curve graph
   - Keyframe gallery with scores

---

## ⚠️ **Important Notes:**

### **During Startup:**
- ✅ **DO** wait for all 4 loading steps
- ✅ **DO** wait for "SERVER READY!" message
- ❌ **DON'T** press CTRL+C during loading
- ❌ **DON'T** close window during startup

### **First Video Processing:**
- Model loads automatically on first use
- Takes 30-60 seconds for first video
- Subsequent videos are faster
- Shows "Loading AI model..." message

### **Stopping the Server:**
- Press **CTRL+C** in the terminal
- Or close the terminal window
- Server stops immediately

---

## 🐛 **Troubleshooting:**

### **"KeyboardInterrupt" Error:**
- **Cause:** You pressed CTRL+C during PyTorch import
- **Fix:** Don't interrupt! Wait for all ✓ marks

### **Server Won't Start:**
```powershell
# Kill any existing Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Try again
.\start_server.bat
```

### **Port 5000 Already in Use:**
```powershell
# Find what's using port 5000
netstat -ano | findstr :5000

# Kill that process (replace PID)
Stop-Process -Id <PID> -Force
```

### **Can't Access Page:**
1. Check server shows "Running on http://127.0.0.1:5000"
2. Try http://127.0.0.1:5000 instead of localhost
3. Check Windows Firewall isn't blocking port 5000
4. Make sure you waited for "SERVER READY!"

---

## 📊 **What Works:**

✅ Server startup with progress indicator  
✅ Both HTML interfaces (new & old)  
✅ CSS with 30+ animations  
✅ Video upload and processing  
✅ Demo video generation  
✅ Keyframe detection  
✅ Importance curve plotting  
✅ Lazy model loading  
✅ Tested and confirmed working  

---

## 🎬 **Usage Tips:**

- **Best video length:** 10-30 seconds
- **Supported formats:** MP4, AVI, MOV, WEBM
- **Processing time:** 30-60 seconds first time, 10-20 seconds after
- **Keyframes selected:** Top 15% most important frames
- **Try demo first:** Test with synthetic video before uploading

---

## 📁 **Project Files:**

```
E:\DL_project_finalized\
│
├── start_server.bat        ⭐ Double-click to start
├── start_server.ps1        PowerShell launcher
├── app_launcher.py         Main app with progress
├── app.py                  Original app (backup)
│
├── templates/
│   ├── index_new.html      New stunning UI ⭐
│   └── index.html          Original UI
│
├── static/
│   └── style.css           All animations
│
├── model.py                AI architecture
├── dataset.py              Video processing
└── data/                   TVSum dataset
```

---

## 🎉 **You're All Set!**

Just run `start_server.bat` and open http://localhost:5000 in your browser!

**Enjoy your AI-powered keyframe detection! 🎬✨**
