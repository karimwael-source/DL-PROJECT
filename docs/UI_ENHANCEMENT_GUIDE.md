# UI Enhancement Guide - Interactive Keyframe Detection

**Last Updated:** December 29, 2025  
**Version:** 2.0 Enhanced

---

## 🎨 New UI Features Overview

The web application has been completely redesigned with an **interactive, modern interface** that gives you full control over the keyframe detection process.

### ✨ What's New

1. **Model Selection**: Choose between Model 1 (ResNet50) or Model 2 (EfficientNet-B0)
2. **Custom Keyframe Count**: Select exactly how many keyframes you want (1-60)
3. **Interactive Dashboard**: Real-time visual feedback and animated components
4. **Enhanced Statistics**: Detailed information about model performance
5. **Responsive Design**: Works seamlessly on desktop, tablet, and mobile

---

## 🚀 Quick Start

### Starting the Server

```bash
# Option 1: Direct launch
python webapp/app.py

# Option 2: Using launcher
python run_webapp.py

# Option 3: Windows batch file
start_server.bat
```

### Access the Application

Open your browser and navigate to:
- **Main App**: http://localhost:5000
- **Alternative Interface v1**: http://localhost:5000/v1
- **Legacy Interface**: http://localhost:5000/old

---

## 🎯 Feature Guide

### 1. Model Selection

**Two powerful models to choose from:**

#### 🚀 Model 1: ResNet50 + Transformer
- **Parameters**: 26.4 Million
- **Accuracy**: High
- **Use Case**: When you need maximum accuracy and have sufficient compute resources
- **Characteristics**:
  - Deep CNN architecture with ResNet50 backbone
  - Dual temporal attention mechanism
  - Transformer-based importance scoring
  - Best for: Professional video analysis, research projects

#### ⚡ Model 2: EfficientNet-B0 + Transformer
- **Parameters**: 7.8 Million (70% fewer than Model 1)
- **Speed**: 28% faster inference
- **Use Case**: Quick processing, resource-constrained environments, real-time applications
- **Characteristics**:
  - Efficient CNN architecture with EfficientNet-B0 backbone
  - Dual temporal attention (optimized)
  - Reduced computational requirements
  - Best for: Fast prototyping, batch processing, mobile deployment

**How to Select:**
1. Click on the model card you want to use
2. The selected model will be highlighted with a gradient border
3. The active model is used for all subsequent processing

---

### 2. Custom Keyframe Count

**Full control over the number of extracted keyframes:**

#### Interactive Slider
- **Range**: 1 to 60 keyframes
- **Default**: 9 keyframes (15% of 60 frames)
- **Visual Feedback**: Real-time value display with gradient fill

#### Use Cases by Count

| Keyframes | Best For | Example Use Cases |
|-----------|----------|-------------------|
| 1-5 | Ultra-short summary | Video thumbnails, quick preview |
| 6-12 | Short summary | Social media content, highlights |
| 13-20 | Medium summary | Video chapters, presentations |
| 21-40 | Detailed analysis | Content review, editing timeline |
| 41-60 | Comprehensive extraction | Frame-by-frame analysis, research |

**How to Use:**
1. Drag the slider to select your desired number
2. The value updates in real-time (shown in the purple badge)
3. The slider gradient visually represents your selection
4. Changes apply to the next video processing

---

### 3. Video Upload

**Three ways to upload your video:**

#### Method 1: Drag & Drop
1. Drag your video file directly onto the upload area
2. The area highlights when a file is detected
3. Release to select the file

#### Method 2: File Browser
1. Click "📁 Choose Video File" button
2. Browse and select your video
3. Click "Open" to confirm

#### Method 3: Demo Video
1. Click "✨ Try Demo Video" to test with a synthetic video
2. No file upload required
3. Instant processing demonstration

**Supported Formats:**
- MP4 (recommended)
- AVI
- MOV
- WEBM
- Maximum file size: 500MB
- Recommended duration: Under 30 seconds for optimal performance

---

### 4. Processing & Results

#### Step 1: Configure
1. Select your model (Model 1 or Model 2)
2. Set the number of keyframes (1-60)
3. Upload your video

#### Step 2: Process
1. Click "🚀 Detect Keyframes"
2. Watch the animated loading spinner
3. See which model is processing your video

#### Step 3: View Results

**Dashboard displays:**

1. **Model Information**
   - Which model was used for processing
   - Model-specific statistics

2. **Video Statistics**
   - Duration (seconds)
   - FPS (frames per second)
   - Total frames analyzed
   - Number of keyframes detected

3. **Importance Curve Graph**
   - Visual representation of frame importance scores
   - Golden stars mark detected keyframes
   - Interactive plot (can be saved)

4. **Keyframe Gallery**
   - Grid layout of all detected keyframes
   - Each card shows:
     - Frame number
     - Timestamp in video
     - Importance score
     - High-quality thumbnail
   - Hover effects for better interaction
   - Smooth animations

---

## 💡 Tips & Best Practices

### Optimal Keyframe Count Selection

**For Different Video Types:**

1. **Short Videos (< 10 seconds)**
   - Use 3-8 keyframes
   - Captures major scene changes

2. **Medium Videos (10-30 seconds)**
   - Use 9-15 keyframes
   - Balanced coverage of content

3. **Long Videos (> 30 seconds)**
   - Use 20-40 keyframes
   - Comprehensive scene representation

### Model Selection Guide

**Choose Model 1 when:**
- Accuracy is paramount
- Processing time is not critical
- Working with complex videos
- Need for research-grade results

**Choose Model 2 when:**
- Speed is important
- Processing multiple videos
- Limited computational resources
- Deploying in production environments
- Mobile or edge device deployment

### Performance Optimization

1. **Video Resolution**
   - Videos are automatically resized to 224x224 for processing
   - Original aspect ratios are preserved in results
   - Higher resolution = longer processing time

2. **Frame Sampling**
   - System samples 60 frames uniformly from video
   - Longer videos = larger time gaps between samples
   - Keep videos under 30 seconds for best results

3. **Batch Processing**
   - Process one video at a time
   - Clear results between uploads for better performance
   - Use Model 2 for faster batch processing

---

## 🎨 UI Components Explained

### Model Cards
- **Interactive Selection**: Click to activate
- **Visual Feedback**: Active border and glow effect
- **Statistics Badges**: Quick reference for model specs
- **Hover Effects**: Smooth transitions and elevation

### Keyframe Slider
- **Real-time Updates**: Value changes as you drag
- **Gradient Fill**: Visual representation of selection
- **Markers**: Quick reference points at 1, 15, 30, 45, 60
- **Smooth Animation**: Polished interaction

### Upload Area
- **Drag & Drop Zone**: Large target area
- **Visual States**:
  - Default: Subtle gradient background
  - Hover: Enhanced glow
  - Dragover: Highlighted border and scale
  - Selected: Checkmark and filename display

### Results Section
- **Animated Entry**: Smooth slide-up animation
- **Interactive Cards**: Hover effects on keyframes
- **Responsive Grid**: Adapts to screen size
- **High Contrast**: Easy to read in any lighting

---

## 📊 Understanding Results

### Importance Curve

**What it shows:**
- X-axis: Frame index (0-59)
- Y-axis: Importance score (0-1)
- Blue line: Continuous importance across all frames
- Golden stars: Selected keyframes

**How to interpret:**
- Higher peaks = More important frames
- Valleys = Less important content
- Stars should be on or near peaks
- Distribution shows video content variation

### Keyframe Cards

**Information displayed:**

1. **Thumbnail Image**
   - High-quality frame capture
   - Maintains aspect ratio
   - Hover to enlarge slightly

2. **Frame Number**
   - Position in 60-frame sample
   - Useful for frame-accurate editing

3. **Timestamp**
   - Exact time in original video
   - Format: XX.XXs
   - Useful for video editors

4. **Importance Score**
   - Numerical value (0-1)
   - Higher = More important
   - Confidence indicator

---

## 🔧 Advanced Usage

### API Endpoints

#### Process Video
```
POST /process
Content-Type: multipart/form-data

Parameters:
- video: Video file (multipart)
- model: "model1" or "model2"
- num_keyframes: Integer (1-60)

Response: JSON with keyframes and statistics
```

#### Demo Video
```
GET /demo?model=model1&num_keyframes=15

Parameters:
- model: "model1" or "model2" (optional, default: model1)
- num_keyframes: Integer (1-60) (optional, default: 9)

Response: JSON with demo results
```

### Custom Integration

**Using fetch API:**
```javascript
const formData = new FormData();
formData.append('video', videoFile);
formData.append('model', 'model2');
formData.append('num_keyframes', 15);

const response = await fetch('/process', {
    method: 'POST',
    body: formData
});

const results = await response.json();
// Process results...
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Video Upload Fails**
- Check file format (MP4, AVI, MOV, WEBM)
- Verify file size < 500MB
- Ensure stable internet connection

**2. Slow Processing**
- Try Model 2 for faster results
- Reduce keyframe count
- Trim video to < 30 seconds
- Close other applications

**3. No Keyframes Detected**
- Check if video has content variation
- Try increasing keyframe count
- Verify video is not corrupted

**4. Browser Compatibility**
- Use modern browsers (Chrome, Firefox, Edge, Safari)
- Enable JavaScript
- Clear browser cache if needed

---

## 📱 Responsive Design

### Desktop (> 1024px)
- Full-width layout
- Multi-column grids
- All features visible

### Tablet (768px - 1024px)
- Responsive columns
- Optimized touch targets
- Simplified navigation

### Mobile (< 768px)
- Single-column layout
- Stacked model cards
- Large touch-friendly buttons
- Optimized image sizes

---

## 🎓 Example Workflows

### Workflow 1: Quick Video Summary
1. Select **Model 2** (fast)
2. Set keyframes to **9**
3. Upload video
4. Click "Detect Keyframes"
5. Download keyframe images

### Workflow 2: Detailed Analysis
1. Select **Model 1** (accurate)
2. Set keyframes to **25**
3. Upload video
4. Review importance curve
5. Identify scene transitions

### Workflow 3: Content Creation
1. Select **Model 2** (efficient)
2. Set keyframes to **12**
3. Process multiple videos
4. Extract thumbnails for social media
5. Create video chapters

---

## 🚀 Performance Metrics

### Model 1 (ResNet50)
- **Loading Time**: ~2-3 seconds
- **Processing Time**: ~2.5 seconds per video (CPU)
- **Memory Usage**: ~150MB
- **Accuracy**: High

### Model 2 (EfficientNet-B0)
- **Loading Time**: ~1-2 seconds
- **Processing Time**: ~1.8 seconds per video (CPU)
- **Memory Usage**: ~95MB
- **Accuracy**: Good (comparable to Model 1)

### System Requirements

**Minimum:**
- CPU: Dual-core 2.0 GHz
- RAM: 4GB
- Browser: Modern browser with JavaScript enabled

**Recommended:**
- CPU: Quad-core 2.5 GHz or GPU
- RAM: 8GB+
- Browser: Chrome 90+, Firefox 88+, Safari 14+

---

## 📝 Changelog

### Version 2.0 (December 29, 2025)
- ✅ Added Model 2 (EfficientNet-B0) support
- ✅ Interactive model selection UI
- ✅ Custom keyframe count slider (1-60)
- ✅ Enhanced dashboard with animations
- ✅ Model statistics display
- ✅ Improved responsive design
- ✅ Better error handling
- ✅ Performance optimizations

### Version 1.0
- Initial release with Model 1
- Basic file upload
- Fixed keyframe count (15% of frames)
- Simple result display

---

## 🙏 Credits

**Deep Learning Architecture:**
- Model 1: ResNet50 + Transformer + Dual Temporal Attention
- Model 2: EfficientNet-B0 + Transformer + Dual Temporal Attention

**Frontend:**
- Modern CSS3 with animations
- Vanilla JavaScript (no frameworks)
- Responsive design principles

**Backend:**
- Flask web framework
- PyTorch deep learning
- OpenCV video processing

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the verification report: [VERIFICATION_TEST_REPORT.md](../VERIFICATION_TEST_REPORT.md)
3. Check model documentation: [docs/MODEL2_README.md](../docs/MODEL2_README.md)

---

**Enjoy your enhanced keyframe detection experience! 🎬✨**
