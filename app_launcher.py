"""
Simple launcher with progress indicator for Flask app
"""
import sys
import time

print("\n" + "="*60)
print("   🚀 AI KEYFRAME DETECTION - Server Launcher")
print("="*60)

print("\n[1/4] Loading Python modules...", end="", flush=True)
time.sleep(0.5)
print(" ✓")

print("[2/4] Importing PyTorch (this may take 10-15 seconds)...", end="", flush=True)
import torch
print(" ✓")

print("[3/4] Loading Flask and dependencies...", end="", flush=True)
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import create_model
from torchvision import transforms
print(" ✓")

print("[4/4] Initializing Flask app...", end="", flush=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global model variable - load lazily
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_if_needed():
    """Load model only when needed."""
    global model
    if model is None:
        print(f"\n🤖 Loading AI model on {device}...")
        model = create_model(freeze_resnet=False)
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully!\n")
    return model

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(" ✓")

def extract_frames(video_path, num_frames=60):
    """Extract frames from video at uniform intervals."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    raw_frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame_rgb)
        
        # Transform for model
        frame_tensor = transform(frame_rgb)
        frames.append(frame_tensor)
    
    cap.release()
    
    frames = torch.stack(frames)
    return frames, raw_frames, fps, duration


def predict_keyframes(video_path):
    """Process video and predict keyframes."""
    model = load_model_if_needed()
    
    # Extract frames
    frames, raw_frames, fps, duration = extract_frames(video_path, num_frames=60)
    
    # Add batch dimension
    frames = frames.unsqueeze(0).to(device)
    
    # Predict importance scores
    with torch.no_grad():
        importance_scores = model(frames)
    
    importance_scores = importance_scores.squeeze(0).cpu().numpy()
    
    # Select top 15% as keyframes
    k = int(0.15 * len(importance_scores))
    keyframe_indices = np.argsort(importance_scores)[-k:]
    keyframe_indices = np.sort(keyframe_indices)
    
    return {
        'importance_scores': importance_scores.tolist(),
        'keyframe_indices': keyframe_indices.tolist(),
        'raw_frames': raw_frames,
        'fps': fps,
        'duration': duration,
        'total_frames': len(raw_frames)
    }


def create_importance_plot(scores, keyframe_indices):
    """Create importance curve plot."""
    plt.figure(figsize=(12, 5))
    
    frames = np.arange(len(scores))
    
    # Plot importance curve
    plt.plot(frames, scores, 'b-', linewidth=2, label='Importance Score', marker='o', markersize=3)
    
    # Highlight keyframes
    plt.scatter(keyframe_indices, [scores[i] for i in keyframe_indices],
                color='gold', s=150, marker='*', zorder=5, label='Detected Keyframes', edgecolors='red', linewidths=1.5)
    
    plt.xlabel('Frame Index', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('Frame Importance Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64


def frame_to_base64(frame):
    """Convert numpy frame to base64."""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


@app.route('/')
def index():
    """Render main page."""
    return render_template('index_new.html')

@app.route('/old')
def index_old():
    """Render old interface."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_video():
    """Process uploaded video."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        print(f"\n📹 Processing video: {filename}")
        
        # Predict keyframes
        results = predict_keyframes(filepath)
        
        # Create importance plot
        plot_base64 = create_importance_plot(results['importance_scores'], results['keyframe_indices'])
        
        # Prepare keyframe data
        keyframes = []
        for idx in results['keyframe_indices']:
            frame = results['raw_frames'][idx]
            img_base64 = frame_to_base64(frame)
            
            time_sec = (idx / len(results['raw_frames'])) * results['duration']
            
            keyframes.append({
                'index': int(idx),
                'image': img_base64,
                'score': float(results['importance_scores'][idx]),
                'time': f"{time_sec:.2f}s"
            })
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'keyframes': keyframes,
            'num_keyframes': len(keyframes),
            'video_info': {
                'fps': float(results['fps']),
                'duration': float(results['duration']),
                'total_frames': results['total_frames']
            }
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/demo', methods=['GET'])
def demo():
    """Demo with synthetic video."""
    # Create synthetic video
    demo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'demo_video.mp4')
    
    # Create a colorful test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(demo_path, fourcc, 30.0, (640, 480))
    
    # Create 900 frames (30 seconds at 30fps)
    for i in range(900):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Different scenes with transitions
        if i < 150:  # Blue scene
            frame[:, :] = [50, 50, 200]
        elif i < 300:  # Green scene
            frame[:, :] = [50, 200, 50]
        elif i < 450:  # Red scene
            frame[:, :] = [200, 50, 50]
        elif i < 600:  # Yellow scene
            frame[:, :] = [200, 200, 50]
        elif i < 750:  # Purple scene
            frame[:, :] = [150, 50, 150]
        else:  # Cyan scene
            frame[:, :] = [50, 200, 200]
        
        # Add frame number
        cv2.putText(frame, f"Frame {i}", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    
    try:
        print(f"\n🎬 Processing demo video...")
        
        # Process the demo video
        results = predict_keyframes(demo_path)
        
        # Create plot
        plot_base64 = create_importance_plot(results['importance_scores'], results['keyframe_indices'])
        
        # Prepare keyframe data
        keyframes = []
        for idx in results['keyframe_indices']:
            frame = results['raw_frames'][idx]
            img_base64 = frame_to_base64(frame)
            
            time_sec = (idx / len(results['raw_frames'])) * results['duration']
            
            keyframes.append({
                'index': int(idx),
                'image': img_base64,
                'score': float(results['importance_scores'][idx]),
                'time': f"{time_sec:.2f}s"
            })
        
        # Clean up
        os.remove(demo_path)
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'keyframes': keyframes,
            'num_keyframes': len(keyframes),
            'video_info': {
                'fps': float(results['fps']),
                'duration': float(results['duration']),
                'total_frames': results['total_frames']
            }
        })
        
    except Exception as e:
        if os.path.exists(demo_path):
            os.remove(demo_path)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("   🌐 SERVER READY!")
    print("="*60)
    print(f"   Device: {device}")
    print(f"   URL: http://localhost:5000")
    print(f"   Alternative URL: http://10.21.3.145:5000")
    print("="*60)
    print("\n   ⚡ Open your browser and go to: http://localhost:5000")
    print("   ⏹️  Press CTRL+C to stop the server\n")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
