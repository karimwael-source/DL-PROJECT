from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
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
        print(f"Loading model on {device}...")
        model = create_model(freeze_resnet=False)
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully!")
    return model

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_frames(video_path, num_frames=60):
    """Extract frames from video at uniform intervals."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    raw_frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            if len(frames) > 0:
                frame = raw_frames[-1]
            else:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame_rgb.copy())
        
        # Transform for model
        frame_tensor = transform(frame_rgb)
        frames.append(frame_tensor)
    
    cap.release()
    
    return torch.stack(frames), raw_frames, fps, duration


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
    plt.title('🎯 Frame Importance Curve', fontsize=14, fontweight='bold')
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
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Save video
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)
    
    try:
        # Process video
        print(f"Processing video: {filename}")
        results = predict_keyframes(video_path)
        
        # Create importance curve plot
        plot_base64 = create_importance_plot(results['importance_scores'], results['keyframe_indices'])
        
        # Convert keyframes to base64
        keyframes_base64 = []
        for idx in results['keyframe_indices']:
            frame = results['raw_frames'][idx]
            keyframes_base64.append({
                'index': int(idx),
                'score': float(results['importance_scores'][idx]),
                'image': frame_to_base64(frame),
                'time': f"{(idx / results['total_frames'] * results['duration']):.2f}s"
            })
        
        # Clean up
        os.remove(video_path)
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'keyframes': keyframes_base64,
            'num_keyframes': len(keyframes_base64),
            'video_info': {
                'fps': float(results['fps']),
                'duration': float(results['duration']),
                'total_frames': results['total_frames']
            }
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return jsonify({'error': str(e)}), 500


@app.route('/demo')
def demo():
    """Create a demo with synthetic video."""
    # Create synthetic video (30 seconds, changing colors)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'demo_video.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    # Generate 900 frames (30 seconds at 30fps)
    for i in range(900):
        # Create frame with changing colors
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Different scenes
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
    
    # Process demo video
    results = predict_keyframes(output_path)
    plot_base64 = create_importance_plot(results['importance_scores'], results['keyframe_indices'])
    
    keyframes_base64 = []
    for idx in results['keyframe_indices']:
        frame = results['raw_frames'][idx]
        keyframes_base64.append({
            'index': int(idx),
            'score': float(results['importance_scores'][idx]),
            'image': frame_to_base64(frame),
            'time': f"{(idx / results['total_frames'] * results['duration']):.2f}s"
        })
    
    os.remove(output_path)
    
    return jsonify({
        'success': True,
        'plot': plot_base64,
        'keyframes': keyframes_base64,
        'num_keyframes': len(keyframes_base64),
        'video_info': {
            'fps': float(results['fps']),
            'duration': float(results['duration']),
            'total_frames': results['total_frames']
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 KEYFRAME DETECTION WEB APP")
    print("="*60)
    print(f"Device: {device}")
    print("Server starting at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
