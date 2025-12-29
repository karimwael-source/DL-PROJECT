"""
Download and setup TVSum dataset using Kaggle API
Run this script to automatically download and prepare the dataset
"""
import os
import shutil
import zipfile
import subprocess
import json

# Configuration
KAGGLE_JSON_PATH = "e:/Downloads/kaggle.json"
DATASET_NAME = "hafianerabah/tvsum-videos"
PROJECT_DIR = "e:/DL_project_finalized"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
KAGGLE_DIR = os.path.expanduser("~/.kaggle")


def setup_kaggle_credentials():
    """Setup Kaggle API credentials."""
    print("="*60)
    print("🔧 Setting up Kaggle credentials...")
    print("="*60)
    
    # Create .kaggle directory
    os.makedirs(KAGGLE_DIR, exist_ok=True)
    
    # Copy kaggle.json
    kaggle_dest = os.path.join(KAGGLE_DIR, "kaggle.json")
    
    if os.path.exists(KAGGLE_JSON_PATH):
        shutil.copy(KAGGLE_JSON_PATH, kaggle_dest)
        print(f"✓ Copied kaggle.json to {kaggle_dest}")
        
        # Set permissions (on Windows, this is less critical)
        try:
            os.chmod(kaggle_dest, 0o600)
            print("✓ Set permissions on kaggle.json")
        except:
            print("⚠ Could not set permissions (Windows doesn't need this)")
    else:
        print(f"❌ Kaggle.json not found at {KAGGLE_JSON_PATH}")
        return False
    
    return True


def install_kaggle():
    """Install Kaggle API package."""
    print("\n" + "="*60)
    print("📦 Installing Kaggle API...")
    print("="*60)
    
    try:
        import kaggle
        print("✓ Kaggle API already installed")
        return True
    except ImportError:
        print("Installing kaggle package...")
        result = subprocess.run(
            ["pip", "install", "kaggle"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Kaggle API installed successfully")
            return True
        else:
            print(f"❌ Failed to install kaggle: {result.stderr}")
            return False


def download_dataset():
    """Download TVSum videos from Kaggle."""
    print("\n" + "="*60)
    print("📥 Downloading TVSum dataset from Kaggle...")
    print("="*60)
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download using kaggle CLI
    print(f"Downloading {DATASET_NAME}...")
    print("This may take a few minutes (dataset is ~10GB)...\n")
    
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", DATA_DIR],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Dataset downloaded successfully")
        return True
    else:
        print(f"❌ Download failed: {result.stderr}")
        return False


def extract_dataset():
    """Extract downloaded zip file."""
    print("\n" + "="*60)
    print("📂 Extracting dataset...")
    print("="*60)
    
    zip_path = os.path.join(DATA_DIR, "tvsum-videos.zip")
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        return False
    
    print(f"Extracting {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        print("✓ Dataset extracted successfully")
        
        # Remove zip file to save space
        os.remove(zip_path)
        print("✓ Cleaned up zip file")
        
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def download_tvsum_annotations():
    """Clone TVSum repo for annotations."""
    print("\n" + "="*60)
    print("📥 Downloading TVSum annotations...")
    print("="*60)
    
    tvsum_dir = os.path.join(DATA_DIR, "tvsum")
    
    # Remove existing directory
    if os.path.exists(tvsum_dir):
        print("Removing existing tvsum directory...")
        shutil.rmtree(tvsum_dir)
    
    # Clone repository
    print("Cloning TVSum repository...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/yalesong/tvsum.git", tvsum_dir],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ TVSum annotations downloaded successfully")
        return True
    else:
        print(f"⚠ Git clone failed: {result.stderr}")
        print("Trying alternative method...")
        
        # Alternative: download directly
        import urllib.request
        mat_url = "https://github.com/yalesong/tvsum/raw/master/matlab/ydata-tvsum50.mat"
        mat_path = os.path.join(DATA_DIR, "ydata-tvsum50.mat")
        
        try:
            os.makedirs(tvsum_dir, exist_ok=True)
            matlab_dir = os.path.join(tvsum_dir, "matlab")
            os.makedirs(matlab_dir, exist_ok=True)
            
            dest_path = os.path.join(matlab_dir, "ydata-tvsum50.mat")
            print(f"Downloading from {mat_url}...")
            urllib.request.urlretrieve(mat_url, dest_path)
            print(f"✓ Downloaded annotations to {dest_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to download annotations: {e}")
            return False


def convert_mat_to_h5():
    """Convert MATLAB file to H5 format compatible with our dataset loader."""
    print("\n" + "="*60)
    print("🔄 Converting annotations to H5 format...")
    print("="*60)
    
    try:
        import h5py
        import scipy.io as sio
    except ImportError:
        print("Installing scipy...")
        subprocess.run(["pip", "install", "scipy"], check=True)
        import scipy.io as sio
    
    mat_path = os.path.join(DATA_DIR, "tvsum", "matlab", "ydata-tvsum50.mat")
    h5_path = os.path.join(DATA_DIR, "tvsum.h5")
    
    if not os.path.exists(mat_path):
        print(f"❌ MATLAB file not found: {mat_path}")
        return False
    
    print(f"Reading {mat_path}...")
    
    try:
        # Read MATLAB file
        mat_data = h5py.File(mat_path, 'r')
        g = mat_data['tvsum50']
        
        # Extract video references
        video_refs = g['video'][:]
        if video_refs.shape[0] == 1:
            video_refs = video_refs[0]
        elif video_refs.shape[1] == 1:
            video_refs = video_refs[:, 0]
        
        # Create new H5 file
        print(f"Creating {h5_path}...")
        with h5py.File(h5_path, 'w') as h5_out:
            
            for i, ref in enumerate(video_refs):
                # Decode video ID
                video_id_arr = mat_data[ref][:]
                video_id = "".join([chr(x) for x in video_id_arr.flatten()])
                
                print(f"Processing video {i+1}/50: {video_id}")
                
                # Get user summaries
                user_ref = g['user_anno'][i, 0]
                user_scores = mat_data[user_ref][:]  # Shape: (num_frames, num_annotators)
                
                # Transpose to (num_annotators, num_frames)
                user_scores = user_scores.T
                
                # Create group for this video
                video_group = h5_out.create_group(video_id)
                video_group.create_dataset('user_summary', data=user_scores)
                video_group.attrs['video_id'] = video_id
        
        mat_data.close()
        
        print(f"✓ Created H5 file: {h5_path}")
        print(f"✓ Processed 50 videos")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def organize_videos():
    """Organize videos into proper directory structure."""
    print("\n" + "="*60)
    print("📁 Organizing video files...")
    print("="*60)
    
    videos_dir = os.path.join(DATA_DIR, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Find all video files in data directory
    video_count = 0
    for file in os.listdir(DATA_DIR):
        if file.endswith(('.mp4', '.avi', '.webm', '.mkv', '.mov')):
            src = os.path.join(DATA_DIR, file)
            dst = os.path.join(videos_dir, file)
            
            if not os.path.exists(dst):
                shutil.move(src, dst)
                video_count += 1
    
    print(f"✓ Organized {video_count} video files")
    
    # List videos
    videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.webm', '.mkv', '.mov'))]
    print(f"✓ Total videos in directory: {len(videos)}")
    
    return True


def verify_setup():
    """Verify that everything is set up correctly."""
    print("\n" + "="*60)
    print("✅ Verifying setup...")
    print("="*60)
    
    videos_dir = os.path.join(DATA_DIR, "videos")
    h5_path = os.path.join(DATA_DIR, "tvsum.h5")
    
    errors = []
    
    # Check videos directory
    if not os.path.exists(videos_dir):
        errors.append(f"Videos directory not found: {videos_dir}")
    else:
        videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.webm', '.mkv', '.mov'))]
        print(f"✓ Videos directory: {len(videos)} videos")
        
        if len(videos) == 0:
            errors.append("No video files found")
    
    # Check H5 file
    if not os.path.exists(h5_path):
        errors.append(f"H5 file not found: {h5_path}")
    else:
        try:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                num_videos = len(f.keys())
                print(f"✓ H5 annotation file: {num_videos} videos")
                
                if num_videos == 0:
                    errors.append("H5 file is empty")
        except Exception as e:
            errors.append(f"H5 file is corrupted: {e}")
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE!")
        print("="*60)
        print(f"\nDataset location:")
        print(f"  Videos: {videos_dir}")
        print(f"  Annotations: {h5_path}")
        print(f"\nYou can now run:")
        print(f"  python test_dataset.py")
        print(f"  python train.py --video_dir {videos_dir} --h5_path {h5_path}")
        return True


def main():
    """Main setup process."""
    print("\n" + "🎬"*30)
    print("TVSUM DATASET SETUP")
    print("🎬"*30 + "\n")
    
    # Step 1: Setup Kaggle credentials
    if not setup_kaggle_credentials():
        print("\n❌ Failed to setup Kaggle credentials")
        return
    
    # Step 2: Install Kaggle API
    if not install_kaggle():
        print("\n❌ Failed to install Kaggle API")
        return
    
    # Step 3: Download dataset
    if not download_dataset():
        print("\n❌ Failed to download dataset")
        return
    
    # Step 4: Extract dataset
    if not extract_dataset():
        print("\n❌ Failed to extract dataset")
        return
    
    # Step 5: Download annotations
    if not download_tvsum_annotations():
        print("\n⚠ Failed to download annotations, but continuing...")
    
    # Step 6: Convert to H5
    if not convert_mat_to_h5():
        print("\n❌ Failed to convert annotations")
        return
    
    # Step 7: Organize videos
    if not organize_videos():
        print("\n❌ Failed to organize videos")
        return
    
    # Step 8: Verify
    verify_setup()


if __name__ == "__main__":
    main()
