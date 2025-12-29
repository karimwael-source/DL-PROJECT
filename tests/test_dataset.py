"""
Test dataset loading (run this AFTER you download TVSum dataset).
This verifies that videos and annotations load correctly.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# You need to set these paths to your actual dataset location
VIDEO_DIR = "E:/DL_project_finalized/data/videos"  # Updated!
H5_PATH = "E:/DL_project_finalized/data/tvsum.h5"    # Updated!


def test_dataset_loading():
    print("="*60)
    print("🧪 TESTING DATASET LOADING")
    print("="*60 + "\n")
    
    # Check if paths exist
    print("1️⃣ Checking paths...")
    
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        print("\n💡 TO FIX THIS:")
        print("   1. Download TVSum dataset")
        print("   2. Extract videos to a folder")
        print("   3. Update VIDEO_DIR in test_dataset.py")
        return False
    else:
        print(f"✓ Video directory found: {VIDEO_DIR}")
        
    if not os.path.exists(H5_PATH):
        print(f"❌ H5 file not found: {H5_PATH}")
        print("\n💡 TO FIX THIS:")
        print("   1. Download TVSum annotations (tvsum.h5)")
        print("   2. Update H5_PATH in test_dataset.py")
        return False
    else:
        print(f"✓ H5 file found: {H5_PATH}")
    
    # Try to import dataset module
    print("\n2️⃣ Loading dataset module...")
    try:
        from src.data.dataset import TVSumDataset
        print("[OK] Dataset module loaded")
    except Exception as e:
        print(f"[ERROR] Failed to import dataset: {e}")
        return False
    
    # Create dataset
    print("\n3️⃣ Creating dataset...")
    try:
        dataset = TVSumDataset(
            video_dir=VIDEO_DIR,
            h5_path=H5_PATH,
            split='train',
            num_frames=60,
            img_size=224
        )
        print(f"✓ Dataset created: {len(dataset)} videos")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        print("\nPossible issues:")
        print("   - H5 file format might be different")
        print("   - Videos might be missing")
        return False
    
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return False
    
    # Test loading one sample
    print("\n4️⃣ Loading sample video...")
    try:
        frames, scores, video_name = dataset[0]
        
        print(f"✓ Loaded video: {video_name}")
        print(f"  Frames shape: {frames.shape}")
        print(f"  Scores shape: {scores.shape}")
        print(f"  Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        # Verify shapes
        assert frames.shape[0] == 60, f"Expected 60 frames, got {frames.shape[0]}"
        assert frames.shape[1] == 3, f"Expected 3 channels (RGB), got {frames.shape[1]}"
        assert frames.shape[2] == 224 and frames.shape[3] == 224, "Expected 224x224 images"
        assert scores.shape[0] == 60, f"Expected 60 scores, got {scores.shape[0]}"
        
        print("\n✓ All shapes correct!")
        
    except Exception as e:
        print(f"❌ Failed to load video: {e}")
        print("\nPossible issues:")
        print("   - Video file is corrupted")
        print("   - Video codec not supported (try installing ffmpeg)")
        print("   - OpenCV not installed properly")
        return False
    
    # Test dataloader
    print("\n5️⃣ Testing DataLoader...")
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0  # Use 0 for Windows to avoid multiprocessing issues
        )
        
        # Load one batch
        frames_batch, scores_batch, names_batch = next(iter(dataloader))
        
        print(f"✓ Batch loaded")
        print(f"  Batch frames shape: {frames_batch.shape}")
        print(f"  Batch scores shape: {scores_batch.shape}")
        print(f"  Video names: {names_batch}")
        
        assert frames_batch.shape[0] == 2, "Expected batch size 2"
        assert frames_batch.shape[1] == 60, "Expected 60 frames"
        
        print("\n✓ DataLoader works!")
        
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL DATASET TESTS PASSED!")
    print("="*60)
    print("\n🎯 You're ready to start training!")
    print("   Run: python train.py --video_dir", VIDEO_DIR, "--h5_path", H5_PATH)
    
    return True


if __name__ == "__main__":
    print("\n⚠️  IMPORTANT: Update VIDEO_DIR and H5_PATH in this file first!\n")
    
    success = test_dataset_loading()
    
    if not success:
        print("\n" + "="*60)
        print("❌ TESTS FAILED")
        print("="*60)
        print("\n📋 Checklist:")
        print("   [ ] Download TVSum dataset")
        print("   [ ] Extract videos to a folder")
        print("   [ ] Update VIDEO_DIR in test_dataset.py")
        print("   [ ] Update H5_PATH in test_dataset.py")
        print("   [ ] Install required packages: pip install -r requirements.txt")
        print("   [ ] Install ffmpeg if needed: conda install ffmpeg")
        sys.exit(1)
