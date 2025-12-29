"""
Comprehensive verification script to check all project files
Tests imports, connections, and basic functionality
"""
import sys
import os
import traceback

# Add project directory to path
sys.path.insert(0, 'e:/DL_project_finalized')

def print_section(title):
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def test_import(module_name, file_path):
    """Test if a module can be imported."""
    try:
        if module_name == "model":
            from model import create_model, KeyFrameDetector
            print(f"✓ {module_name}.py - imports successfully")
            print(f"  - create_model: OK")
            print(f"  - KeyFrameDetector: OK")
            return True
        elif module_name == "dataset":
            from dataset import TVSumDataset, create_dataloaders
            print(f"✓ {module_name}.py - imports successfully")
            print(f"  - TVSumDataset: OK")
            print(f"  - create_dataloaders: OK")
            return True
        elif module_name == "train":
            from train import Trainer, RankingLoss
            print(f"✓ {module_name}.py - imports successfully")
            print(f"  - Trainer: OK")
            print(f"  - RankingLoss: OK")
            return True
        elif module_name == "visualize":
            from visualize import plot_importance_curve, visualize_keyframes, evaluate_model
            print(f"✓ {module_name}.py - imports successfully")
            print(f"  - plot_importance_curve: OK")
            print(f"  - visualize_keyframes: OK")
            print(f"  - evaluate_model: OK")
            return True
        elif module_name == "app":
            from app import app, predict_keyframes
            print(f"✓ {module_name}.py - imports successfully")
            print(f"  - Flask app: OK")
            print(f"  - predict_keyframes: OK")
            return True
        else:
            exec(f"import {module_name}")
            print(f"✓ {module_name}.py - imports successfully")
            return True
    except Exception as e:
        print(f"❌ {module_name}.py - FAILED")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        return False

def test_file_exists(file_path):
    """Check if file exists."""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✓ {os.path.basename(file_path)} exists ({size:,} bytes)")
        return True
    else:
        print(f"❌ {os.path.basename(file_path)} NOT FOUND")
        return False

def test_model_creation():
    """Test model creation and basic forward pass."""
    try:
        import torch
        from model import create_model
        
        print("\nCreating model...")
        model = create_model(freeze_resnet=True)
        print("✓ Model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 60, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, 60), f"Expected shape (1, 60), got {output.shape}"
        print("✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test dataset loading."""
    try:
        from dataset import TVSumDataset
        
        video_dir = "E:/DL_project_finalized/data/videos"
        h5_path = "E:/DL_project_finalized/data/tvsum.h5"
        
        if not os.path.exists(video_dir):
            print("⚠ Videos directory not found - skipping dataset test")
            return None
        
        if not os.path.exists(h5_path):
            print("⚠ H5 file not found - skipping dataset test")
            return None
        
        print("\nLoading dataset...")
        dataset = TVSumDataset(video_dir, h5_path, split='train')
        print(f"✓ Dataset loaded: {len(dataset)} videos")
        
        if len(dataset) > 0:
            frames, scores, name = dataset[0]
            print(f"✓ Sample loaded: {name}")
            print(f"  Frames: {frames.shape}")
            print(f"  Scores: {scores.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_integration():
    """Test that all components work together."""
    try:
        import torch
        from model import create_model
        from dataset import TVSumDataset
        
        video_dir = "E:/DL_project_finalized/data/videos"
        h5_path = "E:/DL_project_finalized/data/tvsum.h5"
        
        if not os.path.exists(video_dir) or not os.path.exists(h5_path):
            print("⚠ Dataset not found - skipping integration test")
            return None
        
        print("\nTesting model + dataset integration...")
        
        # Load model
        model = create_model(freeze_resnet=False)
        model.eval()
        print("✓ Model loaded")
        
        # Load dataset
        dataset = TVSumDataset(video_dir, h5_path, split='test')
        print(f"✓ Dataset loaded: {len(dataset)} videos")
        
        if len(dataset) > 0:
            # Test prediction
            frames, gt_scores, name = dataset[0]
            frames = frames.unsqueeze(0)
            
            with torch.no_grad():
                pred_scores = model(frames)
            
            pred_scores = pred_scores.squeeze(0)
            
            print(f"✓ Prediction successful for {name}")
            print(f"  Predicted scores: {pred_scores.shape}")
            print(f"  Ground truth: {gt_scores.shape}")
            print(f"  Pred range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
            print(f"  GT range: [{gt_scores.min():.3f}, {gt_scores.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_train_script():
    """Check if train.py can be imported and initialized."""
    try:
        from train import Trainer, RankingLoss
        import torch
        
        print("\nTesting training components...")
        
        # Test loss function
        loss_fn = RankingLoss()
        dummy_pred = torch.randn(2, 60)
        dummy_target = torch.randn(2, 60)
        loss = loss_fn(dummy_pred, dummy_target)
        
        print("✓ RankingLoss works")
        print(f"  Sample loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Training test failed: {str(e)}")
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'h5py',
        'matplotlib',
        'scipy',
        'tensorboard',
        'tqdm',
        'flask'
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    print(f"\n✓ Installed: {len(installed)}/{len(required_packages)} packages")
    for pkg in installed:
        print(f"  ✓ {pkg}")
    
    if missing:
        print(f"\n❌ Missing: {len(missing)} packages")
        for pkg in missing:
            print(f"  ❌ {pkg}")
        return False
    
    return True

def main():
    """Run all verification tests."""
    print("\n" + "🔍"*30)
    print("PROJECT VERIFICATION SCRIPT")
    print("🔍"*30)
    
    results = {}
    
    # Check dependencies
    print_section("Checking Dependencies")
    results['dependencies'] = check_dependencies()
    
    # Check file existence
    print_section("Checking File Existence")
    files_to_check = [
        'e:/DL_project_finalized/model.py',
        'e:/DL_project_finalized/dataset.py',
        'e:/DL_project_finalized/train.py',
        'e:/DL_project_finalized/visualize.py',
        'e:/DL_project_finalized/app.py',
        'e:/DL_project_finalized/test_model.py',
        'e:/DL_project_finalized/test_dataset.py',
        'e:/DL_project_finalized/requirements.txt',
        'e:/DL_project_finalized/README.md',
        'e:/DL_project_finalized/templates/index.html'
    ]
    
    file_check = all(test_file_exists(f) for f in files_to_check)
    results['files'] = file_check
    
    # Test imports
    print_section("Testing Module Imports")
    results['model_import'] = test_import('model', 'model.py')
    results['dataset_import'] = test_import('dataset', 'dataset.py')
    results['train_import'] = test_import('train', 'train.py')
    results['visualize_import'] = test_import('visualize', 'visualize.py')
    results['app_import'] = test_import('app', 'app.py')
    
    # Test model
    print_section("Testing Model Functionality")
    results['model_test'] = test_model_creation()
    
    # Test dataset
    print_section("Testing Dataset Loading")
    results['dataset_test'] = test_dataset_loading()
    
    # Test training components
    print_section("Testing Training Components")
    results['train_test'] = test_train_script()
    
    # Test integration
    print_section("Testing Integration")
    results['integration'] = test_integration()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    total = len(results)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is None:
            status = "⚠️  SKIP"
        else:
            status = "❌ FAIL"
        
        print(f"{status:12s} - {test_name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {skipped} skipped, {failed} failed out of {total} tests")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Project is ready to use!")
        print("\n📋 You can now:")
        print("   1. Train: python train.py --video_dir data/videos --h5_path data/tvsum.h5")
        print("   2. Visualize: python visualize.py --checkpoint checkpoints/best_model.pth")
        print("   3. Web app: python app.py (then open http://localhost:5000)")
    elif skipped > 0 and failed == 0:
        print("\n✅ Core components work! Some tests skipped (dataset not found)")
        print("\n📋 To complete setup:")
        print("   1. Run: python download_dataset.py")
        print("   2. Then train your model")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
