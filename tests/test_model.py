"""
Quick test script to verify the model works without dataset.
Tests model architecture and forward pass with dummy data.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.model1 import create_model


def test_model():
    print("="*60)
    print("🧪 TESTING MODEL ARCHITECTURE")
    print("="*60 + "\n")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    print("\n1️⃣ Creating model...")
    model = create_model(freeze_resnet=True)
    model = model.to(device)
    print("✓ Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Test forward pass
    print("\n2️⃣ Testing forward pass...")
    batch_size = 2
    num_frames = 60
    
    # Dummy input: 2 videos, 60 frames each, 224x224 RGB
    dummy_input = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        importance_scores = model(dummy_input)
    
    print(f"Output shape: {importance_scores.shape}")
    print(f"Output range: [{importance_scores.min():.4f}, {importance_scores.max():.4f}]")
    
    # Check if outputs are valid
    assert importance_scores.shape == (batch_size, num_frames), "Wrong output shape!"
    assert torch.all(importance_scores >= 0) and torch.all(importance_scores <= 1), "Scores should be in [0, 1]!"
    
    print("\n✓ Forward pass successful!")
    
    # Test top-k keyframe selection
    print("\n3️⃣ Testing keyframe selection...")
    k = int(0.15 * num_frames)  # Top 15%
    keyframe_indices = torch.argsort(importance_scores[0], descending=True)[:k]
    
    print(f"Selected top {k} keyframes from 60 frames:")
    print(f"Keyframe indices: {keyframe_indices.cpu().numpy()}")
    print(f"Keyframe scores: {importance_scores[0][keyframe_indices].cpu().numpy()}")
    
    # Test unfreeze
    print("\n4️⃣ Testing ResNet unfreezing...")
    model.unfreeze_resnet(unfreeze_last_n_blocks=1)
    
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreezing: {trainable_after:,}")
    
    assert trainable_after > trainable_params, "Unfreezing didn't work!"
    print("✓ Unfreezing successful!")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - MODEL WORKS PERFECTLY!")
    print("="*60 + "\n")
    
    return model


if __name__ == "__main__":
    test_model()
