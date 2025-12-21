import torch
import torch.nn as nn
import torchvision.models as models
import math


class DualTemporalAttention(nn.Module):
    """
    Dual Temporal Attention: Combines local and global temporal attention
    for keyframe importance scoring.
    """
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.1):
        super(DualTemporalAttention, self).__init__()
        
        # Local attention (short-range dependencies)
        self.local_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Global attention (long-range dependencies)
        self.global_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, feature_dim)
        Returns:
            attended: (batch_size, num_frames, feature_dim)
        """
        # Local attention (attend to nearby frames)
        local_out, _ = self.local_attention(x, x, x)
        local_out = self.norm1(x + local_out)
        
        # Global attention (attend to all frames)
        global_out, _ = self.global_attention(x, x, x)
        global_out = self.norm2(x + global_out)
        
        # Fuse local and global information
        fused = torch.cat([local_out, global_out], dim=-1)
        attended = self.fusion(fused)
        
        return attended


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer to capture temporal order."""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, feature_dim)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """Transformer encoder for temporal modeling."""
    def __init__(self, feature_dim=512, num_layers=3, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, feature_dim)
        Returns:
            out: (batch_size, num_frames, feature_dim)
        """
        return self.transformer(x)


class KeyFrameDetector(nn.Module):
    """
    ResNet50 + Transformer + Dual Temporal Attention Model
    for keyframe detection on TVSum dataset.
    
    Architecture:
    1. ResNet50 (pretrained) - Feature extraction
    2. Positional Encoding - Temporal order encoding
    3. Transformer Encoder - Temporal modeling
    4. Dual Temporal Attention - Local/Global attention
    5. Importance Scorer - Frame importance prediction
    """
    def __init__(
        self,
        feature_dim=512,
        num_transformer_layers=3,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        freeze_resnet=True
    ):
        super(KeyFrameDetector, self).__init__()
        
        # 1. Feature extractor: ResNet50 (pretrained on ImageNet)
        resnet = models.resnet50(pretrained=True)
        # Remove last FC layer, use 2048-dim features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optionally freeze ResNet initially
        if freeze_resnet:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Project ResNet features (2048) to transformer dimension (512)
        self.feature_projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Positional encoding
        self.positional_encoding = PositionalEncoding(feature_dim, max_len=100)
        
        # 3. Transformer encoder
        self.transformer = TransformerEncoder(
            feature_dim=feature_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 4. Dual temporal attention
        self.dual_attention = DualTemporalAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 5. Importance scorer (outputs importance score for each frame)
        self.importance_scorer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )
        
    def forward(self, frames):
        """
        Args:
            frames: (batch_size, num_frames, 3, H, W) - Video frames
        Returns:
            importance_scores: (batch_size, num_frames) - Importance score per frame
        """
        batch_size, num_frames, C, H, W = frames.size()
        
        # Reshape to process all frames at once
        frames = frames.view(batch_size * num_frames, C, H, W)
        
        # 1. Extract features using ResNet50
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.feature_extractor.parameters())):
            features = self.feature_extractor(frames)  # (B*T, 2048, 1, 1)
        
        features = features.view(batch_size * num_frames, -1)  # (B*T, 2048)
        
        # Project to transformer dimension
        features = self.feature_projection(features)  # (B*T, 512)
        
        # Reshape back to sequence
        features = features.view(batch_size, num_frames, -1)  # (B, T, 512)
        
        # 2. Add positional encoding
        features = self.positional_encoding(features)
        
        # 3. Transformer encoding
        transformer_out = self.transformer(features)  # (B, T, 512)
        
        # 4. Dual temporal attention
        attended = self.dual_attention(transformer_out)  # (B, T, 512)
        
        # 5. Predict importance scores
        importance_scores = self.importance_scorer(attended)  # (B, T, 1)
        importance_scores = importance_scores.squeeze(-1)  # (B, T)
        
        return importance_scores
    
    def unfreeze_resnet(self, unfreeze_last_n_blocks=1):
        """
        Unfreeze last N blocks of ResNet for fine-tuning.
        ResNet50 has 4 main blocks: layer1, layer2, layer3, layer4
        """
        # First, freeze everything
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks
        layers = [self.feature_extractor[-2][i] for i in range(len(self.feature_extractor[-2]))]
        
        if unfreeze_last_n_blocks >= 1:
            # Unfreeze layer4 (last block)
            for param in self.feature_extractor[7].parameters():
                param.requires_grad = True
        
        if unfreeze_last_n_blocks >= 2:
            # Unfreeze layer3
            for param in self.feature_extractor[6].parameters():
                param.requires_grad = True
                
        print(f"✓ Unfroze last {unfreeze_last_n_blocks} ResNet blocks for fine-tuning")


def create_model(freeze_resnet=True):
    """
    Factory function to create the keyframe detection model.
    
    Args:
        freeze_resnet: If True, freeze ResNet initially for stage 1 training
    """
    model = KeyFrameDetector(
        feature_dim=512,
        num_transformer_layers=3,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        freeze_resnet=freeze_resnet
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_model(freeze_resnet=True)
    
    # Dummy input: 2 videos, 60 frames each, 224x224 RGB
    dummy_input = torch.randn(2, 60, 3, 224, 224)
    
    # Forward pass
    importance_scores = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output importance scores shape: {importance_scores.shape}")
    print(f"Sample importance scores: {importance_scores[0][:10]}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
