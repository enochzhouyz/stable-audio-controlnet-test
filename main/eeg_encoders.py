"""
EEG Encoders for Audio Generation ControlNet

This module provides different EEG encoding strategies:
1. SimpleEEGProjector: Original lightweight Conv1d projector
2. BIOTEEGProjector: Uses pre-trained BIOT encoder for rich EEG representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Literal


class SimpleEEGProjector(nn.Module):
    """
    Original simple EEG projector using Conv1d layers.
    Maps EEG activations to an audio-like control signal.
    """

    def __init__(self, num_eeg_channels: int, hidden_dim: int, target_length: int) -> None:
        super().__init__()
        self.target_length = target_length
        self.net = nn.Sequential(
            nn.Conv1d(num_eeg_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 2, kernel_size=1),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: [batch_size, num_channels, time_steps]
        Returns:
            control_signal: [batch_size, 2, target_length]
        """
        # Normalize per-channel
        eeg = eeg - eeg.mean(dim=-1, keepdim=True)
        eeg = eeg / (eeg.std(dim=-1, keepdim=True) + 1e-6)
        
        # Resample to target length
        eeg_resampled = F.interpolate(
            eeg,
            size=self.target_length,
            mode="linear",
            align_corners=False,
        )
        return torch.tanh(self.net(eeg_resampled))


class BIOTEEGProjector(nn.Module):
    """
    BIOT-based EEG projector that uses pre-trained BIOT encoder.
    
    Architecture:
        EEG → BIOT Encoder (STFT + Transformer) → [256-dim embedding]
        → Projection Network → [2-channel audio-like control signal]
    
    Args:
        num_eeg_channels: Number of EEG channels in input
        target_length: Target length of output control signal
        biot_pretrained_path: Path to pre-trained BIOT weights (optional)
        freeze_biot: Whether to freeze BIOT encoder weights
        biot_config: Configuration for BIOT encoder
    """

    def __init__(
        self,
        num_eeg_channels: int,
        target_length: int,
        biot_pretrained_path: Optional[str] = None,
        freeze_biot: bool = True,
        biot_emb_size: int = 256,
        biot_heads: int = 8,
        biot_depth: int = 4,
        biot_n_fft: int = 200,
        biot_hop_length: int = 100,
        projection_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.freeze_biot = freeze_biot
        self.num_eeg_channels = num_eeg_channels
        
        # Import BIOT encoder (assuming BIOT code is accessible)
        try:
            from model.biot import BIOTEncoder
        except ImportError:
            raise ImportError(
                "BIOT model not found. Please ensure the BIOT code is in the Python path. "
                "You may need to add 'Sample Code BIOT/BIOT' to sys.path or install it as a package."
            )
        
        # Initialize BIOT encoder
        self.biot_encoder = BIOTEncoder(
            emb_size=biot_emb_size,
            heads=biot_heads,
            depth=biot_depth,
            n_channels=num_eeg_channels,
            n_fft=biot_n_fft,
            hop_length=biot_hop_length,
        )
        
        # Load pre-trained weights if provided
        if biot_pretrained_path is not None:
            self._load_pretrained_weights(biot_pretrained_path)
        
        # Freeze BIOT encoder if specified
        if freeze_biot:
            for param in self.biot_encoder.parameters():
                param.requires_grad = False
            self.biot_encoder.eval()
        
        # Projection network: [256-dim embedding] → [2-channel × target_length signal]
        # We use a more sophisticated projection to preserve temporal structure
        self.projection_net = nn.Sequential(
            nn.Linear(biot_emb_size, projection_hidden_dim),
            nn.LayerNorm(projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_hidden_dim, projection_hidden_dim),
            nn.LayerNorm(projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_hidden_dim, 2 * target_length),
        )
        
    def _load_pretrained_weights(self, pretrained_path: str) -> None:
        """Load pre-trained BIOT weights."""
        path = Path(pretrained_path)
        if not path.exists():
            raise FileNotFoundError(f"Pre-trained BIOT model not found at {pretrained_path}")
        
        print(f"Loading pre-trained BIOT weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # Load weights (may have partial match if channel counts differ)
        try:
            self.biot_encoder.load_state_dict(state_dict, strict=False)
            print("✓ Successfully loaded pre-trained BIOT weights")
        except Exception as e:
            print(f"⚠ Warning: Could not load all weights. Error: {e}")
            print("  Continuing with partially loaded weights...")

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: [batch_size, num_channels, time_steps]
        Returns:
            control_signal: [batch_size, 2, target_length]
        """
        # BIOT encoder expects [batch_size, channel, time_steps]
        # and outputs [batch_size, emb_size]
        
        # Set eval mode if frozen
        if self.freeze_biot and self.training:
            self.biot_encoder.eval()
        
        # Extract BIOT features
        with torch.set_grad_enabled(not self.freeze_biot):
            biot_features = self.biot_encoder(eeg)  # [batch_size, 256]
        
        # Project to 2-channel control signal
        projected = self.projection_net(biot_features)  # [batch_size, 2 * target_length]
        
        # Reshape to [batch_size, 2, target_length]
        batch_size = eeg.shape[0]
        control_signal = projected.view(batch_size, 2, self.target_length)
        
        # Apply tanh to keep in [-1, 1] range
        return torch.tanh(control_signal)


class HybridBIOTEEGProjector(nn.Module):
    """
    Hybrid approach combining BIOT features with temporal EEG information.
    
    This version:
    1. Extracts global features using BIOT encoder (semantic understanding)
    2. Extracts local temporal features using Conv1d (time-aligned details)
    3. Fuses both to create rich control signal
    """

    def __init__(
        self,
        num_eeg_channels: int,
        target_length: int,
        biot_pretrained_path: Optional[str] = None,
        freeze_biot: bool = True,
        biot_emb_size: int = 256,
        biot_heads: int = 8,
        biot_depth: int = 4,
        biot_n_fft: int = 200,
        biot_hop_length: int = 100,
        local_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.freeze_biot = freeze_biot
        
        # Import BIOT encoder
        try:
            from model.biot import BIOTEncoder
        except ImportError:
            raise ImportError(
                "BIOT model not found. Please ensure the BIOT code is in the Python path."
            )
        
        # Global feature extractor (BIOT)
        self.biot_encoder = BIOTEncoder(
            emb_size=biot_emb_size,
            heads=biot_heads,
            depth=biot_depth,
            n_channels=num_eeg_channels,
            n_fft=biot_n_fft,
            hop_length=biot_hop_length,
        )
        
        if biot_pretrained_path is not None:
            self._load_pretrained_weights(biot_pretrained_path)
        
        if freeze_biot:
            for param in self.biot_encoder.parameters():
                param.requires_grad = False
            self.biot_encoder.eval()
        
        # Local temporal feature extractor (Conv1d)
        self.local_feature_net = nn.Sequential(
            nn.Conv1d(num_eeg_channels, local_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(local_hidden_dim),
            nn.GELU(),
            nn.Conv1d(local_hidden_dim, local_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(local_hidden_dim),
            nn.GELU(),
        )
        
        # Global feature projection (broadcast to temporal dimension)
        self.global_projection = nn.Sequential(
            nn.Linear(biot_emb_size, local_hidden_dim),
            nn.GELU(),
        )
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Conv1d(local_hidden_dim * 2, local_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(local_hidden_dim, 2, kernel_size=1),
        )
    
    def _load_pretrained_weights(self, pretrained_path: str) -> None:
        """Load pre-trained BIOT weights."""
        path = Path(pretrained_path)
        if not path.exists():
            raise FileNotFoundError(f"Pre-trained BIOT model not found at {pretrained_path}")
        
        print(f"Loading pre-trained BIOT weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu")
        
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        try:
            self.biot_encoder.load_state_dict(state_dict, strict=False)
            print("✓ Successfully loaded pre-trained BIOT weights")
        except Exception as e:
            print(f"⚠ Warning: Could not load all weights. Error: {e}")

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: [batch_size, num_channels, time_steps]
        Returns:
            control_signal: [batch_size, 2, target_length]
        """
        batch_size = eeg.shape[0]
        
        # Normalize EEG
        eeg_norm = eeg - eeg.mean(dim=-1, keepdim=True)
        eeg_norm = eeg_norm / (eeg_norm.std(dim=-1, keepdim=True) + 1e-6)
        
        # Resample to target length for local features
        eeg_resampled = F.interpolate(
            eeg_norm,
            size=self.target_length,
            mode="linear",
            align_corners=False,
        )
        
        # Extract global features with BIOT
        if self.freeze_biot and self.training:
            self.biot_encoder.eval()
        
        with torch.set_grad_enabled(not self.freeze_biot):
            global_features = self.biot_encoder(eeg)  # [batch_size, 256]
        
        # Project global features and broadcast to temporal dimension
        global_projected = self.global_projection(global_features)  # [batch_size, hidden_dim]
        global_broadcast = global_projected.unsqueeze(-1).expand(-1, -1, self.target_length)
        # [batch_size, hidden_dim, target_length]
        
        # Extract local temporal features
        local_features = self.local_feature_net(eeg_resampled)  # [batch_size, hidden_dim, target_length]
        
        # Fuse global and local features
        fused = torch.cat([global_broadcast, local_features], dim=1)  # [batch_size, 2*hidden_dim, target_length]
        control_signal = self.fusion_net(fused)  # [batch_size, 2, target_length]
        
        return torch.tanh(control_signal)


def create_eeg_projector(
    projector_type: Literal["simple", "biot", "hybrid"],
    num_eeg_channels: int,
    target_length: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create EEG projectors.
    
    Args:
        projector_type: Type of projector ("simple", "biot", or "hybrid")
        num_eeg_channels: Number of EEG input channels
        target_length: Target length of control signal
        **kwargs: Additional arguments for specific projector types
        
    Returns:
        EEG projector module
    """
    if projector_type == "simple":
        hidden_dim = kwargs.get("hidden_dim", 128)
        return SimpleEEGProjector(
            num_eeg_channels=num_eeg_channels,
            hidden_dim=hidden_dim,
            target_length=target_length,
        )
    
    elif projector_type == "biot":
        return BIOTEEGProjector(
            num_eeg_channels=num_eeg_channels,
            target_length=target_length,
            biot_pretrained_path=kwargs.get("biot_pretrained_path", None),
            freeze_biot=kwargs.get("freeze_biot", True),
            biot_emb_size=kwargs.get("biot_emb_size", 256),
            biot_heads=kwargs.get("biot_heads", 8),
            biot_depth=kwargs.get("biot_depth", 4),
            biot_n_fft=kwargs.get("biot_n_fft", 200),
            biot_hop_length=kwargs.get("biot_hop_length", 100),
            projection_hidden_dim=kwargs.get("projection_hidden_dim", 512),
        )
    
    elif projector_type == "hybrid":
        return HybridBIOTEEGProjector(
            num_eeg_channels=num_eeg_channels,
            target_length=target_length,
            biot_pretrained_path=kwargs.get("biot_pretrained_path", None),
            freeze_biot=kwargs.get("freeze_biot", True),
            biot_emb_size=kwargs.get("biot_emb_size", 256),
            biot_heads=kwargs.get("biot_heads", 8),
            biot_depth=kwargs.get("biot_depth", 4),
            biot_n_fft=kwargs.get("biot_n_fft", 200),
            biot_hop_length=kwargs.get("biot_hop_length", 100),
            local_hidden_dim=kwargs.get("local_hidden_dim", 128),
        )
    
    else:
        raise ValueError(f"Unknown projector_type: {projector_type}")

