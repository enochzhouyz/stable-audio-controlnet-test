"""
Setup script for BIOT integration with EEG-to-Music pipeline.

This script helps:
1. Check if BIOT is properly installed
2. Verify pre-trained model paths
3. Test BIOT encoder with sample data
4. Add BIOT to Python path
"""

import sys
import os
from pathlib import Path
import torch


def add_biot_to_path():
    """Add BIOT directory to Python path."""
    biot_path = Path(__file__).parent.parent / "Sample Code BIOT" / "BIOT"
    if biot_path.exists():
        sys.path.insert(0, str(biot_path))
        print(f"✓ Added BIOT to Python path: {biot_path}")
        return True
    else:
        print(f"✗ BIOT directory not found: {biot_path}")
        return False


def check_biot_import():
    """Check if BIOT can be imported."""
    try:
        from model.biot import BIOTEncoder, BIOTClassifier
        print("✓ Successfully imported BIOT modules")
        return True
    except ImportError as e:
        print(f"✗ Failed to import BIOT: {e}")
        return False


def check_pretrained_models():
    """Check if pre-trained models exist."""
    pretrained_dir = Path(__file__).parent.parent / "Sample Code BIOT" / "BIOT" / "pretrained-models"
    
    if not pretrained_dir.exists():
        print(f"✗ Pre-trained models directory not found: {pretrained_dir}")
        return False
    
    models = [
        "EEG-PREST-16-channels.ckpt",
        "EEG-SHHS+PREST-18-channels.ckpt",
        "EEG-six-datasets-18-channels.ckpt",
    ]
    
    print(f"\nChecking pre-trained models in {pretrained_dir}:")
    found_models = []
    for model in models:
        model_path = pretrained_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {model} ({size_mb:.1f} MB)")
            found_models.append(str(model_path))
        else:
            print(f"  ✗ {model} (not found)")
    
    return found_models


def test_biot_encoder():
    """Test BIOT encoder with sample data."""
    try:
        from model.biot import BIOTEncoder
        
        print("\nTesting BIOT encoder...")
        
        # Create encoder
        encoder = BIOTEncoder(
            emb_size=256,
            heads=8,
            depth=4,
            n_channels=32,  # DEAP has 32 channels
            n_fft=200,
            hop_length=100,
        )
        
        # Test with random EEG data
        batch_size = 2
        num_channels = 32
        time_steps = 1280  # ~10 seconds at 128Hz
        
        x = torch.randn(batch_size, num_channels, time_steps)
        print(f"  Input shape: {x.shape}")
        
        with torch.no_grad():
            output = encoder(x)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({batch_size}, 256)")
        
        if output.shape == (batch_size, 256):
            print("✓ BIOT encoder test passed!")
            return True
        else:
            print("✗ BIOT encoder output shape mismatch")
            return False
            
    except Exception as e:
        print(f"✗ BIOT encoder test failed: {e}")
        return False


def test_eeg_projectors():
    """Test all three EEG projector types."""
    try:
        from main.eeg_encoders import create_eeg_projector
        
        print("\nTesting EEG projectors...")
        
        num_channels = 32
        target_length = 441000  # 10s at 44.1kHz
        batch_size = 2
        time_steps = 1280
        
        x = torch.randn(batch_size, num_channels, time_steps)
        
        # Test Simple projector
        print("\n  1. Testing Simple Projector:")
        simple_proj = create_eeg_projector(
            projector_type="simple",
            num_eeg_channels=num_channels,
            target_length=target_length,
            hidden_dim=128,
        )
        with torch.no_grad():
            out_simple = simple_proj(x)
        print(f"     Input: {x.shape} → Output: {out_simple.shape}")
        assert out_simple.shape == (batch_size, 2, target_length), "Shape mismatch!"
        print("     ✓ Simple projector works!")
        
        # Test BIOT projector (without pre-trained weights)
        print("\n  2. Testing BIOT Projector (no pre-training):")
        biot_proj = create_eeg_projector(
            projector_type="biot",
            num_eeg_channels=num_channels,
            target_length=target_length,
            biot_pretrained_path=None,
            freeze_biot=False,
        )
        with torch.no_grad():
            out_biot = biot_proj(x)
        print(f"     Input: {x.shape} → Output: {out_biot.shape}")
        assert out_biot.shape == (batch_size, 2, target_length), "Shape mismatch!"
        print("     ✓ BIOT projector works!")
        
        # Test Hybrid projector
        print("\n  3. Testing Hybrid Projector (no pre-training):")
        hybrid_proj = create_eeg_projector(
            projector_type="hybrid",
            num_eeg_channels=num_channels,
            target_length=target_length,
            biot_pretrained_path=None,
            freeze_biot=False,
        )
        with torch.no_grad():
            out_hybrid = hybrid_proj(x)
        print(f"     Input: {x.shape} → Output: {out_hybrid.shape}")
        assert out_hybrid.shape == (batch_size, 2, target_length), "Shape mismatch!"
        print("     ✓ Hybrid projector works!")
        
        print("\n✓ All EEG projectors test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ EEG projectors test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pretrained_loading():
    """Test loading pre-trained BIOT weights."""
    try:
        from main.eeg_encoders import BIOTEEGProjector
        
        pretrained_dir = Path(__file__).parent.parent / "Sample Code BIOT" / "BIOT" / "pretrained-models"
        model_path = pretrained_dir / "EEG-PREST-16-channels.ckpt"
        
        if not model_path.exists():
            print("\n⚠ Skipping pre-trained loading test (model not found)")
            return None
        
        print(f"\nTesting pre-trained weight loading from {model_path}...")
        
        projector = BIOTEEGProjector(
            num_eeg_channels=32,
            target_length=441000,
            biot_pretrained_path=str(model_path),
            freeze_biot=True,
        )
        
        # Test forward pass
        x = torch.randn(2, 32, 1280)
        with torch.no_grad():
            output = projector(x)
        
        print(f"  Input: {x.shape} → Output: {output.shape}")
        print("✓ Pre-trained weight loading test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Pre-trained weight loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all setup checks."""
    print("=" * 60)
    print("BIOT Integration Setup & Verification")
    print("=" * 60)
    
    results = {}
    
    # Check 1: Add BIOT to path
    print("\n[1/6] Adding BIOT to Python path...")
    results['path'] = add_biot_to_path()
    
    # Check 2: Import BIOT
    print("\n[2/6] Checking BIOT imports...")
    results['import'] = check_biot_import()
    
    # Check 3: Pre-trained models
    print("\n[3/6] Checking pre-trained models...")
    found_models = check_pretrained_models()
    results['pretrained'] = len(found_models) > 0
    
    # Check 4: Test BIOT encoder
    if results['import']:
        print("\n[4/6] Testing BIOT encoder...")
        results['encoder'] = test_biot_encoder()
    else:
        print("\n[4/6] Skipping encoder test (import failed)")
        results['encoder'] = False
    
    # Check 5: Test EEG projectors
    if results['import']:
        print("\n[5/6] Testing EEG projectors...")
        results['projectors'] = test_eeg_projectors()
    else:
        print("\n[5/6] Skipping projectors test (import failed)")
        results['projectors'] = False
    
    # Check 6: Test pre-trained loading
    if results['import'] and results['pretrained']:
        print("\n[6/6] Testing pre-trained weight loading...")
        results['pretrained_load'] = test_pretrained_loading()
    else:
        print("\n[6/6] Skipping pre-trained loading test")
        results['pretrained_load'] = None
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    for check, status in results.items():
        if status is True:
            symbol = "✓"
            status_text = "PASSED"
        elif status is False:
            symbol = "✗"
            status_text = "FAILED"
        else:
            symbol = "⚠"
            status_text = "SKIPPED"
        
        print(f"{symbol} {check.ljust(20)}: {status_text}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if not results['import']:
        print("\n⚠ BIOT import failed. Please:")
        print("  1. Install dependencies: cd 'Sample Code BIOT/BIOT' && pip install -r requirements.txt")
        print("  2. Ensure BIOT directory structure is correct")
    
    if not results['pretrained']:
        print("\n⚠ No pre-trained models found. You can:")
        print("  1. Download from BIOT GitHub: https://github.com/ycq091044/BIOT")
        print("  2. Or train from scratch (set biot_pretrained_path: null)")
    
    if all(v in [True, None] for v in results.values()):
        print("\n✓ Setup complete! You're ready to train with BIOT.")
        print("\nQuick start:")
        print("  python train.py exp=train_deap_controlnet_eeg_hybrid")
    else:
        print("\n⚠ Some checks failed. Please resolve issues above before training.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

