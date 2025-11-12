"""
Setup script for Google Colab environment.
Run this BEFORE training to configure environment variables.

Usage:
    python setup_colab_env.py
    
Or in Colab:
    !python setup_colab_env.py
"""

import os
import sys


def setup_environment():
    """Setup required environment variables for Google Colab."""
    
    print("=" * 60)
    print("  Setting up Google Colab Environment")
    print("=" * 60)
    
    # Set environment variables
    env_vars = {
        'DIR_LOGS': '/content/logs',
        'DIR_DATA': '/content/data',
        'TAG': 'colab-training',
    }
    
    print("\n📝 Setting environment variables...")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  ✓ {key} = {value}")
    
    # Create .env file
    print("\n📄 Creating .env file...")
    env_content = """DIR_LOGS=/content/logs
DIR_DATA=/content/data

# Optional: Uncomment if using Weights & Biases
# WANDB_PROJECT=stable-audio
# WANDB_ENTITY=your-username
# WANDB_API_KEY=your-api-key
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("  ✓ .env file created")
    
    # Create directories
    print("\n📁 Creating directories...")
    directories = [
        '/content/logs',
        '/content/data',
        '/content/logs/ckpts',
        '/content/logs/runs',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    
    # Verify setup
    print("\n✅ Environment setup complete!")
    print("\n" + "=" * 60)
    print("  Verification")
    print("=" * 60)
    
    print("\nEnvironment Variables:")
    for key in env_vars:
        value = os.environ.get(key, 'NOT SET')
        status = "✓" if value != 'NOT SET' else "✗"
        print(f"  {status} {key}: {value}")
    
    print("\nDirectories:")
    for directory in directories:
        exists = os.path.exists(directory)
        status = "✓" if exists else "✗"
        print(f"  {status} {directory}")
    
    print("\n.env File:")
    if os.path.exists('.env'):
        print("  ✓ .env file exists")
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    print(f"    {line.strip()}")
    else:
        print("  ✗ .env file not found")
    
    print("\n" + "=" * 60)
    print("  Next Steps")
    print("=" * 60)
    print("\n1. Upload your dataset to /content/data/")
    print("   !cp /path/to/train.tar /content/data/")
    print("   !cp /path/to/test.tar /content/data/")
    print("\n2. Login to Hugging Face:")
    print("   from huggingface_hub import notebook_login")
    print("   notebook_login()")
    print("\n3. Start training:")
    print("   !python train.py \\")
    print("       exp=train_colab_optimized \\")
    print("       datamodule.train_dataset.path=/content/data/train.tar \\")
    print("       datamodule.val_dataset.path=/content/data/test.tar")
    print("\n" + "=" * 60)


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        
        print("\n" + "=" * 60)
        print("  GPU Check")
        print("=" * 60)
        
        if torch.cuda.is_available():
            print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            
            # Recommend configuration based on VRAM
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print("\n📊 Recommended Configuration:")
            if vram_gb < 12:
                print("  ⚠️  Low VRAM detected (<12GB)")
                print("  Use: depth_factor=0.2, batch_size=1")
            elif vram_gb < 20:
                print("  ✓ Medium VRAM (12-20GB) - Good for training")
                print("  Use: depth_factor=0.3, batch_size=2")
            else:
                print("  ✓ High VRAM (>20GB) - Excellent for training")
                print("  Use: depth_factor=0.5, batch_size=4")
        else:
            print("\n✗ No GPU detected!")
            print("  Make sure GPU runtime is enabled:")
            print("  Runtime → Change runtime type → GPU")
            
    except ImportError:
        print("\n⚠️  PyTorch not installed")
        print("  Run: !pip install torch")


def check_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def main():
    """Main function."""
    
    # Check if running in Colab
    is_colab = check_colab()
    
    if is_colab:
        print("\n✓ Running in Google Colab")
    else:
        print("\n⚠️  Not running in Google Colab")
        print("  This script is optimized for Colab but will work locally too.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Setup environment
    setup_environment()
    
    # Check GPU
    check_gpu()
    
    print("\n✅ Setup complete! You're ready to train.")


if __name__ == "__main__":
    main()

