"""
System Requirements Checker for Stable Audio ControlNet
Checks if your system meets the minimum requirements for low-resource inference.

Usage:
    python check_system.py
"""

import sys
import subprocess
import platform


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_status(check_name, status, details=""):
    """Print a check status."""
    status_symbol = "✅" if status else "❌"
    print(f"{status_symbol} {check_name}")
    if details:
        print(f"   {details}")


def check_python_version():
    """Check Python version."""
    print_header("Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python version: {version_str}")
    
    is_valid = version.major == 3 and version.minor >= 9
    print_status(
        "Python 3.9+",
        is_valid,
        "✓ Compatible" if is_valid else "⚠ Python 3.9+ recommended"
    )
    return is_valid


def check_cuda():
    """Check CUDA availability."""
    print_header("CUDA and GPU")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"GPU: {gpu_name}")
            print(f"VRAM: {gpu_memory:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch Version: {torch.__version__}")
            
            # Check VRAM
            vram_ok = gpu_memory >= 3.5
            print_status(
                "VRAM >= 4GB",
                vram_ok,
                f"{'✓ Sufficient' if vram_ok else '⚠ May struggle with 4GB'}"
            )
            
            return True
        else:
            print("CUDA not available")
            print_status("CUDA", False, "⚠ Will run on CPU (very slow)")
            return False
            
    except ImportError:
        print("PyTorch not installed")
        print_status("PyTorch", False, "❌ Please install: pip install torch")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print_header("Dependencies")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("stable_audio_tools", "Stable Audio Tools"),
        ("hydra", "Hydra"),
        ("pytorch_lightning", "PyTorch Lightning"),
    ]
    
    all_installed = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print_status(name, True, "✓ Installed")
        except ImportError:
            print_status(name, False, "❌ Not installed")
            all_installed = False
    
    if not all_installed:
        print("\n⚠ Install missing packages:")
        print("   pip install -r requirements.txt")
    
    return all_installed


def check_huggingface():
    """Check Hugging Face CLI login."""
    print_header("Hugging Face")
    
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        
        if token:
            print_status("Hugging Face Login", True, "✓ Logged in")
            return True
        else:
            print_status("Hugging Face Login", False, "⚠ Not logged in")
            print("\n   Login with: huggingface-cli login")
            return False
            
    except ImportError:
        print_status("Hugging Face Hub", False, "❌ Not installed")
        return False


def check_disk_space():
    """Check available disk space."""
    print_header("Disk Space")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / 1e9
        
        print(f"Free space: {free_gb:.2f} GB")
        
        space_ok = free_gb >= 10
        print_status(
            "Disk Space >= 10GB",
            space_ok,
            f"{'✓ Sufficient' if space_ok else '⚠ May need more space'}"
        )
        
        return space_ok
        
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return None


def check_nvidia_driver():
    """Check NVIDIA driver version."""
    print_header("NVIDIA Driver")
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            print(f"Driver version: {driver_version}")
            print_status("NVIDIA Driver", True, "✓ Installed")
            return True
        else:
            print_status("NVIDIA Driver", False, "⚠ nvidia-smi failed")
            return False
            
    except FileNotFoundError:
        print_status("NVIDIA Driver", False, "❌ nvidia-smi not found")
        print("   Install NVIDIA drivers from: https://www.nvidia.com/drivers")
        return False
    except Exception as e:
        print(f"Could not check driver: {e}")
        return None


def get_system_info():
    """Get basic system information."""
    print_header("System Information")
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f"RAM: {ram_gb:.2f} GB")
        
        ram_ok = ram_gb >= 8
        print_status(
            "RAM >= 8GB",
            ram_ok,
            f"{'✓ Sufficient' if ram_ok else '⚠ 8GB+ recommended'}"
        )
    except ImportError:
        print("RAM: Unknown (psutil not installed)")


def print_recommendations(checks):
    """Print recommendations based on checks."""
    print_header("Recommendations")
    
    python_ok, cuda_ok, deps_ok, hf_ok, space_ok, driver_ok = checks
    
    if all([python_ok, cuda_ok, deps_ok, hf_ok]):
        print("\n🎉 Your system is ready for inference!")
        print("\nNext steps:")
        print("1. Run: python inference_low_resource.py --input_audio your_audio.wav")
        print("2. Or open: notebook/inference_low_resource.ipynb")
        print("3. See: QUICK_START_LOW_RESOURCE.md for detailed guide")
        
    else:
        print("\n⚠ Some requirements are not met. Please fix the following:\n")
        
        if not python_ok:
            print("❌ Python: Install Python 3.9+ from https://www.python.org/")
        
        if not deps_ok:
            print("❌ Dependencies: Run 'pip install -r requirements.txt'")
        
        if not cuda_ok:
            print("❌ CUDA: Install PyTorch with CUDA support")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        if not driver_ok:
            print("❌ NVIDIA Driver: Install from https://www.nvidia.com/drivers")
        
        if not hf_ok:
            print("❌ Hugging Face: Run 'huggingface-cli login'")
        
        if not space_ok:
            print("⚠ Disk Space: Free up at least 10GB of space")
    
    print("\n" + "=" * 60)


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("  Stable Audio ControlNet - System Requirements Checker")
    print("=" * 60)
    print("\nChecking your system for compatibility...")
    
    # Run all checks
    get_system_info()
    python_ok = check_python_version()
    driver_ok = check_nvidia_driver()
    cuda_ok = check_cuda()
    deps_ok = check_dependencies()
    hf_ok = check_huggingface()
    space_ok = check_disk_space()
    
    # Print recommendations
    checks = (python_ok, cuda_ok, deps_ok, hf_ok, space_ok, driver_ok)
    print_recommendations(checks)


if __name__ == "__main__":
    main()

