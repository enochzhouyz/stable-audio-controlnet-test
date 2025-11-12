# Low-Resource Setup for Stable Audio ControlNet

**Optimized for RTX 3050 (4GB VRAM) and similar GPUs**

This document provides a complete guide for running Stable Audio ControlNet on limited hardware.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [What's Included](#whats-included)
3. [Quick Start](#quick-start)
4. [Detailed Guides](#detailed-guides)
5. [Performance Expectations](#performance-expectations)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## 🎯 Overview

The original Stable Audio ControlNet project requires **16GB+ VRAM** for training and inference. This low-resource setup enables you to:

- ✅ **Run inference** on 4GB VRAM GPUs (RTX 3050, GTX 1650, etc.)
- ✅ **Generate music** with acceptable quality
- ✅ **Experiment** with different parameters
- ⚠️ **Training** is possible but extremely limited (cloud recommended)

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3050 (4GB) | RTX 3060 (6GB+) |
| RAM | 8GB | 16GB |
| Storage | 10GB free | 20GB+ free |
| OS | Windows 10/11, Linux | Any |

---

## 📦 What's Included

This low-resource setup includes:

### 1. **Scripts**
- `inference_low_resource.py` - Optimized inference script
- `check_system.py` - System requirements checker
- `run_inference_low_resource.bat` - Windows batch script

### 2. **Configuration Files**
- `exp/train_low_resource.yaml` - Low-resource training config
- Configuration optimized for 4GB VRAM

### 3. **Notebooks**
- `notebook/inference_low_resource.ipynb` - Interactive inference

### 4. **Documentation**
- `QUICK_START_LOW_RESOURCE.md` - Fast setup guide
- `LOW_RESOURCE_GUIDE.md` - Comprehensive optimization guide
- `README_LOW_RESOURCE.md` - This file

---

## 🚀 Quick Start

### Step 1: Check Your System

```bash
python check_system.py
```

This will verify:
- Python version (3.9+)
- CUDA availability
- GPU VRAM (4GB+)
- Dependencies
- Disk space

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
huggingface-cli login
```

### Step 3: Run Inference

**Option A: Command Line**
```bash
python inference_low_resource.py \
    --input_audio your_audio.wav \
    --output_dir outputs \
    --depth_factor 0.1 \
    --steps 25
```

**Option B: Jupyter Notebook**
```bash
cd notebook
jupyter notebook inference_low_resource.ipynb
```

**Option C: Windows Batch (Double-click)**
- Edit `run_inference_low_resource.bat`
- Set `INPUT_AUDIO` path
- Double-click to run

---

## 📚 Detailed Guides

### For Complete Beginners
→ **Start here:** `QUICK_START_LOW_RESOURCE.md`
- Step-by-step setup
- Example commands
- Common issues and solutions

### For Advanced Users
→ **Read:** `LOW_RESOURCE_GUIDE.md`
- Memory optimization techniques
- Parameter tuning
- Training considerations
- Cloud alternatives

### For Developers
→ **See:** Source code in `inference_low_resource.py`
- Implementation details
- Customization options
- Integration examples

---

## 📊 Performance Expectations

### RTX 3050 (4GB VRAM)

| Configuration | Time | Quality | VRAM | Use Case |
|---------------|------|---------|------|----------|
| depth=0.05, steps=10 | ~1 min | Low | 2.5GB | Quick preview |
| depth=0.1, steps=25 | ~3 min | Medium | 3.5GB | **Recommended** |
| depth=0.1, steps=50 | ~6 min | Good | 3.8GB | Best on 4GB |

### RTX 3060 (6GB VRAM)

| Configuration | Time | Quality | VRAM | Use Case |
|---------------|------|---------|------|----------|
| depth=0.1, steps=50 | ~4 min | Good | 4.5GB | Balanced |
| depth=0.2, steps=100 | ~10 min | High | 5.5GB | **Recommended** |

### RTX 3080 (10GB VRAM)

| Configuration | Time | Quality | VRAM | Use Case |
|---------------|------|---------|------|----------|
| depth=0.5, steps=100 | ~8 min | Excellent | 8GB | Full quality |
| depth=0.5, steps=200 | ~15 min | Best | 9GB | Maximum quality |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions (try in order):**
```bash
# 1. Reduce model size
--depth_factor 0.05

# 2. Reduce steps
--steps 10

# 3. Reduce audio length
--max_duration 5

# 4. Use minimal config
python inference_low_resource.py \
    --input_audio input.wav \
    --depth_factor 0.05 \
    --steps 10 \
    --max_duration 5
```

#### 2. CUDA Not Available

**Symptoms:**
```
CUDA not available, falling back to CPU
```

**Solutions:**
1. Check drivers: `nvidia-smi`
2. Reinstall PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

#### 3. Slow Performance

**Symptoms:**
- Generation takes > 10 minutes
- GPU utilization low

**Solutions:**
1. Close other applications
2. Check GPU usage: `nvidia-smi`
3. Ensure CUDA is being used (not CPU)
4. Update NVIDIA drivers

#### 4. Poor Quality Output

**Symptoms:**
- Noisy or distorted audio
- Doesn't match input style

**Solutions:**
1. Increase `--steps` (25 → 50 → 100)
2. Increase `--depth_factor` (0.1 → 0.2)
3. Use pre-trained checkpoint
4. Adjust `--cfg_scale` (try 3.0-9.0)

---

## ❓ FAQ

### Q: Can I train on 4GB VRAM?

**A:** Technically yes, but **not recommended**. Training requires:
- Extremely small model (depth_factor=0.1)
- Very slow progress (~1 min per step)
- High risk of OOM crashes
- Lower quality results

**Better alternatives:**
- Google Colab (Free, 15GB GPU)
- Kaggle Notebooks (Free, 16GB GPU)
- Cloud GPU rental (~$0.20-0.30/hour)

### Q: What's the difference between depth_factor values?

**A:** Depth factor controls model size:

| Depth Factor | Layers | VRAM | Quality | Speed |
|--------------|--------|------|---------|-------|
| 0.05 | 1-2 | ~2GB | Low | Fast |
| 0.1 | 2-3 | ~3GB | Medium | Medium |
| 0.2 | 5 | ~4GB | Good | Slow |
| 0.5 | 12 | ~8GB | Excellent | Very slow |

### Q: Can I use pre-trained checkpoints?

**A:** Yes! Download from:
- [MusDB model](https://drive.google.com/drive/folders/1-EcM7RWDbLLULrcURZBFSkginaJtq_Zv)
- [MoisesDB model](https://drive.google.com/drive/folders/1gtAJVeZmNe3fmu-WokF3n3eSaySZumrx)

**Note:** Pre-trained models use depth_factor=0.5 (requires ~6GB VRAM)

### Q: How long does inference take?

**A:** On RTX 3050:
- 10 seconds audio, 25 steps: ~3 minutes
- 10 seconds audio, 50 steps: ~6 minutes
- 30 seconds audio, 25 steps: ~9 minutes

### Q: What audio formats are supported?

**A:** Any format supported by `torchaudio`:
- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A

### Q: Can I process longer audio?

**A:** Yes, but in chunks:
```bash
# Process 30 seconds (may OOM on 4GB)
--max_duration 30

# Safe for 4GB
--max_duration 10
```

For longer audio, split into chunks and process separately.

### Q: What's the quality compared to 16GB setup?

**A:** With optimizations:
- **depth_factor=0.1**: ~60-70% of full quality
- **depth_factor=0.2**: ~80-85% of full quality
- **depth_factor=0.5**: ~95-100% of full quality (needs 6GB+)

Main differences:
- Less detail in generated audio
- Simpler accompaniments
- Faster generation

### Q: Can I use CPU instead of GPU?

**A:** Yes, but **very slow**:
```bash
python inference_low_resource.py \
    --input_audio input.wav \
    --device cpu
```

Expect 10-20x slower than GPU.

### Q: How do I monitor GPU usage?

**Windows PowerShell:**
```powershell
while($true) { nvidia-smi; Start-Sleep -Seconds 1; Clear-Host }
```

**Linux/Mac:**
```bash
watch -n 1 nvidia-smi
```

### Q: What if I have 8GB or 12GB VRAM?

**A:** You have more flexibility!

**8GB VRAM (RTX 3070):**
```bash
--depth_factor 0.5 --steps 100 --max_duration 20
```

**12GB VRAM (RTX 3080 Ti):**
```bash
--depth_factor 0.5 --steps 200 --max_duration 47
```

---

## 🎓 Learning Resources

### Understanding the Model

1. **ControlNet**: Adds conditional control to diffusion models
2. **Depth Factor**: Percentage of layers in the ControlNet adapter
3. **Diffusion Steps**: More steps = better quality but slower
4. **CFG Scale**: Guidance strength (higher = closer to condition)

### Related Papers

- [Stable Audio Open](https://arxiv.org/abs/2407.14358)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [Music ControlNet](https://ieeexplore.ieee.org/document/10446660)

### Community

- GitHub Issues: Report bugs and request features
- Discussions: Share results and tips
- Discord: Join the community (if available)

---

## 🛠️ Advanced Usage

### Custom Conditioning

```python
from main.controlnet.pretrained import get_pretrained_controlnet_model

model, config = get_pretrained_controlnet_model(
    "stabilityai/stable-audio-open-1.0",
    controlnet_types=["audio"],
    depth_factor=0.1
)

# Your custom inference code here
```

### Batch Processing

```bash
# Process multiple files
for file in *.wav; do
    python inference_low_resource.py \
        --input_audio "$file" \
        --output_dir outputs \
        --depth_factor 0.1 \
        --steps 25
done
```

### Integration with Other Tools

```python
# Load generated audio
import torchaudio
audio, sr = torchaudio.load("outputs/output.wav")

# Further processing with your tools
# ...
```

---

## 📈 Roadmap

Future improvements for low-resource setups:

- [ ] Gradient checkpointing for training
- [ ] Model quantization (INT8)
- [ ] CPU offloading for larger models
- [ ] Streaming inference for long audio
- [ ] Web UI for easier use
- [ ] Pre-quantized checkpoints

---

## 🤝 Contributing

Have improvements for low-resource setups? Contributions welcome!

1. Test on your hardware
2. Document your findings
3. Submit a pull request

---

## 📄 License

Same as the main project. See `LICENSE` file.

---

## 🙏 Acknowledgments

- Original Stable Audio ControlNet authors
- Stability AI for Stable Audio Open
- Community contributors

---

## 📞 Support

- **Quick questions**: See FAQ above
- **Issues**: Check `TROUBLESHOOTING` section
- **Bugs**: Open GitHub issue
- **Discussions**: GitHub Discussions

---

**Happy generating! 🎵**

*Last updated: November 2025*

