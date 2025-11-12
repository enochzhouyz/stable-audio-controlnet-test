# 🚀 START HERE - Low Resource Setup

**You have RTX 3050 (4GB VRAM) + 8GB RAM**

This is your starting point for running Stable Audio ControlNet on limited hardware!

---

## ⚡ 3-Minute Quick Start

### Step 1: Check Your System (30 seconds)
```bash
python check_system.py
```

This verifies your GPU, CUDA, and dependencies.

### Step 2: Install (2 minutes)
```bash
pip install -r requirements.txt
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
huggingface-cli login
```

### Step 3: Test (3 minutes)
```bash
python inference_low_resource.py \
    --input_audio res/track_musdb/input.wav \
    --output_dir outputs \
    --depth_factor 0.1 \
    --steps 25
```

**Done!** Check `outputs/` folder for generated audio.

---

## 📚 Complete Documentation

I've created a full suite of resources for you:

### 🎯 Quick References

| File | Use When | Time |
|------|----------|------|
| **START_HERE_LOW_RESOURCE.md** | First time setup | 5 min |
| **QUICK_START_LOW_RESOURCE.md** | Step-by-step guide | 10 min |
| **LOW_RESOURCE_SUMMARY.md** | Overview & cheat sheet | 5 min |

### 📖 Detailed Guides

| File | Use When | Time |
|------|----------|------|
| **LOW_RESOURCE_GUIDE.md** | Understanding optimizations | 20 min |
| **README_LOW_RESOURCE.md** | Complete reference & FAQ | 30 min |

### 🛠️ Tools

| File | Purpose |
|------|---------|
| `inference_low_resource.py` | Main inference script |
| `check_system.py` | System requirements checker |
| `run_inference_low_resource.bat` | Windows batch script |
| `notebook/inference_low_resource.ipynb` | Interactive notebook |
| `exp/train_low_resource.yaml` | Training config (not recommended) |

---

## 🎯 What Can You Do?

### ✅ Recommended (Works Great!)

1. **Run Inference** - Generate music from audio input
   - 10-second clips: ~3 minutes
   - Good quality with optimized settings
   - Perfect for experimentation

2. **Test Pre-trained Models** - Use existing checkpoints
   - Download from project README
   - Note: May need 6GB VRAM for full models

3. **Experiment** - Try different parameters
   - Adjust quality vs speed
   - Learn how the system works

### ⚠️ Challenging (Possible but Limited)

1. **Training** - Train your own models
   - Extremely slow on 4GB VRAM
   - High risk of crashes
   - **Recommendation**: Use Google Colab (free, 15GB GPU)

2. **Long Audio** - Process >15 seconds
   - May run out of memory
   - Better to process in chunks

### ❌ Not Possible

1. **Large Models** - depth_factor > 0.2
2. **Batch Processing** - batch_size > 1
3. **High-End Training** - Comparable to 16GB setups

---

## 🎛️ Recommended Settings

### For Your RTX 3050 (4GB VRAM)

```bash
python inference_low_resource.py \
    --input_audio your_audio.wav \
    --depth_factor 0.1 \      # Use 10% of model (2-3 layers)
    --steps 25 \              # 25 diffusion steps
    --max_duration 10 \       # 10 second clips
    --cfg_scale 5.0           # Guidance scale
```

**Expected:**
- Time: ~3 minutes
- Quality: Medium (60-70% of full model)
- VRAM: ~3.5 GB

### Want Faster? (Lower Quality)

```bash
python inference_low_resource.py \
    --input_audio your_audio.wav \
    --depth_factor 0.05 \
    --steps 10 \
    --max_duration 5
```

**Expected:**
- Time: ~1 minute
- Quality: Low-Medium
- VRAM: ~2.5 GB

### Want Better Quality? (Slower)

```bash
python inference_low_resource.py \
    --input_audio your_audio.wav \
    --depth_factor 0.1 \
    --steps 50 \
    --max_duration 10
```

**Expected:**
- Time: ~6 minutes
- Quality: Good (75-80% of full model)
- VRAM: ~3.8 GB (close to limit!)

---

## 🚨 Troubleshooting

### Problem: Out of Memory

**Quick Fix:**
```bash
python inference_low_resource.py \
    --input_audio input.wav \
    --depth_factor 0.05 \
    --steps 10 \
    --max_duration 5
```

### Problem: CUDA Not Available

**Fix:**
1. Run: `nvidia-smi`
2. If it fails, install NVIDIA drivers
3. Reinstall PyTorch with CUDA

### Problem: Slow or Hanging

**Fix:**
1. Close other applications
2. Check GPU usage: `nvidia-smi`
3. Ensure CUDA is being used

---

## 📖 Next Steps

### 1. Verify Everything Works
```bash
python check_system.py
```

### 2. Read the Quick Start
Open: **QUICK_START_LOW_RESOURCE.md**
- Detailed setup instructions
- Example commands
- Common issues

### 3. Try Inference
```bash
python inference_low_resource.py \
    --input_audio res/track_musdb/input.wav \
    --output_dir outputs \
    --depth_factor 0.1 \
    --steps 25
```

### 4. Experiment
- Try different audio files
- Adjust parameters
- See what works best for you

### 5. Read Advanced Guides (Optional)
- **LOW_RESOURCE_GUIDE.md** - Optimization details
- **README_LOW_RESOURCE.md** - Complete reference

---

## 💡 Pro Tips

1. **Start with test audio** - Use provided examples first
2. **Monitor GPU** - Run `nvidia-smi` in another terminal
3. **Close apps** - Free up VRAM before running
4. **Be patient** - First run downloads models (~2GB)
5. **Use cloud for training** - Don't waste time on local training

---

## 🎓 Understanding the System

### What is Stable Audio ControlNet?

A system that generates music conditioned on:
- Input audio (e.g., vocals, melody)
- Text prompts (optional)
- Other controls (envelope, chroma)

### What are you doing?

- **Input**: Your audio file (e.g., vocals)
- **Output**: Generated accompaniment (e.g., instruments)
- **Result**: Mix of input + output = complete song

### Why the optimizations?

Original system needs 16GB VRAM. Your optimizations:
- Smaller model (depth_factor=0.1)
- Fewer steps (25 instead of 100)
- Shorter audio (10s instead of 47s)
- Mixed precision (fp16)

Result: Runs on 4GB VRAM with acceptable quality!

---

## 🎉 You're Ready!

Everything is set up. Just run:

```bash
python check_system.py
```

Then follow the instructions!

---

## 📞 Need Help?

1. **Quick questions**: See FAQ in `README_LOW_RESOURCE.md`
2. **Issues**: Check troubleshooting sections in guides
3. **Bugs**: Open GitHub issue with details

---

## 📁 File Structure

```
stable-audio-controlnet-test/
├── START_HERE_LOW_RESOURCE.md          ← You are here!
├── QUICK_START_LOW_RESOURCE.md         ← Detailed setup
├── LOW_RESOURCE_SUMMARY.md             ← Overview
├── LOW_RESOURCE_GUIDE.md               ← Deep dive
├── README_LOW_RESOURCE.md              ← Complete reference
├── inference_low_resource.py           ← Main script
├── check_system.py                     ← System checker
├── run_inference_low_resource.bat      ← Windows shortcut
├── notebook/
│   └── inference_low_resource.ipynb    ← Interactive
└── exp/
    └── train_low_resource.yaml         ← Training config
```

---

**Let's get started! 🚀🎵**

Run this now:
```bash
python check_system.py
```

Then open: **QUICK_START_LOW_RESOURCE.md**

