# Quick Start Guide for Low-Resource Hardware

**Optimized for: RTX 3050 (4GB VRAM) + 8GB RAM**

This guide will get you up and running with inference in under 10 minutes!

---

## Prerequisites

1. **NVIDIA GPU**: RTX 3050 or similar (4GB VRAM minimum)
2. **CUDA**: Version 11.8 or 12.x
3. **Python**: 3.10 or 3.11
4. **Storage**: ~10GB free space for models and checkpoints

---

## Step 1: Install Dependencies (5 minutes)

```bash
# Navigate to project directory
cd stable-audio-controlnet-test

# Install requirements
pip install -r requirements.txt

# Install nightly torchaudio (required)
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

# Login to Hugging Face (required for Stable Audio Open)
huggingface-cli login
```

When prompted, enter your Hugging Face token (get one from https://huggingface.co/settings/tokens)

---

## Step 2: Choose Your Method

### Option A: Command Line (Fastest)

```bash
python inference_low_resource.py \
    --input_audio path/to/your/audio.wav \
    --output_dir outputs \
    --depth_factor 0.1 \
    --steps 25
```

**On Windows:**
```cmd
python inference_low_resource.py --input_audio path\to\your\audio.wav --output_dir outputs --depth_factor 0.1 --steps 25
```

Or simply double-click `run_inference_low_resource.bat` (after editing the INPUT_AUDIO path)

### Option B: Jupyter Notebook (Interactive)

```bash
cd notebook
jupyter notebook inference_low_resource.ipynb
```

Then run all cells!

---

## Step 3: Test with Example Audio

```bash
# Use the provided example audio
python inference_low_resource.py \
    --input_audio res/track_musdb/input.wav \
    --output_dir outputs \
    --depth_factor 0.1 \
    --steps 25
```

**Expected output:**
- `outputs/input.wav` - Your input audio
- `outputs/output.wav` - Generated accompaniment
- `outputs/mix.wav` - Input + output mixed

**Time:** ~2-5 minutes on RTX 3050

---

## Understanding Parameters

### Critical Parameters (Affect Memory)

| Parameter | 4GB VRAM | 6GB VRAM | 8GB+ VRAM |
|-----------|----------|----------|-----------|
| `--depth_factor` | 0.05-0.1 | 0.1-0.2 | 0.2-0.5 |
| `--steps` | 10-25 | 25-50 | 50-100 |
| `--max_duration` | 5-10s | 10-20s | 20-47s |

### Quality Parameters

| Parameter | Description | Default | Low Resource |
|-----------|-------------|---------|--------------|
| `--cfg_scale` | Guidance strength | 7.0 | 5.0 |
| `--seed` | Random seed | 42 | Any |

---

## Troubleshooting

### ❌ Out of Memory Error

**Error:** `RuntimeError: CUDA out of memory`

**Solutions (try in order):**
1. Reduce `--depth_factor` to `0.05`
2. Reduce `--steps` to `10`
3. Reduce `--max_duration` to `5`
4. Close all other applications
5. Restart your computer

**Example minimal command:**
```bash
python inference_low_resource.py \
    --input_audio input.wav \
    --depth_factor 0.05 \
    --steps 10 \
    --max_duration 5
```

### ❌ CUDA Not Available

**Error:** `CUDA not available, falling back to CPU`

**Solutions:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### ❌ Hugging Face Login Error

**Error:** `Repository not found` or `Authentication required`

**Solution:**
```bash
huggingface-cli login
```
Enter your token from https://huggingface.co/settings/tokens

### ❌ Audio File Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:**
- Use absolute path: `C:\path\to\audio.wav`
- Or relative path from project root: `res/track_musdb/input.wav`

---

## Performance Expectations

### RTX 3050 (4GB VRAM)

| Configuration | Time | Quality | VRAM Usage |
|---------------|------|---------|------------|
| depth=0.05, steps=10 | ~1 min | Low | ~2.5 GB |
| depth=0.1, steps=25 | ~3 min | Medium | ~3.5 GB |
| depth=0.1, steps=50 | ~6 min | Good | ~3.8 GB |

### RTX 3060 (6GB VRAM)

| Configuration | Time | Quality | VRAM Usage |
|---------------|------|---------|------------|
| depth=0.1, steps=50 | ~4 min | Good | ~4.5 GB |
| depth=0.2, steps=100 | ~10 min | High | ~5.5 GB |

---

## Next Steps

### 1. Use Pre-trained Checkpoints (Recommended)

Download pre-trained models for better quality:

- **MusDB model**: https://drive.google.com/drive/folders/1-EcM7RWDbLLULrcURZBFSkginaJtq_Zv
- **MoisesDB model**: https://drive.google.com/drive/folders/1gtAJVeZmNe3fmu-WokF3n3eSaySZumrx

Place in `ckpts/musdb-audio/` or `ckpts/moisesdb-audio/`

Then run:
```bash
python inference_low_resource.py \
    --checkpoint ckpts/musdb-audio/checkpoint.ckpt \
    --input_audio your_audio.wav \
    --depth_factor 0.5 \
    --steps 100
```

**Note:** Pre-trained checkpoints use `depth_factor=0.5`, which requires ~5-6GB VRAM. If you have 4GB, you can still load them but may need to use CPU offloading (slower).

### 2. Training (Not Recommended for 4GB)

Training on 4GB VRAM is extremely challenging. Consider:
- **Google Colab** (Free, 15GB T4 GPU)
- **Kaggle Notebooks** (Free, 16GB P100 GPU)
- **Cloud GPUs** (Vast.ai, RunPod.io - ~$0.20-0.30/hour)

If you must train locally, see `LOW_RESOURCE_GUIDE.md`

### 3. Experiment with Parameters

Try different configurations:

```bash
# Fast preview (30 seconds)
python inference_low_resource.py \
    --input_audio input.wav \
    --depth_factor 0.05 \
    --steps 10 \
    --max_duration 5

# Balanced (3 minutes)
python inference_low_resource.py \
    --input_audio input.wav \
    --depth_factor 0.1 \
    --steps 25 \
    --max_duration 10

# Best quality on 4GB (6 minutes)
python inference_low_resource.py \
    --input_audio input.wav \
    --depth_factor 0.1 \
    --steps 50 \
    --max_duration 10
```

---

## Monitoring GPU Usage

### Windows PowerShell
```powershell
# Watch GPU usage in real-time
while($true) { nvidia-smi; Start-Sleep -Seconds 1; Clear-Host }
```

### Linux/Mac
```bash
watch -n 1 nvidia-smi
```

Look for:
- **Memory-Usage**: Should stay under 3.8GB for 4GB cards
- **GPU-Util**: Should be high (80-100%) during generation
- **Temperature**: Should stay under 85°C

---

## Tips for Best Results

1. **Start small**: Test with 5-second clips first
2. **Monitor memory**: Watch `nvidia-smi` during generation
3. **Close applications**: Free up VRAM before running
4. **Use SSD**: Faster model loading
5. **Update drivers**: Latest NVIDIA drivers recommended
6. **Be patient**: 4GB VRAM is challenging but workable!

---

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Try the troubleshooting steps above
3. Review `LOW_RESOURCE_GUIDE.md` for detailed explanations
4. Check GPU memory with `nvidia-smi`
5. Open an issue on GitHub with:
   - Your GPU model and VRAM
   - Full error message
   - Command you ran
   - Output of `nvidia-smi`

---

## Success Checklist

- [ ] Dependencies installed
- [ ] Hugging Face login completed
- [ ] Test inference runs successfully
- [ ] Output audio files generated
- [ ] GPU memory usage monitored
- [ ] Parameters adjusted for your hardware

---

**You're all set! Happy generating! 🎵**

For more advanced usage, see:
- `LOW_RESOURCE_GUIDE.md` - Detailed optimization guide
- `README.md` - Full project documentation
- `notebook/inference_low_resource.ipynb` - Interactive notebook

