# Running Stable Audio ControlNet on Low-Resource Hardware

This guide helps you run training and inference on limited hardware like your **RTX 3050 (4GB VRAM) with 8GB RAM**.

## Hardware Constraints

- **GPU**: RTX 3050 with 4GB VRAM
- **RAM**: 8GB system memory
- **Challenge**: Original project requires 16GB+ VRAM

## Key Optimizations for Low-Resource Systems

### 1. **Reduce Model Size**
- Use smaller `depth_factor` (0.1 instead of 0.2-0.5)
- This reduces ControlNet layers from 24 to just 2-3 layers

### 2. **Memory Management**
- Batch size = 1 (no batching)
- Gradient accumulation for effective larger batches
- Mixed precision (fp16) training
- Gradient checkpointing
- Smaller audio chunks (10-20 seconds instead of 47 seconds)

### 3. **Inference Optimizations**
- Fewer diffusion steps (25-50 instead of 100)
- Lower sample rate if acceptable (22050 Hz instead of 44100 Hz)
- Process one sample at a time

### 4. **System Optimizations**
- Close all other applications
- Disable Windows visual effects
- Use Windows page file on SSD if available
- Monitor GPU memory with `nvidia-smi`

---

## Quick Start: Inference Only (Recommended for 4GB VRAM)

Inference is much more feasible than training on 4GB VRAM. You can generate music with pre-trained models.

### Setup

```bash
# Install requirements
pip install -r requirements.txt

# Login to Hugging Face (required for Stable Audio Open weights)
huggingface-cli login

# Download a pre-trained checkpoint (optional, for audio conditioning)
# Place it in: ckpts/musdb-audio/checkpoint.ckpt
```

### Run Lightweight Inference

Use the provided `inference_low_resource.py` script:

```bash
python inference_low_resource.py \
    --checkpoint ckpts/musdb-audio/checkpoint.ckpt \
    --input_audio path/to/your/audio.wav \
    --output_dir outputs \
    --depth_factor 0.1 \
    --steps 25 \
    --cfg_scale 5.0
```

**Parameters:**
- `--depth_factor 0.1`: Minimal model size (saves ~2GB VRAM)
- `--steps 25`: Fewer diffusion steps (faster, uses less memory)
- `--cfg_scale 5.0`: Lower guidance scale (default 7.0)
- `--chunk_duration 10`: Process 10-second chunks

---

## Training on Low-Resource Hardware

⚠️ **Warning**: Training on 4GB VRAM is extremely challenging and may not work reliably. Consider using:
- Google Colab (free tier has ~15GB VRAM)
- Kaggle Notebooks (free, 16GB VRAM)
- Cloud GPU services (vast.ai, runpod.io)

### If You Must Train Locally

Use the `train_low_resource.yaml` configuration:

```bash
python train.py exp=train_low_resource \
    datamodule.train_dataset.path=data/your_dataset/train \
    datamodule.val_dataset.path=data/your_dataset/val
```

**Key settings in `train_low_resource.yaml`:**
```yaml
model:
  depth_factor: 0.1  # Minimal model (2-3 layers)

datamodule:
  batch_size_train: 1
  batch_size_val: 1
  chunk_dur: 10.0  # 10 seconds instead of 47

trainer:
  precision: 16  # Mixed precision
  accumulate_grad_batches: 8  # Effective batch size = 8
  gradient_clip_val: 1.0
  max_epochs: 50
```

### Expected Training Performance

With these optimizations:
- **VRAM usage**: ~3.5-3.8GB (close to limit!)
- **Training speed**: Very slow (~30-60 seconds per step)
- **Risk**: May crash with out-of-memory errors
- **Quality**: Lower quality due to tiny model size

### Memory Monitoring

```bash
# Monitor GPU memory in real-time
watch -n 1 nvidia-smi

# Or in PowerShell
while($true) { nvidia-smi; Start-Sleep -Seconds 1; Clear-Host }
```

---

## Alternative: Use Pre-trained Models

The best approach for your hardware is to use pre-trained checkpoints:

1. **Download pre-trained models:**
   - [MusDB model (0.5 DiT)](https://drive.google.com/drive/folders/1-EcM7RWDbLLULrcURZBFSkginaJtq_Zv?usp=sharing)
   - [MoisesDB model (0.5 DiT)](https://drive.google.com/drive/folders/1gtAJVeZmNe3fmu-WokF3n3eSaySZumrx?usp=sharing)

2. **Use inference notebook** (modified for low memory):
   - See `notebook/inference_low_resource.ipynb`

---

## Troubleshooting

### Out of Memory Errors

```python
# Add these to your script
torch.cuda.empty_cache()
import gc
gc.collect()
```

### Reduce memory further:
1. Lower `depth_factor` to 0.05 (1 layer only)
2. Reduce `chunk_dur` to 5 seconds
3. Use `steps=10` for inference (very fast, lower quality)
4. Process audio in smaller segments

### CUDA Out of Memory during inference:

```python
# Use CPU offloading
model.model.model.cpu()  # Move base model to CPU
# Keep only ControlNet on GPU
```

---

## Realistic Expectations

### ✅ What Works on 4GB VRAM:
- Inference with depth_factor=0.1 and short audio clips
- Fine-tuning pre-trained tiny models (with patience)
- Generating 10-20 second audio samples

### ❌ What Likely Won't Work:
- Training from scratch
- Using depth_factor > 0.2
- Processing 47-second audio chunks
- Batch size > 1
- High-quality training comparable to 16GB+ setups

---

## Recommended Workflow

1. **Start with inference** using pre-trained models
2. **Test on short clips** (5-10 seconds)
3. **If training is needed**, use cloud services
4. **Fine-tune** only the smallest models locally

---

## Cloud Alternatives (Free/Cheap)

If local training doesn't work:

1. **Google Colab** (Free)
   - 15GB Tesla T4 GPU
   - 12GB RAM
   - Limited to 12-hour sessions

2. **Kaggle Notebooks** (Free)
   - 16GB P100 GPU
   - 30 hours/week free

3. **Vast.ai** (Paid, ~$0.20/hour)
   - Rent GPUs by the hour
   - RTX 3090 (24GB) for ~$0.30/hour

4. **RunPod.io** (Paid, ~$0.30/hour)
   - Similar to Vast.ai
   - Good for longer training runs

---

## Next Steps

1. Try the lightweight inference script first
2. If it works, experiment with parameters
3. Consider cloud training if local training fails
4. Report any issues with memory usage

Good luck! 🚀

