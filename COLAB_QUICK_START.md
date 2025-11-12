# Google Colab Quick Start

## The Problem: "Nothing Happens"

When you run `python train.py ...` in Colab, nothing happens because **environment variables are missing**.

---

## The Solution (Copy-Paste This!)

### Step 1: Setup Environment (Run First!)

```python
# Run this in a Colab cell BEFORE training
!python setup_colab_env.py
```

Or manually:

```python
import os

# Set environment variables
os.environ['DIR_LOGS'] = '/content/logs'
os.environ['DIR_DATA'] = '/content/data'
os.environ['TAG'] = 'colab-training'

# Create .env file
with open('.env', 'w') as f:
    f.write("DIR_LOGS=/content/logs\nDIR_DATA=/content/data\n")

# Create directories
!mkdir -p /content/logs /content/data

print("✅ Environment ready!")
```

### Step 2: Upload Dataset

```python
# Copy from Google Drive
!cp /content/drive/MyDrive/datasets/train.tar /content/data/
!cp /content/drive/MyDrive/datasets/test.tar /content/data/

# Verify
!ls -lh /content/data/
```

### Step 3: Login to Hugging Face

```python
from huggingface_hub import notebook_login
notebook_login()
```

### Step 4: Train!

```bash
!python train.py \
    exp=train_colab_optimized \
    datamodule.train_dataset.path=/content/data/train.tar \
    datamodule.val_dataset.path=/content/data/test.tar
```

---

## Complete Colab Notebook (All Steps)

```python
# ===== CELL 1: Check GPU =====
!nvidia-smi

# ===== CELL 2: Mount Drive =====
from google.colab import drive
drive.mount('/content/drive')

# ===== CELL 3: Clone/Upload Project =====
# Option A: Clone from GitHub
!git clone YOUR_REPO_URL
%cd YOUR_REPO/stable-audio-controlnet-test

# Option B: Upload files
# from google.colab import files
# uploaded = files.upload()

# ===== CELL 4: Install Dependencies =====
!pip install -q -r requirements.txt
!pip install -q --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

# ===== CELL 5: Setup Environment (CRITICAL!) =====
!python setup_colab_env.py

# ===== CELL 6: Login to Hugging Face =====
from huggingface_hub import notebook_login
notebook_login()

# ===== CELL 7: Upload Dataset =====
!cp /content/drive/MyDrive/datasets/*.tar /content/data/
!ls -lh /content/data/

# ===== CELL 8: Start Training =====
!python train.py \
    exp=train_colab_optimized \
    datamodule.train_dataset.path=/content/data/train.tar \
    datamodule.val_dataset.path=/content/data/test.tar

# ===== CELL 9: Save Checkpoints =====
!mkdir -p /content/drive/MyDrive/stable-audio-checkpoints
!cp -r /content/logs/ckpts/* /content/drive/MyDrive/stable-audio-checkpoints/
```

---

## Why This Happens

The training script (`train.py`) uses Hydra configuration that expects:

1. **Environment variables**: `DIR_LOGS`, `DIR_DATA`
2. **.env file**: Contains these variables
3. **Directories**: `/content/logs`, `/content/data`

Without these, Hydra fails silently and nothing happens.

---

## Configuration Options

### For T4 GPU (15GB) - Colab Free

```bash
!python train.py \
    exp=train_colab_optimized \
    model.depth_factor=0.3 \
    datamodule.batch_size_train=2 \
    ...
```

### For P100/V100 (16-32GB) - Colab Pro

```bash
!python train.py \
    exp=train_colab_optimized \
    model.depth_factor=0.5 \
    datamodule.batch_size_train=4 \
    ...
```

### For A100 (40GB) - Colab Pro+

```bash
!python train.py \
    exp=train_colab_optimized \
    model.depth_factor=0.5 \
    datamodule.batch_size_train=8 \
    ...
```

---

## Troubleshooting

### Still nothing happens?

```bash
# Check environment
!echo $DIR_LOGS
!echo $DIR_DATA
!cat .env

# Check logs
!ls -lh /content/logs/
!tail /content/logs/runs/*/main.log
```

### Out of memory?

```bash
# Reduce model size
!python train.py \
    exp=train_colab_optimized \
    model.depth_factor=0.2 \
    datamodule.batch_size_train=1 \
    ...
```

### Dataset not found?

```bash
# Verify dataset
!ls -lh /content/data/

# Use correct path
datamodule.train_dataset.path=/content/data/train.tar
```

---

## Quick Reference

| Issue | Solution |
|-------|----------|
| Nothing happens | Run `setup_colab_env.py` first |
| Missing .env | Create with `DIR_LOGS` and `DIR_DATA` |
| No GPU | Runtime → Change runtime type → GPU |
| OOM error | Reduce `depth_factor` and `batch_size` |
| Dataset error | Check path with `!ls /content/data/` |
| Colab disconnects | Save checkpoints to Google Drive |

---

## Save Your Work!

```python
# Save checkpoints to Google Drive (important!)
!cp -r /content/logs/ckpts/* /content/drive/MyDrive/stable-audio-checkpoints/

# Resume training later
!python train.py \
    exp=train_colab_optimized \
    +ckpt=/content/drive/MyDrive/stable-audio-checkpoints/last.ckpt \
    ...
```

---

## Need More Help?

See detailed guide: `COLAB_TRAINING_GUIDE.md`

---

**TL;DR: Run `setup_colab_env.py` before training!** 🚀

