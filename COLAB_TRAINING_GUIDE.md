# Google Colab Training Guide

## Why "Nothing Happens" When You Run Training

The training script requires **environment variables** that are missing in Google Colab. Here's how to fix it:

---

## Quick Fix (Run These First!)

### Step 1: Set Environment Variables

```python
import os

# REQUIRED: Set these environment variables
os.environ['DIR_LOGS'] = '/content/logs'
os.environ['DIR_DATA'] = '/content/data'
os.environ['TAG'] = 'colab-training'

# Create directories
!mkdir -p /content/logs
!mkdir -p /content/data

print("✅ Environment configured!")
```

### Step 2: Create .env File

```python
# Create .env file
env_content = """
DIR_LOGS=/content/logs
DIR_DATA=/content/data
"""

with open('.env', 'w') as f:
    f.write(env_content.strip())

print("✅ .env file created!")
```

### Step 3: Now Run Training

```bash
python train.py \
    exp=train_musdb_controlnet_audio \
    datamodule.train_dataset.path=/content/data/train.tar \
    datamodule.val_dataset.path=/content/data/test.tar \
    model.depth_factor=0.3 \
    datamodule.batch_size_train=2
```

---

## Complete Colab Setup (Copy-Paste Ready)

### 1. Check GPU

```python
!nvidia-smi
```

### 2. Mount Google Drive (for saving checkpoints)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Clone/Upload Your Project

```bash
# Option A: Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO/stable-audio-controlnet-test

# Option B: Upload files
from google.colab import files
uploaded = files.upload()
```

### 4. Install Dependencies

```bash
!pip install -q -r requirements.txt
!pip install -q --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

### 5. Setup Environment (CRITICAL!)

```python
import os

# Set environment variables
os.environ['DIR_LOGS'] = '/content/logs'
os.environ['DIR_DATA'] = '/content/data'
os.environ['TAG'] = 'colab-training'

# Create .env file
with open('.env', 'w') as f:
    f.write("""DIR_LOGS=/content/logs
DIR_DATA=/content/data
""")

# Create directories
!mkdir -p /content/logs /content/data

print("✅ Environment ready!")
```

### 6. Login to Hugging Face

```python
from huggingface_hub import notebook_login
notebook_login()
```

### 7. Upload Dataset

```python
# Option A: From Google Drive
!cp /content/drive/MyDrive/datasets/train.tar /content/data/
!cp /content/drive/MyDrive/datasets/test.tar /content/data/

# Option B: Upload directly
from google.colab import files
uploaded = files.upload()

# Verify
!ls -lh /content/data/
```

### 8. Start Training

```bash
!python train.py \
    exp=train_musdb_controlnet_audio \
    datamodule.train_dataset.path=/content/data/train.tar \
    datamodule.val_dataset.path=/content/data/test.tar \
    model.depth_factor=0.3 \
    datamodule.batch_size_train=2 \
    datamodule.batch_size_val=2 \
    trainer.max_epochs=50
```

### 9. Monitor Training

```python
# Check GPU usage
!nvidia-smi

# Check logs
!tail -f /content/logs/runs/*/main.log

# Check checkpoints
!ls -lh /content/logs/ckpts/
```

### 10. Save Checkpoints to Drive

```bash
# Save to Google Drive (important!)
!mkdir -p /content/drive/MyDrive/stable-audio-checkpoints
!cp -r /content/logs/ckpts/* /content/drive/MyDrive/stable-audio-checkpoints/

echo "✅ Checkpoints saved!"
```

---

## Configuration for Google Colab T4 (15GB VRAM)

### Recommended Settings

```yaml
# Good balance for T4 GPU
model.depth_factor: 0.3          # 30% of model (7 layers)
datamodule.batch_size_train: 2   # Batch size 2
datamodule.batch_size_val: 2
datamodule.chunk_dur: 30.0       # 30 second chunks
trainer.accumulate_grad_batches: 4  # Effective batch = 8
trainer.max_epochs: 50
```

### Conservative Settings (if OOM)

```yaml
# Use if you get out of memory errors
model.depth_factor: 0.2          # 20% of model (5 layers)
datamodule.batch_size_train: 1   # Batch size 1
datamodule.batch_size_val: 1
datamodule.chunk_dur: 20.0       # 20 second chunks
trainer.accumulate_grad_batches: 8  # Effective batch = 8
```

### Aggressive Settings (if you have Colab Pro)

```yaml
# For A100 or V100 GPUs
model.depth_factor: 0.5          # 50% of model (12 layers)
datamodule.batch_size_train: 4   # Batch size 4
datamodule.batch_size_val: 4
datamodule.chunk_dur: 47.0       # Full 47 second chunks
trainer.accumulate_grad_batches: 2  # Effective batch = 8
```

---

## Troubleshooting

### Problem 1: "Nothing happens" when running train.py

**Cause:** Missing environment variables

**Solution:**
```python
import os
os.environ['DIR_LOGS'] = '/content/logs'
os.environ['DIR_DATA'] = '/content/data'
os.environ['TAG'] = 'colab-training'

# Also create .env file
with open('.env', 'w') as f:
    f.write("DIR_LOGS=/content/logs\nDIR_DATA=/content/data\n")
```

### Problem 2: "Repository not found" error

**Cause:** Not logged into Hugging Face

**Solution:**
```python
from huggingface_hub import notebook_login
notebook_login()
```

### Problem 3: Out of Memory

**Cause:** Model too large for GPU

**Solution:**
```bash
# Reduce model size
python train.py \
    exp=train_musdb_controlnet_audio \
    model.depth_factor=0.2 \
    datamodule.batch_size_train=1 \
    ...
```

### Problem 4: Dataset not found

**Cause:** Dataset path incorrect

**Solution:**
```bash
# Check dataset location
!ls -lh /content/data/

# Use correct path in training command
datamodule.train_dataset.path=/content/data/train.tar
```

### Problem 5: Colab disconnects

**Cause:** 12-hour session limit

**Solution:**
```python
# Save checkpoints frequently to Google Drive
!cp -r /content/logs/ckpts/* /content/drive/MyDrive/stable-audio-checkpoints/

# Resume training with checkpoint
!python train.py \
    exp=train_musdb_controlnet_audio \
    +ckpt=/content/drive/MyDrive/stable-audio-checkpoints/last.ckpt \
    ...
```

---

## Performance Expectations

### Google Colab Free (T4 GPU, 15GB VRAM)

| Config | Time/Epoch | VRAM Usage | Quality |
|--------|------------|------------|---------|
| depth=0.2, batch=1 | ~30 min | ~10 GB | Medium |
| depth=0.3, batch=2 | ~40 min | ~13 GB | Good |
| depth=0.5, batch=1 | ~50 min | ~14 GB | Excellent |

### Google Colab Pro (A100 GPU, 40GB VRAM)

| Config | Time/Epoch | VRAM Usage | Quality |
|--------|------------|------------|---------|
| depth=0.5, batch=4 | ~15 min | ~28 GB | Excellent |
| depth=0.5, batch=8 | ~12 min | ~35 GB | Excellent |

---

## Tips for Successful Training

### 1. Save Frequently
```python
# Auto-save to Google Drive every N epochs
# Add this to your config:
callbacks:
  model_checkpoint:
    every_n_epochs: 5
```

### 2. Monitor GPU
```python
# Run in separate cell
import time
while True:
    !nvidia-smi
    time.sleep(5)
```

### 3. Use Weights & Biases (Optional)
```python
# For better logging
os.environ['WANDB_PROJECT'] = 'stable-audio'
os.environ['WANDB_ENTITY'] = 'your-username'
os.environ['WANDB_API_KEY'] = 'your-api-key'
```

### 4. Test Configuration First
```bash
# Run for 1 epoch to test
python train.py \
    exp=train_musdb_controlnet_audio \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10 \
    trainer.limit_val_batches=5
```

### 5. Keep Colab Active
- Don't close the browser tab
- Interact with notebook every few hours
- Consider Colab Pro for longer sessions

---

## Example: Complete Colab Session

```python
# 1. Setup
!nvidia-smi
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone project
!git clone YOUR_REPO
%cd stable-audio-controlnet-test

# 3. Install
!pip install -q -r requirements.txt
!pip install -q --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

# 4. Configure environment
import os
os.environ['DIR_LOGS'] = '/content/logs'
os.environ['DIR_DATA'] = '/content/data'
os.environ['TAG'] = 'colab-training'

with open('.env', 'w') as f:
    f.write("DIR_LOGS=/content/logs\nDIR_DATA=/content/data\n")

!mkdir -p /content/logs /content/data

# 5. Login to HF
from huggingface_hub import notebook_login
notebook_login()

# 6. Upload dataset
!cp /content/drive/MyDrive/datasets/*.tar /content/data/

# 7. Train!
!python train.py \
    exp=train_musdb_controlnet_audio \
    datamodule.train_dataset.path=/content/data/train.tar \
    datamodule.val_dataset.path=/content/data/test.tar \
    model.depth_factor=0.3 \
    datamodule.batch_size_train=2

# 8. Save checkpoints
!cp -r /content/logs/ckpts/* /content/drive/MyDrive/stable-audio-checkpoints/
```

---

## Next Steps

1. **Start with this guide** - Follow the steps above
2. **Test with small config** - Verify everything works
3. **Scale up gradually** - Increase model size if successful
4. **Monitor closely** - Watch GPU and logs
5. **Save frequently** - Don't lose your progress!

---

**Good luck with training! 🚀**

If you still have issues, check:
- Are environment variables set? `!echo $DIR_LOGS`
- Does .env file exist? `!cat .env`
- Is dataset present? `!ls /content/data/`
- Any errors in logs? `!tail /content/logs/runs/*/main.log`

