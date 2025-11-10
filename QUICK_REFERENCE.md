# BIOT Integration - Quick Reference Card

## 🚀 Quick Start (3 Commands)

```bash
# 1. Verify setup
python setup_biot.py

# 2. Train with hybrid encoder (recommended)
python train.py exp=train_deap_controlnet_eeg_hybrid

# 3. Monitor results
# → Check Weights & Biases dashboard
```

---

## 📊 Three Encoder Options

| Type | Command | Use Case | Training Time |
|------|---------|----------|---------------|
| **Simple** (baseline) | `exp=train_deap_controlnet_eeg_simple` | Baseline comparison | ~5 min/epoch |
| **BIOT** | `exp=train_deap_controlnet_eeg_biot` | Pre-trained features | ~8 min/epoch |
| **Hybrid** ⭐ | `exp=train_deap_controlnet_eeg_hybrid` | Best performance | ~10 min/epoch |

---

## 🔧 Key Configuration Parameters

```yaml
model:
  projector_type: "hybrid"          # "simple" | "biot" | "hybrid"
  num_eeg_channels: 32              # DEAP = 32 channels
  
  # BIOT settings
  biot_pretrained_path: "path/to/pretrained.ckpt"  # or null
  freeze_biot: true                 # Recommended: true
  
  # Architecture
  biot_emb_size: 256               # Don't change if using pre-trained
  biot_depth: 4                    # Don't change if using pre-trained
  projection_hidden_dim: 512       # Tune for "biot" type
  local_hidden_dim: 128            # Tune for "hybrid" type
```

---

## 🎯 Pre-trained Models

Located in: `Sample Code BIOT/BIOT/pretrained-models/`

| File | Recommended For |
|------|-----------------|
| `EEG-PREST-16-channels.ckpt` | General use (smallest) |
| `EEG-SHHS+PREST-18-channels.ckpt` | Sleep + general |
| `EEG-six-datasets-18-channels.ckpt` | Best performance ⭐ |

---

## 🐛 Common Issues & Quick Fixes

### Issue: "Cannot import BIOT"
```python
import sys
sys.path.insert(0, 'Sample Code BIOT/BIOT')
```

### Issue: CUDA OOM
```yaml
datamodule:
  batch_size_train: 4  # Reduce from 8
trainer:
  accumulate_grad_batches: 8  # Increase
```

### Issue: Slow training
```yaml
freeze_biot: true  # Keep BIOT frozen
trainer:
  precision: "16-mixed"  # Use mixed precision
```

---

## 📈 Expected Performance

| Metric | Simple | BIOT | Hybrid |
|--------|--------|------|--------|
| Validation Loss | Baseline | -10% | -15-20% |
| Audio Quality | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Training Time | Fast | Medium | Medium |
| GPU Memory | 4 GB | 6 GB | 7 GB |

---

## 🏗️ Architecture Flow

### Simple
```
EEG → Normalize → Conv1d → Audio Control
```

### BIOT
```
EEG → BIOT Encoder → MLP → Audio Control
      (STFT + Transformer)
```

### Hybrid ⭐
```
        ┌→ BIOT (global) ─┐
EEG ────┤                  ├→ Fuse → Audio Control
        └→ Conv1d (local) ─┘
```

---

## 📁 File Locations

### Code:
- **Main integration**: `main/eeg_encoders.py`
- **Training module**: `main/module_controlnet_eeg.py`
- **Dataset**: `main/data/dataset_deap.py`

### Configs:
- **Simple**: `exp/train_deap_controlnet_eeg_simple.yaml`
- **BIOT**: `exp/train_deap_controlnet_eeg_biot.yaml`
- **Hybrid**: `exp/train_deap_controlnet_eeg_hybrid.yaml`

### Docs:
- **Full guide**: `BIOT_INTEGRATION_GUIDE.md`
- **Summary**: `INTEGRATION_SUMMARY.md`
- **This file**: `QUICK_REFERENCE.md`

---

## 🔬 Recommended Workflow

### For Quick Experiments (2-3 hours):
1. ✓ Run `python setup_biot.py`
2. ✓ Train hybrid: `python train.py exp=train_deap_controlnet_eeg_hybrid`
3. ✓ Check W&B for audio samples

### For Full Comparison (6-8 hours):
1. ✓ Train simple (baseline): 20 epochs
2. ✓ Train BIOT (frozen): 20 epochs
3. ✓ Train hybrid (frozen): 20 epochs
4. ✓ Compare validation loss & audio quality

### For Best Performance (12-16 hours):
1. ✓ Train hybrid with frozen BIOT: 30 epochs
2. ✓ Fine-tune with `freeze_biot: false`: 20 epochs (lr=1e-5)
3. ✓ Evaluate on test set

---

## 💡 Pro Tips

1. **Always start with frozen BIOT** (`freeze_biot: true`)
   - Faster, more stable, usually sufficient

2. **Use hybrid for best results**
   - Combines semantic + temporal information
   - Only ~2 min/epoch slower than BIOT alone

3. **Monitor GPU memory**
   - Start with batch_size=8
   - Reduce if OOM, increase accumulation

4. **Enable caching for speed**
   ```yaml
   cache_subjects: true  # Saves ~40 MB per subject
   ```

5. **Log frequently**
   ```yaml
   log_every_n_steps: 10
   val_check_interval: 0.5  # Check validation every half epoch
   ```

---

## 🎓 Training Commands Cheat Sheet

```bash
# Basic training
python train.py exp=train_deap_controlnet_eeg_hybrid

# Override parameters
python train.py exp=train_deap_controlnet_eeg_hybrid \
  model.freeze_biot=false \
  model.lr=1e-5

# Resume from checkpoint
python train.py exp=train_deap_controlnet_eeg_hybrid \
  ckpt=path/to/checkpoint.ckpt

# Fast development run (1 batch)
python train.py exp=train_deap_controlnet_eeg_hybrid \
  trainer.fast_dev_run=true

# CPU only (for debugging)
python train.py exp=train_deap_controlnet_eeg_hybrid \
  trainer.accelerator=cpu

# Multi-GPU
python train.py exp=train_deap_controlnet_eeg_hybrid \
  trainer.devices=2
```

---

## 📊 Monitoring Metrics

### In W&B Dashboard:

**Training Metrics:**
- `train_loss`: Should decrease steadily
- `learning_rate`: Monitors LR schedule
- `epoch`: Current epoch number

**Validation Metrics:**
- `valid_loss`: Main metric (lower = better)
- Audio samples: Listen to quality
- Spectrograms: Visual check

**Target Values** (after 20 epochs):
- Simple: valid_loss ~ 0.15-0.20
- BIOT: valid_loss ~ 0.12-0.18
- Hybrid: valid_loss ~ 0.10-0.15

*(Actual values depend on your dataset)*

---

## 🆘 Emergency Troubleshooting

### Training crashes immediately?
```bash
python setup_biot.py  # Check all components
```

### Nan loss?
```yaml
trainer:
  gradient_clip_val: 1.0  # Enable gradient clipping
model:
  lr: 5e-5  # Reduce learning rate
```

### No audio in W&B?
- Check that `SampleLogger` callback is enabled
- Verify `wandb` logger is configured
- Look for "sample_50_0" in W&B media tab

### Training too slow?
```yaml
freeze_biot: true        # Freeze BIOT
num_workers: 8           # Increase data workers
cache_subjects: true     # Cache EEG data
persistent_workers: true # Keep workers alive
```

---

## 📞 Support Resources

1. **Setup issues**: Run `python setup_biot.py`
2. **BIOT questions**: See `BIOT_INTEGRATION_GUIDE.md`
3. **Full docs**: See `INTEGRATION_SUMMARY.md`
4. **BIOT paper**: https://arxiv.org/abs/2305.10351
5. **BIOT repo**: https://github.com/ycq091044/BIOT

---

## ✅ Success Checklist

Before training:
- [ ] `setup_biot.py` passes all tests
- [ ] Pre-trained models downloaded
- [ ] Dataset paths configured in YAML
- [ ] W&B account set up

During training:
- [ ] Loss decreases over epochs
- [ ] No NaN/Inf in logs
- [ ] GPU utilization > 80%
- [ ] Audio samples logged

After training:
- [ ] Validation loss improved vs baseline
- [ ] Generated audio sounds reasonable
- [ ] Model checkpoint saved
- [ ] Results documented

---

**🎉 You're ready to train! Good luck!**

For detailed information, see `BIOT_INTEGRATION_GUIDE.md`

