# Complete Data Flow: BIOT EEG Projector in Action

## Overview

This document traces the **exact execution path** of how the BIOT EEG projector is called and contributes to the EEG-to-music generation system.

---

## 🔍 Key Call Sites

The `forward()` function of the EEG projector is called in **two main places**:

1. **Training/Validation** (`module_controlnet_eeg.py:153`): During the training step
2. **Inference/Logging** (`module_controlnet_eeg.py:300`): During sample generation for W&B logging

---

## 📊 Complete Training Pipeline with BIOT

### Step-by-Step Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. TRAINING STEP BEGINS                                             │
│     (module_controlnet_eeg.py: Model.step())                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. LOAD BATCH FROM DEAP DATASET                                     │
│     Input: batch = (x, eeg, prompts, start_seconds, total_seconds)  │
│                                                                       │
│     x: [batch=8, 2 channels, 441000 samples]  (target audio)       │
│     eeg: [batch=8, 32 channels, 1280 samples] (EEG signals)        │
│     prompts: List[str] (text descriptions)                          │
│     start_seconds: List[float] (timing info)                        │
│     total_seconds: List[float] (duration info)                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. ENCODE TARGET AUDIO (Line 152)                                  │
│     diffusion_input = self.model.pretransform.encode(x)             │
│                                                                       │
│     [8, 2, 441000] → VAE Encoder → [8, 64, 1024]                   │
│                                     (latent representation)          │
│                                                                       │
│     Status: FROZEN (pre-trained Stable Audio VAE)                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. PROJECT EEG TO CONTROL SIGNAL ⭐ (Line 153)                     │
│     projected_eeg = self.eeg_projector(eeg.to(self.device))         │
│                                                                       │
│     THIS IS WHERE BIOT IS CALLED!                                   │
│                                                                       │
│     Input:  eeg [8, 32, 1280]                                       │
│     Output: projected_eeg [8, 2, 441000]                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────┴───────────────────────┐
        │                                                │
        │   BIOT EEG PROJECTOR FORWARD PASS             │
        │   (eeg_encoders.py: BIOTEEGProjector.forward())│
        │                                                │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4a. INSIDE BIOT PROJECTOR - Step 1: BIOT Encoder                  │
│      (eeg_encoders.py:166)                                          │
│                                                                       │
│      with torch.set_grad_enabled(not self.freeze_biot):            │
│          biot_features = self.biot_encoder(eeg)                     │
│                                                                       │
│      Input:  [8, 32, 1280]                                          │
│      Output: [8, 256]  (global semantic embedding)                  │
│                                                                       │
│      IF freeze_biot=True: Gradients disabled, uses pre-trained      │
│      IF freeze_biot=False: Gradients enabled, fine-tuning           │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                                │
        │   BIOT ENCODER INTERNAL PROCESSING             │
        │   (Sample Code BIOT/BIOT/model/biot.py)       │
        │                                                │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BIOT ENCODER DETAILED STEPS:                                        │
│                                                                       │
│  For each of 32 EEG channels:                                        │
│    │                                                                  │
│    ├─▶ Step 1: STFT (Short-Time Fourier Transform)                 │
│    │   Input:  [batch, 1, 1280] (single channel)                    │
│    │   STFT:   n_fft=200, hop_length=100                            │
│    │   Output: [batch, 101 freq bins, ~13 time frames]              │
│    │                                                                  │
│    ├─▶ Step 2: Frequency Embedding                                  │
│    │   Linear projection: [101] → [256]                             │
│    │   Output: [batch, ~13 time, 256]                               │
│    │                                                                  │
│    ├─▶ Step 3: Add Channel Token                                    │
│    │   Each channel gets unique learnable embedding                  │
│    │   channel_token[channel_idx] broadcasts to all timesteps        │
│    │                                                                  │
│    └─▶ Step 4: Add Positional Encoding                              │
│        Sinusoidal encoding for temporal position                     │
│        Output: [batch, ~13 time, 256] for this channel              │
│                                                                       │
│  Concatenate all 32 channels:                                        │
│    → [batch, 32 × ~13 ≈ 416 tokens, 256]                           │
│                                                                       │
│  Pass through Linear Attention Transformer:                          │
│    ├─▶ 4 layers of Linear Attention                                 │
│    ├─▶ 8 attention heads                                            │
│    ├─▶ Dimension: 256                                               │
│    └─▶ Dropout: 0.2                                                 │
│                                                                       │
│  Mean pooling across all tokens:                                     │
│    [batch, 416 tokens, 256] → Mean → [batch, 256]                  │
│                                                                       │
│  Final output: [batch, 256] semantic EEG embedding                  │
│                                                                       │
│  ✨ This captures:                                                   │
│     - Frequency patterns across channels                             │
│     - Temporal dynamics                                              │
│     - Cross-channel relationships                                    │
│     - Semantic meaning from pre-training                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4b. INSIDE BIOT PROJECTOR - Step 2: MLP Projection                │
│      (eeg_encoders.py:169)                                          │
│                                                                       │
│      projected = self.projection_net(biot_features)                 │
│                                                                       │
│      MLP Architecture:                                               │
│      ┌─────────────────────────────────────────────────────┐       │
│      │ Linear(256 → 512)                                   │       │
│      │ LayerNorm(512) + GELU + Dropout(0.1)               │       │
│      │ Linear(512 → 512)                                   │       │
│      │ LayerNorm(512) + GELU + Dropout(0.1)               │       │
│      │ Linear(512 → 882000)  [2 × 441000]                 │       │
│      └─────────────────────────────────────────────────────┘       │
│                                                                       │
│      Input:  [8, 256]                                               │
│      Output: [8, 882000]                                            │
│                                                                       │
│      Status: TRAINABLE (learns to map BIOT features → audio space) │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4c. INSIDE BIOT PROJECTOR - Step 3: Reshape & Tanh                │
│      (eeg_encoders.py:173-176)                                      │
│                                                                       │
│      control_signal = projected.view(batch_size, 2, target_length)  │
│      return torch.tanh(control_signal)                              │
│                                                                       │
│      [8, 882000] → Reshape → [8, 2, 441000]                        │
│                           → Tanh → [8, 2, 441000] in [-1, 1]       │
│                                                                       │
│      Output: 2-channel audio-like control signal                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. BUILD CONDITIONING DICT (Line 171)                              │
│     conditioning = self._build_conditioning(                        │
│         projected_eeg, prompts, start_seconds, total_seconds        │
│     )                                                                │
│                                                                       │
│     Creates list of dicts, one per sample:                          │
│     [                                                                │
│       {                                                              │
│         "prompt": "subject: s01; valence: 5.2; ...",               │
│         "seconds_start": 0.0,                                       │
│         "seconds_total": 10.0,                                      │
│         "audio": projected_eeg[0:1]  ← BIOT output used here!     │
│       },                                                             │
│       { ... },  # for each sample in batch                          │
│     ]                                                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. DIFFUSION NOISE SCHEDULE (Lines 155-169)                        │
│     - Sample timestep t ~ Logit-Normal                              │
│     - Compute α(t) and σ(t)                                         │
│     - Add noise: noised = latent × α(t) + noise × σ(t)             │
│     - Compute target: target = noise × α(t) - latent × σ(t)        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  7. FORWARD PASS THROUGH CONTROLNET (Lines 173-178)                │
│     output = self.model(                                            │
│         x=noised_inputs,        [8, 64, 1024] noised latent        │
│         t=t,                    timestep                            │
│         cond=conditioner(conditioning),  ← BIOT control included!  │
│         cfg_dropout_prob=0.1,                                       │
│     )                                                                │
│                                                                       │
│     Inside ControlNet:                                               │
│     ├─▶ Text conditioning processed (FROZEN)                        │
│     ├─▶ EEG control signal processed (TRAINABLE ControlNet layers) │
│     │   - BIOT output guides the generation                         │
│     │   - ControlNet learns to use EEG features                     │
│     └─▶ Predict velocity for denoising                              │
│                                                                       │
│     Output: [8, 64, 1024] predicted velocity                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  8. COMPUTE LOSS & BACKPROP (Line 179)                              │
│     loss = MSE(output, targets).mean()                              │
│                                                                       │
│     Gradients flow back through:                                     │
│     ├─▶ ControlNet layers (TRAINABLE)                               │
│     ├─▶ MLP Projection in BIOT projector (TRAINABLE)               │
│     └─▶ BIOT Encoder (IF freeze_biot=False, else stopped)          │
│                                                                       │
│     Optimizer updates:                                               │
│     - ControlNet weights                                             │
│     - EEG projector weights (MLP layers)                            │
│     - Optionally BIOT encoder weights                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                         TRAINING STEP COMPLETE

```

---

## 🎨 Inference Pipeline (Sample Generation)

During validation, samples are generated to log to W&B:

```
┌─────────────────────────────────────────────────────────────────────┐
│  INFERENCE: SampleLogger.log_sample()                               │
│  (module_controlnet_eeg.py:285)                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. LOAD VALIDATION BATCH                                            │
│     x, eeg, prompts, start_seconds, total_seconds = batch           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. PROJECT EEG ⭐ (Line 300)                                       │
│     projected = pl_module.eeg_projector(eeg[:num_samples])          │
│                                                                       │
│     SAME BIOT FORWARD PASS AS TRAINING!                             │
│     [num_samples, 32, 1280] → [num_samples, 2, 441000]            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. BUILD CONDITIONING (Line 301-306)                               │
│     conditioning = _build_conditioning(projected, prompts, ...)     │
│                                                                       │
│     BIOT output becomes "audio" control signal                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. GENERATE AUDIO (Line 325)                                        │
│     output = generate_diffusion_cond(                               │
│         model=pl_module.model,                                      │
│         conditioning=conditioning,  ← BIOT-based control            │
│         steps=100,                                                  │
│         cfg_scale=7.0,                                              │
│     )                                                                │
│                                                                       │
│     Iterative denoising process:                                     │
│     ├─▶ Start with random noise [num_samples, 64, 1024]           │
│     ├─▶ For 100 steps:                                             │
│     │   ├─▶ ControlNet predicts velocity                           │
│     │   │   (guided by BIOT EEG features)                          │
│     │   ├─▶ Apply classifier-free guidance                         │
│     │   └─▶ Step to next noise level                               │
│     └─▶ Decode final latent to audio                               │
│                                                                       │
│     Output: [num_samples, 2, 441000] generated audio               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. LOG TO WEIGHTS & BIASES                                          │
│     - Target audio                                                   │
│     - Generated audio                                                │
│     - Spectrograms                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Contributions of BIOT

### 1. **Rich Feature Extraction**

**Without BIOT (Simple Projector):**
```
EEG [32, 1280] → Normalize → Conv1d(32→128→2) → [2, 441000]
```
- Only learns linear transformations
- No frequency analysis
- No pre-trained knowledge

**With BIOT:**
```
EEG [32, 1280] 
  → STFT (frequency decomposition)
  → Transformer (cross-channel attention, pre-trained on 5M samples)
  → [256] rich semantic features
  → MLP projection
  → [2, 441000]
```
- **Frequency-domain understanding**: STFT captures spectral patterns
- **Cross-channel reasoning**: Transformer learns relationships between channels
- **Pre-trained knowledge**: Leverages patterns from millions of EEG samples
- **Semantic understanding**: 256-dim embedding captures high-level EEG states

### 2. **Pre-trained Knowledge Transfer**

BIOT encoder was pre-trained on:
- 5M+ EEG samples from clinical datasets
- Multiple EEG recording scenarios (sleep, resting, seizure detection)
- Diverse subjects and conditions

This pre-training teaches BIOT to understand:
- Normal vs abnormal EEG patterns
- Emotional states (valence, arousal)
- Cognitive states (attention, relaxation)
- Individual differences

**Your music generation benefits** from this knowledge without needing to learn from scratch!

### 3. **Trainable Components**

```
┌─────────────────────────────────────────────────────────────────┐
│                      PARAMETER UPDATES                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ALWAYS TRAINABLE:                                               │
│  ├─▶ ControlNet layers (~50M params)                            │
│  └─▶ MLP Projection (~200K params)                              │
│                                                                  │
│  OPTIONALLY TRAINABLE (freeze_biot setting):                    │
│  └─▶ BIOT Encoder (~2.2M params)                                │
│                                                                  │
│  FROZEN (never updated):                                         │
│  ├─▶ Stable Audio base model (~500M params)                     │
│  ├─▶ VAE encoder/decoder                                        │
│  └─▶ Text conditioner                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**When `freeze_biot=True` (Recommended):**
- BIOT encoder stays frozen with pre-trained weights
- Only MLP projection learns to map BIOT features → audio control
- Fast training, stable convergence
- Good generalization

**When `freeze_biot=False`:**
- BIOT encoder fine-tunes on your specific task
- Adapts to music-listening EEG patterns
- Slower training, needs more data
- Potentially better task-specific performance

---

## 🔄 Comparison: Simple vs BIOT vs Hybrid

### Simple Projector Flow
```
EEG [8, 32, 1280]
  ↓ normalize
  ↓ interpolate to [8, 32, 441000]
  ↓ Conv1d(32 → 128)
  ↓ GELU
  ↓ Conv1d(128 → 2)
  ↓ tanh
Control [8, 2, 441000]

Features learned:
- Simple linear mappings
- Direct time-domain processing
- No pre-trained knowledge
```

### BIOT Projector Flow
```
EEG [8, 32, 1280]
  ↓ For each channel: STFT → [8, 101 freq, 13 time]
  ↓ Frequency embedding → [8, 13, 256]
  ↓ Channel tokens + positional encoding
  ↓ Concatenate channels → [8, 416 tokens, 256]
  ↓ Linear Attention Transformer (4 layers, pre-trained)
  ↓ Mean pooling → [8, 256]
  ↓ MLP(256 → 512 → 512 → 882000)
  ↓ Reshape to [8, 2, 441000]
  ↓ tanh
Control [8, 2, 441000]

Features learned:
- Frequency-domain patterns (via STFT)
- Cross-channel relationships (via Transformer)
- Semantic EEG states (via pre-training)
- Audio space mapping (via MLP)
```

### Hybrid Projector Flow
```
EEG [8, 32, 1280]
  ├─▶ BIOT Path:
  │   ↓ STFT + Transformer (pre-trained)
  │   ↓ Global features [8, 256]
  │   ↓ Project to [8, 128]
  │   ↓ Broadcast to [8, 128, 441000]
  │
  └─▶ Local Path:
      ↓ Normalize + interpolate to [8, 32, 441000]
      ↓ Conv1d(32 → 128)
      ↓ BatchNorm + GELU
      ↓ Conv1d(128 → 128)
      ↓ BatchNorm + GELU
      ↓ Local features [8, 128, 441000]

  Combine:
  ↓ Concatenate [8, 256, 441000]
  ↓ Conv1d(256 → 128)
  ↓ GELU
  ↓ Conv1d(128 → 2)
  ↓ tanh
Control [8, 2, 441000]

Features learned:
- Global semantic understanding (BIOT)
- Local temporal details (Conv1d)
- Fusion of both perspectives
- Best of both worlds!
```

---

## 📈 Information Flow Summary

```
┌──────────────────┐
│   DEAP Dataset   │
│   EEG Signals    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     Pre-trained on      ┌─────────────────┐
│  BIOT Encoder    │◀─────5M+ samples────────│  Clinical EEG   │
│  (Frozen)        │                          │  Datasets       │
└────────┬─────────┘                          └─────────────────┘
         │
         │ 256-dim semantic features
         │ (emotional states, patterns)
         ▼
┌──────────────────┐
│  MLP Projection  │
│  (Trainable)     │
└────────┬─────────┘
         │
         │ 2-channel audio-like control
         │
         ▼
┌──────────────────┐     Pre-trained         ┌─────────────────┐
│   ControlNet     │◀─────on audio───────────│  Stable Audio   │
│   (Trainable)    │                          │  Model          │
└────────┬─────────┘                          └─────────────────┘
         │
         │ Guided generation
         │
         ▼
┌──────────────────┐
│ Generated Music  │
│ (reflects EEG    │
│  emotional state)│
└──────────────────┘
```

---

## 💡 Key Takeaways

1. **BIOT forward() is called twice per batch**:
   - Once during training step for loss computation
   - Once during validation for sample generation

2. **BIOT contributes pre-trained EEG understanding**:
   - Transforms raw EEG → semantic embeddings
   - Embeddings capture emotional/cognitive states
   - Learned from millions of clinical EEG samples

3. **The MLP projection is the critical adapter**:
   - Maps BIOT's 256-dim space → audio control space
   - Learns task-specific transformation (EEG → music)
   - Always trainable, enables end-to-end learning

4. **ControlNet uses BIOT output as guidance**:
   - EEG control signal influences every denoising step
   - Steers generation toward EEG-appropriate music
   - Learns to interpret BIOT features for music generation

5. **Training is efficient**:
   - Most parameters frozen (Stable Audio base)
   - BIOT optionally frozen (recommended)
   - Only ControlNet + projection trained
   - Fast convergence due to pre-training

---

## 🎯 The "Magic" of BIOT Integration

The key insight is **knowledge transfer**:

```
Clinical EEG Understanding  ──┐
(seizures, sleep, emotions)   │
                               ├─▶ BIOT Pre-training
Millions of samples            │
from diverse subjects      ────┘
                                    │
                                    │ Transfer
                                    ▼
                            Your Music Generation
                            (learns from BIOT features)
                                    │
                                    │ Specialization
                                    ▼
                            EEG → Music Mapping
                            (task-specific adaptation)
```

You don't need millions of music-EEG pairs! You leverage BIOT's understanding of EEG patterns and only learn the final mapping to music.

---

This is the complete picture of how BIOT contributes to your EEG-to-music system! 🎵🧠

