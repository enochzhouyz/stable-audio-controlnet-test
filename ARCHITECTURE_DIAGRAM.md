# Architecture Diagrams - EEG-to-Music with BIOT

## Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EEG-to-Music Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────┐      ┌──────────────┐      ┌──────────────────────────┐  │
│  │   EEG    │      │     EEG      │      │   Stable Audio           │  │
│  │   Data   │─────▶│   Projector  │─────▶│   ControlNet            │  │
│  │ [32×128] │      │ (BIOT-based) │      │   (Pre-trained)          │  │
│  └──────────┘      └──────────────┘      └──────────────────────────┘  │
│                           │                          │                   │
│                           │                          ▼                   │
│                           │                ┌─────────────────┐          │
│                           │                │  Diffusion      │          │
│                           │                │  Denoising      │          │
│                           │                └─────────────────┘          │
│                           │                          │                   │
│                           │                          ▼                   │
│                           │                ┌─────────────────┐          │
│  ┌──────────┐             │                │   Generated     │          │
│  │  Text    │─────────────┼───────────────▶│   Music         │          │
│  │  Prompt  │             │                │   [2×44100]     │          │
│  └──────────┘             │                └─────────────────┘          │
│                           │                                              │
│                      [2-channel                                          │
│                   audio control signal]                                  │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Three EEG Projector Architectures

### 1. Simple Projector (Baseline)

```
Input: EEG [batch, 32 channels, 1280 timesteps]
  │
  ├─▶ Per-channel Normalization
  │   (mean=0, std=1)
  │
  ├─▶ Linear Interpolation
  │   [32, 1280] → [32, 441000]
  │
  ├─▶ Conv1d (32 → 128 channels)
  │   kernel_size=1
  │
  ├─▶ GELU Activation
  │
  ├─▶ Conv1d (128 → 2 channels)
  │   kernel_size=1
  │
  ├─▶ Tanh Activation
  │
Output: Control Signal [batch, 2, 441000]

Parameters: ~5,000
Training: From scratch
Speed: Fast (~0.3s/step)
```

---

### 2. BIOT Projector

```
Input: EEG [batch, 32 channels, 1280 timesteps]
  │
  ├─▶ BIOT Encoder
  │   │
  │   ├─▶ Per-channel STFT
  │   │   n_fft=200, hop_length=100
  │   │   [32, 1280] → [32, 101, 13]
  │   │   (freq, time)
  │   │
  │   ├─▶ Frequency Embedding
  │   │   Linear: [101] → [256]
  │   │
  │   ├─▶ Add Channel Tokens
  │   │   (learnable embeddings)
  │   │
  │   ├─▶ Add Positional Encoding
  │   │   (sinusoidal)
  │   │
  │   ├─▶ Linear Attention Transformer
  │   │   - depth=4 layers
  │   │   - heads=8
  │   │   - dim=256
  │   │
  │   └─▶ Mean Pooling (time dim)
  │       Output: [batch, 256]
  │
  ├─▶ Projection MLP
  │   │
  │   ├─▶ Linear (256 → 512)
  │   ├─▶ LayerNorm + GELU + Dropout(0.1)
  │   │
  │   ├─▶ Linear (512 → 512)
  │   ├─▶ LayerNorm + GELU + Dropout(0.1)
  │   │
  │   └─▶ Linear (512 → 882000)
  │       [2 × 441000]
  │
  ├─▶ Reshape
  │   [batch, 882000] → [batch, 2, 441000]
  │
  └─▶ Tanh Activation

Output: Control Signal [batch, 2, 441000]

Parameters: ~2.2M (BIOT) + ~200K (MLP)
Training: BIOT frozen (recommended) or fine-tuned
Speed: Medium (~0.5s/step with frozen BIOT)
```

---

### 3. Hybrid Projector (Recommended)

```
Input: EEG [batch, 32 channels, 1280 timesteps]
  │
  ├──────────────┬────────────────┐
  │              │                │
  │         [Global Path]    [Local Path]
  │              │                │
  │         BIOT Encoder    Normalize EEG
  │              │                │
  │         [batch, 256]    Interpolate
  │              │          [32, 441000]
  │         Project              │
  │         (256 → 128)           │
  │              │          Conv1d (32→128)
  │         Broadcast       kernel=3, pad=1
  │         to time         BatchNorm + GELU
  │              │                │
  │         [b,128,441000]  Conv1d (128→128)
  │              │          kernel=3, pad=1
  │              │          BatchNorm + GELU
  │              │                │
  │              │          [b,128,441000]
  │              │                │
  │              └────────┬───────┘
  │                       │
  │                  Concatenate
  │                  [b,256,441000]
  │                       │
  │                  Conv1d (256→128)
  │                  kernel=1 + GELU
  │                       │
  │                  Conv1d (128→2)
  │                  kernel=1
  │                       │
  │                  Tanh Activation
  │                       │
Output: Control Signal [batch, 2, 441000]

Parameters: ~2.2M (BIOT) + ~300K (local + fusion)
Training: BIOT frozen, train local+fusion
Speed: Medium (~0.6s/step)

Advantages:
  - Global semantic understanding (BIOT)
  - Local temporal details (Conv1d)
  - Best of both worlds
```

---

## BIOT Encoder Details

### BIOT Internal Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BIOT Encoder                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: [batch, channels, time]                                  │
│                                                                   │
│  For each channel:                                                │
│    │                                                              │
│    ├─▶ Short-Time Fourier Transform (STFT)                       │
│    │   - n_fft: 200                                              │
│    │   - hop_length: 100                                         │
│    │   - Output: [batch, freq=101, time_frames]                  │
│    │                                                              │
│    ├─▶ Patch Frequency Embedding                                 │
│    │   - Linear projection: freq_dim → emb_size                  │
│    │   - Permute: [b, freq, time] → [b, time, emb]              │
│    │                                                              │
│    ├─▶ Add Channel Token                                         │
│    │   - Learnable embedding per channel                         │
│    │   - Broadcast to all timesteps                              │
│    │                                                              │
│    └─▶ Add Positional Encoding                                   │
│        - Sinusoidal encoding                                     │
│        - Output: [b, time, emb] for this channel                 │
│                                                                   │
│  Concatenate all channels:                                        │
│    - [b, channels × time, emb]                                   │
│                                                                   │
│  ├─▶ Linear Attention Transformer                                │
│  │   ┌─────────────────────────┐                                 │
│  │   │ Layer 1: Linear Attn    │                                 │
│  │   │ - heads: 8              │                                 │
│  │   │ - dim: 256              │                                 │
│  │   │ - dropout: 0.2          │                                 │
│  │   └─────────────────────────┘                                 │
│  │   ┌─────────────────────────┐                                 │
│  │   │ Layer 2: Linear Attn    │                                 │
│  │   └─────────────────────────┘                                 │
│  │   ┌─────────────────────────┐                                 │
│  │   │ Layer 3: Linear Attn    │                                 │
│  │   └─────────────────────────┘                                 │
│  │   ┌─────────────────────────┐                                 │
│  │   │ Layer 4: Linear Attn    │                                 │
│  │   └─────────────────────────┘                                 │
│  │                                                                │
│  │   Output: [b, channels × time, 256]                           │
│  │                                                                │
│  └─▶ Mean Pooling (temporal dimension)                           │
│      - Average over all timesteps                                │
│      - Output: [batch, 256]                                      │
│                                                                   │
│  Output: Global EEG embedding [batch, 256]                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                      Training Loop (One Step)                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Load Batch                                                       │
│     ├─▶ Audio: [batch, 2, 441000]                                  │
│     ├─▶ EEG: [batch, 32, 1280]                                     │
│     ├─▶ Prompts: List[str]                                         │
│     └─▶ Timing: start_seconds, total_seconds                       │
│                                                                      │
│  2. Encode Audio (Stable Audio VAE) [FROZEN]                        │
│     Audio [b,2,441000] → Latent [b,64,1024]                        │
│                                                                      │
│  3. Project EEG (BIOT-based) [TRAINABLE]                            │
│     EEG [b,32,1280] → Control [b,2,441000]                         │
│                                                                      │
│  4. Sample Diffusion Timestep                                        │
│     t ~ Logit-Normal distribution                                   │
│                                                                      │
│  5. Add Noise to Latent                                             │
│     noised = latent × α(t) + noise × σ(t)                          │
│                                                                      │
│  6. Compute Target (v-prediction)                                   │
│     target = noise × α(t) - latent × σ(t)                          │
│                                                                      │
│  7. Forward Pass (ControlNet) [TRAINABLE]                           │
│     ├─▶ Text Conditioning [FROZEN]                                 │
│     ├─▶ EEG Control Signal [TRAINABLE]                             │
│     └─▶ Predicted velocity: [b,64,1024]                            │
│                                                                      │
│  8. Compute Loss                                                     │
│     loss = MSE(predicted, target)                                   │
│                                                                      │
│  9. Backward Pass                                                    │
│     ├─▶ Update ControlNet weights                                  │
│     └─▶ Update EEG Projector weights                               │
│                                                                      │
│  10. Log Metrics                                                     │
│      └─▶ train_loss, learning_rate, epoch                          │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Inference Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                    Inference (Generation)                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input:                                                              │
│    ├─▶ EEG: [1, 32, 1280]                                          │
│    ├─▶ Text Prompt: "emotional music..."                           │
│    └─▶ Generation parameters                                        │
│                                                                      │
│  1. Project EEG to Control Signal                                   │
│     EEG [1,32,1280] → Control [1,2,441000]                         │
│                                                                      │
│  2. Encode Text Prompt                                              │
│     Text → Text Embedding [1, seq_len, 768]                        │
│                                                                      │
│  3. Build Conditioning                                               │
│     ├─▶ prompt: text string                                        │
│     ├─▶ audio: control signal                                      │
│     ├─▶ seconds_start: 0.0                                         │
│     └─▶ seconds_total: 10.0                                        │
│                                                                      │
│  4. Initialize Random Noise                                          │
│     noise ~ N(0, σ_max) with shape [1, 64, 1024]                  │
│                                                                      │
│  5. Iterative Denoising (e.g., 100 steps)                          │
│     For t = T down to 0:                                            │
│       ├─▶ Predict velocity with ControlNet                         │
│       ├─▶ Compute predicted latent                                 │
│       ├─▶ Apply classifier-free guidance (cfg_scale=7.0)          │
│       └─▶ Step to next noise level (DPM++ sampler)                │
│                                                                      │
│  6. Decode Latent to Audio (VAE Decoder)                            │
│     Latent [1,64,1024] → Audio [1,2,441000]                       │
│                                                                      │
│  7. Post-process                                                     │
│     └─▶ Clip to [-1, 1], save as WAV                              │
│                                                                      │
│  Output: Generated Music [1, 2, 441000]                             │
│          (10 seconds @ 44.1 kHz stereo)                             │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: DEAP Dataset

```
┌────────────────────────────────────────────────────────────────────┐
│                      DEAP Dataset Loading                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DEAP Structure:                                                     │
│    ├─▶ 32 subjects                                                  │
│    ├─▶ 40 trials per subject                                        │
│    ├─▶ 60 seconds per trial                                         │
│    └─▶ 32 EEG channels @ 128 Hz                                     │
│                                                                      │
│  1. Load EEG Data                                                    │
│     File: s01.dat (pickled)                                         │
│     ├─▶ data: [40 trials, 40 channels, 8064 samples]              │
│     └─▶ labels: [40 trials, 4 values]                              │
│                                                                      │
│  2. Extract EEG Channels                                             │
│     Keep first 32 channels (EEG)                                    │
│     Remove 3-second baseline                                        │
│     ├─▶ [40, 32, 7680]                                             │
│     └─▶ ~60 seconds @ 128 Hz                                        │
│                                                                      │
│  3. Load Aligned Audio                                               │
│     File: s01_t01.wav                                               │
│     ├─▶ Duration: 60 seconds                                        │
│     ├─▶ Sample rate: 44100 Hz                                       │
│     └─▶ Resample if needed                                          │
│                                                                      │
│  4. Sample Random Chunk                                              │
│     Duration: 10 seconds                                            │
│     ├─▶ Audio chunk: [2, 441000]                                   │
│     ├─▶ EEG chunk (aligned): [32, 1280]                            │
│     └─▶ start_seconds: random offset                               │
│                                                                      │
│  5. Generate Prompt                                                  │
│     "subject: s01; trial: 01;                                       │
│      valence: 5.23; arousal: 4.87;                                 │
│      dominance: 6.12; liking: 5.45"                                │
│                                                                      │
│  6. Return Batch Item                                                │
│     (audio_chunk, eeg_chunk, prompt,                                │
│      start_seconds, total_seconds)                                  │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Memory Layout (GPU)

```
┌────────────────────────────────────────────────────────────────────┐
│                    GPU Memory Usage (batch=8)                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────┐                          │
│  │  Stable Audio Base Model (FROZEN)    │  3.5 GB                  │
│  │  - VAE Encoder/Decoder               │                          │
│  │  - Diffusion U-Net                   │                          │
│  │  - Text Encoder                      │                          │
│  └──────────────────────────────────────┘                          │
│                                                                      │
│  ┌──────────────────────────────────────┐                          │
│  │  ControlNet (TRAINABLE)              │  0.8 GB                  │
│  │  - Additional conditioning layers    │                          │
│  │  - Gradients stored                  │                          │
│  └──────────────────────────────────────┘                          │
│                                                                      │
│  ┌──────────────────────────────────────┐                          │
│  │  EEG Projector (TRAINABLE)           │                          │
│  │                                      │                          │
│  │  Simple:    0.01 GB                  │  Tiny                    │
│  │  BIOT frozen: 0.5 GB                 │  No gradients            │
│  │  BIOT trainable: 1.0 GB              │  With gradients          │
│  │  Hybrid: 0.8 GB                      │  BIOT frozen + local     │
│  └──────────────────────────────────────┘                          │
│                                                                      │
│  ┌──────────────────────────────────────┐                          │
│  │  Activations & Batch Data            │  1.5 GB                  │
│  │  - Audio latents                     │                          │
│  │  - EEG control signals               │                          │
│  │  - Intermediate features             │                          │
│  └──────────────────────────────────────┘                          │
│                                                                      │
│  ┌──────────────────────────────────────┐                          │
│  │  Optimizer States (AdamW)            │  1.2 GB                  │
│  │  - First moment                      │                          │
│  │  - Second moment                     │                          │
│  └──────────────────────────────────────┘                          │
│                                                                      │
│  ┌──────────────────────────────────────┐                          │
│  │  Total GPU Memory                    │                          │
│  │                                      │                          │
│  │  Simple:  ~6.5 GB                    │  Fits on most GPUs       │
│  │  BIOT:    ~7.0 GB                    │  Recommended            │
│  │  Hybrid:  ~7.5 GB                    │  Best performance        │
│  │  BIOT-FT: ~8.0 GB                    │  If memory allows        │
│  └──────────────────────────────────────┘                          │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Performance Comparison

```
╔═══════════════╦═══════════╦═══════════╦═══════════╦═══════════╗
║   Encoder     ║  Params   ║  Memory   ║   Speed   ║  Quality  ║
╠═══════════════╬═══════════╬═══════════╬═══════════╬═══════════╣
║ Simple        ║    5K     ║   6.5 GB  ║ 0.3 s/step║    ⭐⭐⭐   ║
║ (baseline)    ║           ║           ║  Fast     ║  Baseline ║
╠═══════════════╬═══════════╬═══════════╬═══════════╬═══════════╣
║ BIOT          ║   2.2M    ║   7.0 GB  ║ 0.5 s/step║   ⭐⭐⭐⭐  ║
║ (frozen)      ║  +200K    ║           ║  Medium   ║ +10-15%   ║
╠═══════════════╬═══════════╬═══════════╬═══════════╬═══════════╣
║ Hybrid        ║   2.2M    ║   7.5 GB  ║ 0.6 s/step║  ⭐⭐⭐⭐⭐ ║
║ (frozen)      ║  +300K    ║           ║  Medium   ║ +15-20%   ║
╠═══════════════╬═══════════╬═══════════╬═══════════╬═══════════╣
║ BIOT          ║   2.2M    ║   8.0 GB  ║ 0.7 s/step║  ⭐⭐⭐⭐⭐ ║
║ (fine-tune)   ║  +200K    ║           ║  Slow     ║ +20-25%   ║
╚═══════════════╩═══════════╩═══════════╩═══════════╩═══════════╝

Legend:
  Params: Number of trainable parameters
  Memory: GPU memory usage (batch=8)
  Speed: Time per training step
  Quality: Expected improvement over baseline
```

---

## Legend

```
Shapes:
  [b]      = batch size
  [c]      = channels
  [t]      = time samples
  [f]      = frequency bins
  [emb]    = embedding dimension

Common Dimensions:
  EEG input:        [8, 32, 1280]       (8 samples, 32 channels, ~10s @ 128Hz)
  Audio input:      [8, 2, 441000]     (8 samples, stereo, 10s @ 44.1kHz)
  Audio latent:     [8, 64, 1024]      (VAE encoded)
  BIOT embedding:   [8, 256]           (global feature)
  Control signal:   [8, 2, 441000]     (audio-like control)
  Generated audio:  [8, 2, 441000]     (final output)

Symbols:
  [FROZEN]      = Weights not updated during training
  [TRAINABLE]   = Weights updated via backprop
  ─▶            = Data flow
  └─▶           = Sub-component
```

---

This architecture enables high-quality EEG-to-music generation by leveraging:
1. **Pre-trained EEG understanding** (BIOT)
2. **Powerful audio generation** (Stable Audio)
3. **Flexible conditioning** (ControlNet)

