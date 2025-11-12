"""
Lightweight inference script for Stable Audio ControlNet on low-resource hardware.
Optimized for 4GB VRAM GPUs like RTX 3050.

Usage:
    python inference_low_resource.py \
        --checkpoint ckpts/musdb-audio/checkpoint.ckpt \
        --input_audio input.wav \
        --output_dir outputs \
        --depth_factor 0.1 \
        --steps 25

Author: Optimized for low-resource systems
"""

import argparse
import os
import gc
from pathlib import Path

import torch
import torchaudio
from stable_audio_tools.inference.generation import generate_diffusion_cond
from main.controlnet.pretrained import get_pretrained_controlnet_model


def clear_memory():
    """Clear GPU and system memory."""
    gc.collect()
    torch.cuda.empty_cache()


def load_audio(audio_path, sample_rate=44100, max_duration=None):
    """Load and preprocess audio file."""
    print(f"Loading audio from {audio_path}...")
    audio, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != sample_rate:
        print(f"Resampling from {sr}Hz to {sample_rate}Hz...")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Trim to max duration if specified
    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if audio.shape[1] > max_samples:
            print(f"Trimming audio to {max_duration} seconds...")
            audio = audio[:, :max_samples]
    
    # Normalize to [-1, 1]
    audio = torch.clamp(audio, -1, 1)
    
    return audio


def load_model(checkpoint_path=None, depth_factor=0.1, device="cuda"):
    """Load ControlNet model with memory optimizations."""
    print(f"Loading model with depth_factor={depth_factor}...")
    
    # Load base model
    model, model_config = get_pretrained_controlnet_model(
        "stabilityai/stable-audio-open-1.0",
        controlnet_types=["audio"],
        depth_factor=depth_factor
    )
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        # Remove 'model.' prefix if present (from PyTorch Lightning)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded successfully!")
    else:
        print("No checkpoint provided or file not found. Using base model.")
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    # Freeze unnecessary parts to save memory
    model.model.model.requires_grad_(False)
    model.conditioner.requires_grad_(False)
    model.pretransform.requires_grad_(False)
    
    clear_memory()
    
    return model, model_config


def generate_audio(
    model,
    conditioning_audio,
    prompt="",
    steps=25,
    cfg_scale=5.0,
    seed=42,
    device="cuda"
):
    """Generate audio with memory-efficient settings."""
    print(f"Generating audio with {steps} steps, cfg_scale={cfg_scale}...")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Prepare conditioning
    conditioning_audio = conditioning_audio.to(device)
    
    conditioning = [{
        "audio": conditioning_audio,
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": conditioning_audio.shape[-1] / 44100,
    }]
    
    # Generate with memory optimizations
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
            output = generate_diffusion_cond(
                model.model,
                seed=seed,
                batch_size=1,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=conditioning_audio.shape[-1],
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=device
            )
    
    clear_memory()
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight inference for Stable Audio ControlNet"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)"
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for generated audio"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for generation (optional)"
    )
    parser.add_argument(
        "--depth_factor",
        type=float,
        default=0.1,
        help="ControlNet depth factor (0.1 for low VRAM, 0.2-0.5 for more capacity)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of diffusion steps (25-50 for low VRAM, 100 for quality)"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (5.0 for low VRAM, 7.0 for quality)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=10.0,
        help="Maximum audio duration in seconds (10 for low VRAM, 30+ for more)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--no_mix",
        action="store_true",
        help="Don't save mixed audio (input + output)"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Print GPU info
    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, model_config = load_model(
        checkpoint_path=args.checkpoint,
        depth_factor=args.depth_factor,
        device=args.device
    )
    
    # Load input audio
    input_audio = load_audio(
        args.input_audio,
        sample_rate=model_config["sample_rate"],
        max_duration=args.max_duration
    )
    
    print(f"Input audio shape: {input_audio.shape}")
    print(f"Duration: {input_audio.shape[-1] / model_config['sample_rate']:.2f} seconds")
    
    # Print memory usage before generation
    if args.device == "cuda":
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Generate audio
    try:
        output_audio = generate_audio(
            model=model,
            conditioning_audio=input_audio,
            prompt=args.prompt,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            device=args.device
        )
        
        print("Generation successful!")
        
        # Print memory usage after generation
        if args.device == "cuda":
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ OUT OF MEMORY ERROR!")
            print("\nTry these solutions:")
            print("1. Reduce --depth_factor (e.g., 0.05)")
            print("2. Reduce --steps (e.g., 10)")
            print("3. Reduce --max_duration (e.g., 5)")
            print("4. Close other applications")
            print("5. Use --device cpu (much slower)")
            return
        else:
            raise e
    
    # Save outputs
    input_filename = Path(args.input_audio).stem
    sample_rate = model_config["sample_rate"]
    
    # Save input
    input_path = os.path.join(args.output_dir, f"{input_filename}_input.wav")
    torchaudio.save(input_path, input_audio.cpu(), sample_rate)
    print(f"Saved input: {input_path}")
    
    # Save output
    output_path = os.path.join(args.output_dir, f"{input_filename}_output.wav")
    torchaudio.save(output_path, output_audio[0].cpu(), sample_rate)
    print(f"Saved output: {output_path}")
    
    # Save mix
    if not args.no_mix:
        mix_audio = input_audio + output_audio[0].cpu()
        mix_path = os.path.join(args.output_dir, f"{input_filename}_mix.wav")
        torchaudio.save(mix_path, mix_audio, sample_rate)
        print(f"Saved mix: {mix_path}")
    
    print("\n✅ Inference complete!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

