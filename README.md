# CycleVAR

This repository is a practical reimplementation of key ideas from
**CycleVAR (arXiv:2506.23347)**, built by combining components from
CycleGAN-Turbo (`src/`) and VAR (`models/`, `utils/`).

## Core Files

- `dist.py`
  - Distributed utility module used by VAR code.
- `src/cyclevar.py`
  - Main CycleVAR logic:
    - Multi-scale source token prefill
    - Softmax Relaxed Quantization (SRQ)
    - `parallel one-step` / `serial multi-step` generation
    - Differentiable tokenization path (optional STE) for cycle loss training
- `src/train_cyclevar.py`
  - Unpaired image translation training script
  - Losses: `L_cycle + L_gan + L_idt` (LPIPS optional)
- `src/inference_cyclevar.py`
  - Inference script for folder-based image translation from a trained checkpoint

## Dataset Layout

Expected structure:

```text
DATASET_ROOT/
  train_A/
  train_B/
```

## Training Example

```bash
python src/train_cyclevar.py \
  --dataset_folder /path/to/dataset \
  --output_dir ./outputs/cyclevar \
  --vae_ckpt ./vae_ch160v4096z32.pth \
  --var_ckpt /path/to/pretrained_var.pt \
  --label_src 339 \
  --label_tgt 340 \
  --generation_mode parallel \
  --batch_size 2 \
  --max_steps 20000
```

## Inference Example

```bash
python src/inference_cyclevar.py \
  --checkpoint ./outputs/cyclevar/cyclevar_last.pt \
  --input_dir /path/to/test_A \
  --output_dir ./outputs/cyclevar/test_A_to_B \
  --target_label 340 \
  --generation_mode parallel \
  --image_prep resize_256
```

## Important Options

- `--generation_mode {parallel,serial}`
  - Two generation modes from the paper.
- `--alpha`
  - Fusion ratio between source feature and predicted feature.
- `--srq_temperature`
  - Softmax temperature for SRQ.
- `--tokenize_temperature`
  - Temperature used in tokenization-side relaxed/hard assignment.
- `--disable_tokenizer_ste`
  - Disables tokenizer STE.

## Checkpoint Contents

`src/train_cyclevar.py` saves:

- `cyclevar` (full model state)
- `var`, `vae`
- `disc_x`, `disc_y`
- `opt_gen`, `opt_disc`
- `step`, `epoch`, `args`
