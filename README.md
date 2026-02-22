# CycleVAR

`src`는 CycleGAN-Turbo 기반, `models/utils/train.py/trainer.py`는 VAR 기반으로 가져온 상태에서,
논문 **CycleVAR (arXiv:2506.23347)** 의 핵심 아이디어를 반영해 재구현한 코드가 포함되어 있습니다.

## 추가된 핵심 파일

- `dist.py`
  - VAR 코드에서 사용하는 분산 유틸 모듈.
- `src/cyclevar.py`
  - CycleVAR 핵심 로직:
    - 멀티스케일 source token prefill
    - Softmax Relaxed Quantization (SRQ)
    - `parallel one-step` / `serial multi-step` 생성
    - cycle loss 경로를 위한 differentiable tokenization(STE 옵션)
- `src/train_cyclevar.py`
  - unpaired image translation 학습 스크립트
  - loss: `L_cycle + L_gan + L_idt` (LPIPS 옵션)

## 데이터 폴더 구조

아래 형태를 기본으로 가정합니다.

```text
DATASET_ROOT/
  train_A/
  train_B/
```

## 학습 실행 예시

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

## 추론 실행 예시

```bash
python src/inference_cyclevar.py \
  --checkpoint ./outputs/cyclevar/cyclevar_last.pt \
  --input_dir /path/to/test_A \
  --output_dir ./outputs/cyclevar/test_A_to_B \
  --target_label 340 \
  --generation_mode parallel \
  --image_prep resize_256
```

## 주요 옵션

- `--generation_mode {parallel,serial}`
  - 논문의 두 생성 모드.
- `--alpha`
  - source feature와 predicted feature 융합 가중치.
- `--srq_temperature`
  - SRQ softmax temperature.
- `--tokenize_temperature`
  - tokenizer 쪽 relaxed/hard token temperature.
- `--disable_tokenizer_ste`
  - tokenizer STE 비활성화.

## 체크포인트

`src/train_cyclevar.py`는 아래를 저장합니다.

- `cyclevar` (전체 모델 상태)
- `var`, `vae`
- `disc_x`, `disc_y`
- `opt_gen`, `opt_disc`
- `step`, `epoch`, `args`
