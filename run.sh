#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run.sh
# Optional env:
#   CUDA_VISIBLE_DEVICES=1
#   GEN_BACKBONE=var      # var|infinity (default: var)
#   PRECHECK=1            # run 1-step NaN precheck before full training (default: 1)
#   MIXED_PRECISION=no    # no|fp16|bf16 (default: no)
#   TRAIN_STEPS=20000
#   TRAIN_BATCH=2

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
GEN_BACKBONE="${GEN_BACKBONE:-var}"
PRECHECK="${PRECHECK:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
INFINITY_MODEL_TYPE="${INFINITY_MODEL_TYPE:-infinity_2b}"
INFINITY_PN="${INFINITY_PN:-0.06M}"

DATASET="./datasets/my_horse2zebra"
if [[ "${GEN_BACKBONE}" == "infinity" ]]; then
  OUT_ROOT="${OUT_ROOT:-./outputs/cycleinfinity_h2z}"
  PROJECT="${PROJECT:-cycleinfinity_h2z}"
else
  OUT_ROOT="${OUT_ROOT:-./outputs/cyclevar_h2z}"
  PROJECT="${PROJECT:-cyclevar_h2z}"
fi

COMMON_ARGS=(
  --dataset_folder "${DATASET}"
  --train_img_prep resize_286_randomcrop_256x256_hflip
  --val_img_prep no_resize
  --generator_backbone "${GEN_BACKBONE}"
  --train_batch_size "${TRAIN_BATCH}"
  --seed 42
  --report_to wandb
  --mixed_precision "${MIXED_PRECISION}"
  --no-gan_diffaug
  --gan_output_type pool
)

if [[ "${GEN_BACKBONE}" == "var" ]]; then
  COMMON_ARGS+=(
    --vqvae_ckpt_path ./ckpts/vae_ch160v4096z32.pth
    --var_ckpt_path ./ckpts/var_d16.pth
    --label_a 339
    --label_b 340
  )
elif [[ "${GEN_BACKBONE}" == "infinity" ]]; then
  : "${INFINITY_MODEL_PATH:?Set INFINITY_MODEL_PATH to Infinity transformer checkpoint path}"
  : "${INFINITY_VAE_PATH:?Set INFINITY_VAE_PATH to Infinity VAE checkpoint path}"
  : "${INFINITY_T5_PATH:?Set INFINITY_T5_PATH to text encoder/tokenizer checkpoint path}"
  COMMON_ARGS+=(
    --infinity_model_path "${INFINITY_MODEL_PATH}"
    --infinity_vae_path "${INFINITY_VAE_PATH}"
    --infinity_text_encoder_ckpt "${INFINITY_T5_PATH}"
    --infinity_model_type "${INFINITY_MODEL_TYPE}"
    --infinity_pn "${INFINITY_PN}"
    --infinity_rope2d_each_sa_layer 1
    --infinity_rope2d_normalized_by_hw 2
    --infinity_add_lvl_embeding_only_first_block 1
    --infinity_use_bit_label 1
    --infinity_apply_spatial_patchify 0
    --infinity_use_flex_attn 0
  )
  if [[ -n "${PROMPT_A:-}" ]]; then
    COMMON_ARGS+=(--prompt_a "${PROMPT_A}")
  fi
  if [[ -n "${PROMPT_B:-}" ]]; then
    COMMON_ARGS+=(--prompt_b "${PROMPT_B}")
  fi
else
  echo "[run] unsupported GEN_BACKBONE=${GEN_BACKBONE} (expected var|infinity)"
  exit 1
fi

if [[ "${PRECHECK}" == "1" ]]; then
  PRECHECK_OUT="${OUT_ROOT}_precheck"
  rm -rf "${PRECHECK_OUT}"
  echo "[run] precheck start (1 step)"
  python src/train_cyclevar.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${PRECHECK_OUT}" \
    --tracker_project_name "${PROJECT}_precheck" \
    --max_train_steps 1 \
    --debug_nan_guard \
    --debug_nan_abort \
    --debug_nan_every 1 \
    --debug_disc_state_scan \
    --debug_disc_scan_max_keys 20 \
    --debug_disc_scan_save_jsonl
  echo "[run] precheck done"
fi

mkdir -p "${OUT_ROOT}"
echo "[run] train start (backbone=${GEN_BACKBONE}, steps=${TRAIN_STEPS}, batch=${TRAIN_BATCH}, mp=${MIXED_PRECISION}, gpu=${CUDA_VISIBLE_DEVICES})"
python src/train_cyclevar.py \
  "${COMMON_ARGS[@]}" \
  --output_dir "${OUT_ROOT}" \
  --tracker_project_name "${PROJECT}" \
  --learning_rate 1e-6 \
  --lambda_cycle 1.0 \
  --lambda_cycle_lpips 10.0 \
  --use_srq_gumbel \
  --max_train_steps "${TRAIN_STEPS}"
echo "[run] train done"
