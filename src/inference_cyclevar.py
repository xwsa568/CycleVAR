import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from cyclevar import CycleVAR
from my_utils.training_utils import build_transform


def _default_infinity_root() -> Path:
    env_root = os.environ.get("INFINITY_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return (Path(__file__).resolve().parents[2] / "Infinity").resolve()


def _preflight_infinity(args):
    required = {
        "infinity_model_path": args.infinity_model_path,
        "infinity_vae_path": args.infinity_vae_path,
        "infinity_text_encoder_ckpt": args.infinity_text_encoder_ckpt,
    }
    for key, path in required.items():
        if path is None:
            raise ValueError(f"--{key} is required when --generator_backbone infinity")
        if not os.path.exists(path):
            raise FileNotFoundError(f"--{key} not found: {path}")

    infinity_root = _default_infinity_root()
    if not infinity_root.exists():
        raise FileNotFoundError(f"Infinity repository not found at {infinity_root}")
    if str(infinity_root) not in sys.path:
        sys.path.insert(0, str(infinity_root))

    try:
        importlib.import_module("flash_attn")
    except Exception as exc:
        raise RuntimeError("flash_attn import failed. Install FlashAttention for Infinity inference.") from exc

    try:
        importlib.import_module("infinity")
    except Exception as exc:
        raise RuntimeError("infinity import failed. Check Infinity dependencies and INFINITY_ROOT.") from exc


def _read_prompt(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _resolve_prompts(args):
    prompt_a = args.prompt_a
    prompt_b = args.prompt_b
    if args.dataset_folder:
        if prompt_a is None:
            prompt_a = _read_prompt(os.path.join(args.dataset_folder, "fixed_prompt_a.txt"))
        if prompt_b is None:
            prompt_b = _read_prompt(os.path.join(args.dataset_folder, "fixed_prompt_b.txt"))
    return prompt_a, prompt_b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned CycleVAR checkpoint")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--image_prep", type=str, default="no_resize", help="Image prep in training_utils.build_transform")
    parser.add_argument("--direction", type=str, required=True, choices=["a2b", "b2a"])
    parser.add_argument("--generator_backbone", type=str, default="var", choices=["var", "infinity"])

    parser.add_argument("--vqvae_ckpt_path", type=str, default=None, help="Optional VQVAE ckpt path")
    parser.add_argument("--var_ckpt_path", type=str, default=None, help="Optional VAR ckpt path")
    parser.add_argument("--var_patch_nums", type=str, default="1,2,3,4,5,6,8,10,13,16")
    parser.add_argument("--var_depth", type=int, default=16)
    parser.add_argument("--var_num_classes", type=int, default=1000)
    parser.add_argument("--label_a", type=int, default=0)
    parser.add_argument("--label_b", type=int, default=1)
    parser.add_argument("--srq_temperature", type=float, default=2.0)
    parser.add_argument("--source_temperature", type=float, default=1.0)
    parser.add_argument("--use_srq_gumbel", action="store_true")
    parser.add_argument("--src_fusion_alpha", type=float, default=1.0)
    parser.add_argument("--prompt_a", type=str, default=None)
    parser.add_argument("--prompt_b", type=str, default=None)
    parser.add_argument("--dataset_folder", type=str, default=None)
    parser.add_argument("--infinity_model_path", type=str, default=None)
    parser.add_argument("--infinity_vae_path", type=str, default=None)
    parser.add_argument("--infinity_text_encoder_ckpt", type=str, default=None)
    parser.add_argument(
        "--infinity_model_type",
        type=str,
        default="infinity_2b",
        choices=[
            "infinity_2b",
            "infinity_layer12",
            "infinity_layer16",
            "infinity_layer24",
            "infinity_layer32",
            "infinity_layer40",
            "infinity_layer48",
        ],
    )
    parser.add_argument("--infinity_pn", type=str, default="0.06M", choices=["0.06M", "0.25M", "0.60M", "1M"])
    parser.add_argument("--infinity_rope2d_each_sa_layer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--infinity_rope2d_normalized_by_hw", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--infinity_add_lvl_embeding_only_first_block", type=int, default=1, choices=[0, 1])
    parser.add_argument("--infinity_use_bit_label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--infinity_apply_spatial_patchify", type=int, default=0, choices=[0, 1])
    parser.add_argument("--infinity_use_flex_attn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--hard_decode", action="store_true", help="Use argmax decode instead of SRQ at inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.generator_backbone == "infinity":
        _preflight_infinity(args)
        prompt_a, prompt_b = _resolve_prompts(args)
        from cycleinfinity import CycleInfinity

        model = CycleInfinity(
            infinity_model_path=args.infinity_model_path,
            infinity_vae_path=args.infinity_vae_path,
            infinity_text_encoder_ckpt=args.infinity_text_encoder_ckpt,
            cycleinfinity_ckpt_path=args.model_path,
            model_type=args.infinity_model_type,
            pn=args.infinity_pn,
            rope2d_each_sa_layer=args.infinity_rope2d_each_sa_layer,
            rope2d_normalized_by_hw=args.infinity_rope2d_normalized_by_hw,
            add_lvl_embeding_only_first_block=args.infinity_add_lvl_embeding_only_first_block,
            use_bit_label=args.infinity_use_bit_label,
            apply_spatial_patchify=args.infinity_apply_spatial_patchify,
            use_flex_attn=args.infinity_use_flex_attn,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            srq_temperature=args.srq_temperature,
            source_temperature=args.source_temperature,
            use_srq_gumbel=args.use_srq_gumbel,
            src_fusion_alpha=args.src_fusion_alpha,
        )
    else:
        model = CycleVAR(
            vqvae_ckpt_path=args.vqvae_ckpt_path,
            var_ckpt_path=args.var_ckpt_path,
            cyclevar_ckpt_path=args.model_path,
            patch_nums=args.var_patch_nums,
            var_depth=args.var_depth,
            num_classes=args.var_num_classes,
            label_a=args.label_a,
            label_b=args.label_b,
            srq_temperature=args.srq_temperature,
            source_temperature=args.source_temperature,
            use_srq_gumbel=args.use_srq_gumbel,
            src_fusion_alpha=args.src_fusion_alpha,
        )
    model.eval().to(device)

    t_val = build_transform(args.image_prep)

    input_image = Image.open(args.input_image).convert("RGB")
    with torch.no_grad():
        input_img = t_val(input_image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).to(device)
        output = model(x_t, direction=args.direction, hard_decode=args.hard_decode)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

    os.makedirs(args.output_dir, exist_ok=True)
    output_pil.save(os.path.join(args.output_dir, os.path.basename(args.input_image)))
