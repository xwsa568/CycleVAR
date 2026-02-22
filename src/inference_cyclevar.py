from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_vae_var  # noqa: E402
from src.cyclevar import CycleVAR  # noqa: E402
from src.my_utils.training_utils import build_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CycleVAR inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_label", type=int, required=True)
    parser.add_argument("--generation_mode", choices=["parallel", "serial"], default="parallel")
    parser.add_argument("--image_prep", type=str, default="resize_256")
    parser.add_argument("--hard_quantization", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_patch_nums(pn: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in pn.replace("-", "_").split("_"))


def list_images(folder: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths: List[Path] = []
    for ext in exts:
        paths.extend(folder.glob(ext))
    return sorted(paths)


def to_tensor(img: Image.Image, transform) -> torch.Tensor:
    x = transform(img.convert("RGB"))
    x = TF.to_tensor(x)
    x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return x


def save_tensor(x: torch.Tensor, out_path: Path):
    # x is [-1, 1], CHW
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    TF.to_pil_image(x).save(out_path)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    pn = train_args.get("pn", "1_2_3_4_5_6_8_10_13_16")
    depth = int(train_args.get("depth", 16))
    num_classes = int(train_args.get("num_classes", 1000))

    vae, var = build_vae_var(
        device=device,
        patch_nums=parse_patch_nums(pn),
        num_classes=num_classes,
        depth=depth,
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=True,
        fused_if_available=True,
    )

    model = CycleVAR(
        vae=vae,
        var=var,
        alpha=float(train_args.get("alpha", 0.7)),
        srq_temperature=float(train_args.get("srq_temperature", 2.0)),
        tokenize_temperature=float(train_args.get("tokenize_temperature", 1.0)),
        use_tokenizer_ste=not bool(train_args.get("disable_tokenizer_ste", False)),
        freeze_tokenizer=True,
    ).to(device)
    model.load_state_dict(ckpt["cyclevar"], strict=False)
    model.eval()

    transform = build_transform(args.image_prep)
    in_paths = list_images(Path(args.input_dir))
    if not in_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for p in in_paths:
            x = to_tensor(Image.open(p), transform).unsqueeze(0).to(device)
            y = model(
                x,
                labels=args.target_label,
                mode=args.generation_mode,
                differentiable_tokenize=False,
                hard_quantization=args.hard_quantization,
            )
            save_tensor(y[0], out_dir / p.name)

    print(f"[done] translated {len(in_paths)} images to {out_dir}")


if __name__ == "__main__":
    main()
