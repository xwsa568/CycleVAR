import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from cyclevar import CycleVAR
from my_utils.training_utils import build_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned CycleVAR checkpoint")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--image_prep", type=str, default="no_resize", help="Image prep in training_utils.build_transform")
    parser.add_argument("--direction", type=str, required=True, choices=["a2b", "b2a"])

    parser.add_argument("--vqvae_ckpt_path", type=str, default=None, help="Optional VQVAE ckpt path")
    parser.add_argument("--var_ckpt_path", type=str, default=None, help="Optional VAR ckpt path")
    parser.add_argument("--var_patch_nums", type=str, default="1,2,3,4,5,6,8,10,13,16")
    parser.add_argument("--var_depth", type=int, default=16)
    parser.add_argument("--var_num_classes", type=int, default=1000)
    parser.add_argument("--label_a", type=int, default=0)
    parser.add_argument("--label_b", type=int, default=1)
    parser.add_argument("--hard_decode", action="store_true", help="Use argmax decode instead of SRQ at inference")
    args = parser.parse_args()

    model = CycleVAR(
        vqvae_ckpt_path=args.vqvae_ckpt_path,
        var_ckpt_path=args.var_ckpt_path,
        cyclevar_ckpt_path=args.model_path,
        patch_nums=args.var_patch_nums,
        var_depth=args.var_depth,
        num_classes=args.var_num_classes,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    model.eval().cuda()

    t_val = build_transform(args.image_prep)

    input_image = Image.open(args.input_image).convert("RGB")
    with torch.no_grad():
        input_img = t_val(input_image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
        output = model(x_t, direction=args.direction, hard_decode=args.hard_decode)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

    os.makedirs(args.output_dir, exist_ok=True)
    output_pil.save(os.path.join(args.output_dir, os.path.basename(args.input_image)))
