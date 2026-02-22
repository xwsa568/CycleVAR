from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_vae_var  # noqa: E402
from src.cyclevar import CycleVAR  # noqa: E402
from src.my_utils.training_utils import build_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CycleVAR training (unpaired)")

    parser.add_argument("--dataset_folder", type=str, required=True, help="Folder containing train_A/train_B.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_img_prep", type=str, default="resize_286_randomcrop_256x256_hflip")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--vae_ckpt", type=str, default="vae_ch160v4096z32.pth")
    parser.add_argument("--var_ckpt", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--pn", type=str, default="1_2_3_4_5_6_8_10_13_16")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--label_src", type=int, default=0)
    parser.add_argument("--label_tgt", type=int, default=1)

    parser.add_argument("--generation_mode", choices=["parallel", "serial"], default="parallel")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--srq_temperature", type=float, default=2.0)
    parser.add_argument("--tokenize_temperature", type=float, default=1.0)
    parser.add_argument("--disable_tokenizer_ste", action="store_true")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr_gen", type=float, default=1e-5)
    parser.add_argument("--lr_disc", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lambda_gan", type=float, default=0.5)
    parser.add_argument("--lambda_cycle", type=float, default=1.0)
    parser.add_argument("--lambda_idt", type=float, default=1.0)
    parser.add_argument("--lambda_cycle_lpips", type=float, default=10.0)
    parser.add_argument("--lambda_idt_lpips", type=float, default=1.0)
    parser.add_argument("--gan_disc_type", type=str, default="vagan_clip")
    parser.add_argument("--gan_loss_type", type=str, default="multilevel_sigmoid")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def parse_patch_nums(pn: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in pn.replace("-", "_").split("_"))


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths: List[Path] = []
    for ext in exts:
        paths.extend(folder.glob(ext))
    return sorted(paths)


class UnpairedImageDataset(Dataset):
    def __init__(self, root: str, split: str, image_prep: str):
        super().__init__()
        root_path = Path(root)
        self.src = list_images(root_path / f"{split}_A")
        self.tgt = list_images(root_path / f"{split}_B")
        if not self.src:
            raise FileNotFoundError(f"No images found in {root_path / f'{split}_A'}")
        if not self.tgt:
            raise FileNotFoundError(f"No images found in {root_path / f'{split}_B'}")
        self.transform = build_transform(image_prep)

    def __len__(self) -> int:
        return max(len(self.src), len(self.tgt))

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        x = TF.to_tensor(img)
        x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return x

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        src_path = self.src[index % len(self.src)]
        tgt_path = random.choice(self.tgt)
        return {
            "src": self._load(src_path),
            "tgt": self._load(tgt_path),
        }


def build_discriminator(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.gan_disc_type != "vagan_clip":
        raise NotImplementedError(
            f"Unsupported --gan_disc_type={args.gan_disc_type}. "
            "Use 'vagan_clip' (paper setting: CLIP feature extractor + MLP decoder)."
        )

    try:
        import vision_aided_loss  # type: ignore
    except Exception as e:
        raise ImportError(
            "vision_aided_loss is required for paper-faithful discriminator. "
            "Install it in your environment and retry."
        ) from e

    disc = vision_aided_loss.Discriminator(
        cv_type="clip",
        loss_type=args.gan_loss_type,
        device=str(device),
    ).to(device)
    disc.train()
    disc.requires_grad_(True)
    if hasattr(disc, "cv_ensemble"):
        disc.cv_ensemble.requires_grad_(False)
    for name, module in disc.named_modules():
        if "attn" in name and hasattr(module, "fused_attn"):
            module.fused_attn = False
    return disc


def maybe_build_lpips(device: torch.device):
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    net = lpips.LPIPS(net="vgg").to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    prefixes = ("module.", "_orig_mod.", "model.", "generator.", "var.", "vae.")
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        cleaned[nk] = v
    return cleaned


def candidate_state_dicts(ckpt: object) -> List[Dict[str, torch.Tensor]]:
    cands: List[Dict[str, torch.Tensor]] = []
    if isinstance(ckpt, dict):
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            cands.append(ckpt)  # direct state_dict
        for k in ("state_dict", "model", "var", "vae"):
            v = ckpt.get(k)
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                cands.append(v)
        trainer = ckpt.get("trainer")
        if isinstance(trainer, dict):
            for k in ("var_wo_ddp", "var", "vae_local"):
                v = trainer.get(k)
                if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                    cands.append(v)
    return cands


def load_weights(module: nn.Module, path: Optional[str], tag: str):
    if path is None:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"{tag} checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    module_keys = set(module.state_dict().keys())
    best_sd = None
    best_overlap = -1
    for cand in candidate_state_dicts(ckpt):
        cand = normalize_state_dict_keys(cand)
        overlap = len(module_keys.intersection(cand.keys()))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sd = cand
    if best_sd is None:
        raise RuntimeError(f"No compatible state_dict candidates found in {path}")

    missing, unexpected = module.load_state_dict(best_sd, strict=False)
    print(
        f"[load:{tag}] {path} overlap={best_overlap} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def set_requires_grad(modules: Iterable[nn.Module], flag: bool):
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(flag)
        if flag and hasattr(module, "cv_ensemble"):
            module.cv_ensemble.requires_grad_(False)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model,
    lpips_weight: float,
) -> torch.Tensor:
    loss = F.l1_loss(pred, target)
    if lpips_model is not None and lpips_weight > 0:
        loss = loss + lpips_weight * lpips_model(pred, target).mean()
    return loss


def build_constant_with_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    warmup_steps = max(int(warmup_steps), 0)
    if warmup_steps == 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def lr_lambda(cur_step: int):
        if cur_step < warmup_steps:
            return float(cur_step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_training_state(
    path: Path,
    step: int,
    epoch: int,
    model: CycleVAR,
    disc_x: nn.Module,
    disc_y: nn.Module,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    gen_sche,
    disc_sche,
    args: argparse.Namespace,
):
    payload = {
        "step": step,
        "epoch": epoch,
        "cyclevar": model.state_dict(),
        "var": model.var.state_dict(),
        "vae": model.vae.state_dict(),
        "disc_x": disc_x.state_dict(),
        "disc_y": disc_y.state_dict(),
        "opt_gen": gen_opt.state_dict(),
        "opt_disc": disc_opt.state_dict(),
        "sche_gen": gen_sche.state_dict(),
        "sche_disc": disc_sche.state_dict(),
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_nums = parse_patch_nums(args.pn)

    dataset = UnpairedImageDataset(args.dataset_folder, split="train", image_prep=args.train_img_prep)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if len(loader) == 0:
        raise RuntimeError("Empty dataloader. Check batch_size and dataset size.")

    total_steps = args.max_steps if args.max_steps is not None else args.max_epochs * len(loader)
    min_epochs_for_steps = (total_steps + len(loader) - 1) // len(loader)
    epoch_limit = max(args.max_epochs, min_epochs_for_steps)
    print(
        f"[setup] device={device}, total_steps={total_steps}, "
        f"steps/epoch={len(loader)}, epoch_limit={epoch_limit}"
    )

    vae, var = build_vae_var(
        device=device,
        patch_nums=patch_nums,
        num_classes=args.num_classes,
        depth=args.depth,
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=False,
        fused_if_available=True,
    )
    load_weights(vae, args.vae_ckpt, "vae")
    if args.var_ckpt is not None:
        load_weights(var, args.var_ckpt, "var")
    else:
        print("[warn] --var_ckpt is not set. VAR will train from random init.")

    model = CycleVAR(
        vae=vae,
        var=var,
        alpha=args.alpha,
        srq_temperature=args.srq_temperature,
        tokenize_temperature=args.tokenize_temperature,
        use_tokenizer_ste=not args.disable_tokenizer_ste,
        freeze_tokenizer=True,
    ).to(device)
    model.train()

    disc_x = build_discriminator(args, device)
    disc_y = build_discriminator(args, device)

    lpips_model = maybe_build_lpips(device)
    if lpips_model is None:
        print("[warn] lpips is unavailable. LPIPS terms will be skipped.")

    gen_params = [p for p in model.var.parameters() if p.requires_grad]
    gen_opt = torch.optim.AdamW(
        gen_params, lr=args.lr_gen, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
    )
    disc_params = [p for p in list(disc_x.parameters()) + list(disc_y.parameters()) if p.requires_grad]
    disc_opt = torch.optim.AdamW(
        disc_params,
        lr=args.lr_disc,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    gen_sche = build_constant_with_warmup_scheduler(gen_opt, args.lr_warmup_steps)
    disc_sche = build_constant_with_warmup_scheduler(disc_opt, args.lr_warmup_steps)

    start_step = 0
    start_epoch = 0
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume not found: {args.resume}")
        resume = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume["cyclevar"], strict=False)
        disc_x.load_state_dict(resume["disc_x"], strict=False)
        disc_y.load_state_dict(resume["disc_y"], strict=False)
        try:
            gen_opt.load_state_dict(resume["opt_gen"])
        except Exception as e:
            print(f"[warn] failed to load gen optimizer state: {e}")
        try:
            disc_opt.load_state_dict(resume["opt_disc"])
        except Exception as e:
            print(f"[warn] failed to load disc optimizer state: {e}")
        if "sche_gen" in resume:
            try:
                gen_sche.load_state_dict(resume["sche_gen"])
            except Exception as e:
                print(f"[warn] failed to load gen scheduler state: {e}")
        if "sche_disc" in resume:
            try:
                disc_sche.load_state_dict(resume["sche_disc"])
            except Exception as e:
                print(f"[warn] failed to load disc scheduler state: {e}")
        start_step = int(resume.get("step", 0))
        start_epoch = int(resume.get("epoch", 0))
        print(f"[resume] step={start_step}, epoch={start_epoch}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = start_step
    tic = time.time()
    epoch = start_epoch
    for epoch in range(start_epoch, epoch_limit):
        for batch in loader:
            if global_step >= total_steps:
                break

            src = batch["src"].to(device, non_blocking=True)
            tgt = batch["tgt"].to(device, non_blocking=True)
            bsz = src.shape[0]
            label_src = torch.full((bsz,), args.label_src, device=device, dtype=torch.long)
            label_tgt = torch.full((bsz,), args.label_tgt, device=device, dtype=torch.long)

            # Generator update.
            set_requires_grad([disc_x, disc_y], False)
            gen_opt.zero_grad(set_to_none=True)

            fake_tgt = model(src, label_tgt, mode=args.generation_mode, differentiable_tokenize=False)
            rec_src = model(fake_tgt, label_src, mode=args.generation_mode, differentiable_tokenize=True)
            fake_src = model(tgt, label_src, mode=args.generation_mode, differentiable_tokenize=False)
            rec_tgt = model(fake_src, label_tgt, mode=args.generation_mode, differentiable_tokenize=True)

            idt_src = model(src, label_src, mode=args.generation_mode, differentiable_tokenize=False)
            idt_tgt = model(tgt, label_tgt, mode=args.generation_mode, differentiable_tokenize=False)

            loss_cycle = reconstruction_loss(rec_src, src, lpips_model, args.lambda_cycle_lpips) + reconstruction_loss(
                rec_tgt, tgt, lpips_model, args.lambda_cycle_lpips
            )
            loss_idt = reconstruction_loss(idt_src, src, lpips_model, args.lambda_idt_lpips) + reconstruction_loss(
                idt_tgt, tgt, lpips_model, args.lambda_idt_lpips
            )
            loss_gan_g = disc_y(fake_tgt, for_G=True).mean() + disc_x(fake_src, for_G=True).mean()
            loss_g = (
                args.lambda_cycle * loss_cycle
                + args.lambda_idt * loss_idt
                + args.lambda_gan * loss_gan_g
            )
            loss_g.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(gen_params, max_norm=args.max_grad_norm)
            gen_opt.step()
            gen_sche.step()

            # Discriminator update.
            set_requires_grad([disc_x, disc_y], True)
            disc_opt.zero_grad(set_to_none=True)
            loss_d_x = 0.5 * (
                disc_x(src, for_real=True).mean() + disc_x(fake_src.detach(), for_real=False).mean()
            )
            loss_d_y = 0.5 * (
                disc_y(tgt, for_real=True).mean() + disc_y(fake_tgt.detach(), for_real=False).mean()
            )
            loss_d = args.lambda_gan * (loss_d_x + loss_d_y)
            loss_d.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    list(disc_x.parameters()) + list(disc_y.parameters()),
                    max_norm=args.max_grad_norm,
                )
            disc_opt.step()
            disc_sche.step()

            global_step += 1

            if global_step % args.log_every == 0 or global_step == 1:
                elapsed = time.time() - tic
                print(
                    f"[step {global_step:07d}/{total_steps}] "
                    f"lr_g={gen_opt.param_groups[0]['lr']:.2e} "
                    f"lr_d={disc_opt.param_groups[0]['lr']:.2e} "
                    f"loss_G={loss_g.item():.4f} "
                    f"(cycle={loss_cycle.item():.4f}, idt={loss_idt.item():.4f}, gan={loss_gan_g.item():.4f}) "
                    f"loss_D={loss_d.item():.4f} "
                    f"({elapsed / max(global_step - start_step, 1):.3f}s/step)"
                )

            if global_step % args.save_every == 0:
                ckpt_path = out_dir / f"cyclevar_step_{global_step}.pt"
                save_training_state(
                    ckpt_path,
                    step=global_step,
                    epoch=epoch,
                    model=model,
                    disc_x=disc_x,
                    disc_y=disc_y,
                    gen_opt=gen_opt,
                    disc_opt=disc_opt,
                    gen_sche=gen_sche,
                    disc_sche=disc_sche,
                    args=args,
                )
                print(f"[ckpt] saved {ckpt_path}")

        if global_step >= total_steps:
            break

    last_ckpt = out_dir / "cyclevar_last.pt"
    save_training_state(
        last_ckpt,
        step=global_step,
        epoch=epoch,
        model=model,
        disc_x=disc_x,
        disc_y=disc_y,
        gen_opt=gen_opt,
        disc_opt=disc_opt,
        gen_sche=gen_sche,
        disc_sche=disc_sche,
        args=args,
    )
    print(f"[done] saved {last_ckpt}")


if __name__ == "__main__":
    main()
