import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import vision_aided_loss

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from my_utils.training_utils import UnpairedDataset


class _DummyTokenizer:
    model_max_length = 1

    def __call__(self, *args, **kwargs):
        class _Out:
            input_ids = torch.zeros((1, 1), dtype=torch.long)

        return _Out()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def disable_spectral_norm_in_decoder(discriminator):
    removed = 0
    for _, module in discriminator.decoder.named_modules():
        try:
            torch.nn.utils.remove_spectral_norm(module)
            removed += 1
        except Exception:
            continue
    print(f"[repro] removed spectral_norm modules: {removed}")


def _record_state(module_path: str, param_or_buffer: str, tensor: torch.Tensor, when: str) -> Dict:
    with torch.no_grad():
        t = tensor.detach()
        t_float = t.float()
        if t.is_floating_point() or t.is_complex():
            finite_mask = torch.isfinite(t)
            has_nan = bool(torch.isnan(t).any().item())
            has_inf = bool(torch.isinf(t).any().item())
        else:
            finite_mask = torch.ones_like(t, dtype=torch.bool)
            has_nan = False
            has_inf = False

        is_finite = bool(finite_mask.all().item())
        if bool(finite_mask.any().item()):
            valid = t_float[finite_mask]
            min_v = float(valid.min().item())
            max_v = float(valid.max().item())
            mean_v = float(valid.mean().item())
            std_v = float(valid.std(unbiased=False).item())
        else:
            min_v = float("nan")
            max_v = float("nan")
            mean_v = float("nan")
            std_v = float("nan")

    return {
        "when": when,
        "module_path": module_path,
        "param_or_buffer": param_or_buffer,
        "dtype": str(t.dtype),
        "device": str(t.device),
        "shape": list(t.shape),
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
        "isnan": has_nan,
        "isinf": has_inf,
        "is_finite": is_finite,
    }


def _iter_named_params_and_buffers(module) -> List[Tuple[str, torch.Tensor]]:
    entries: List[Tuple[str, torch.Tensor]] = []
    for name, p in module.named_parameters(recurse=True):
        entries.append((f"param:{name}", p))
    for name, b in module.named_buffers(recurse=True):
        entries.append((f"buffer:{name}", b))
    return entries


def scan_disc_state(module_path: str, module, when: str, max_keys: int, jsonl_fp=None) -> Optional[Dict]:
    entries = _iter_named_params_and_buffers(module)
    non_finite: List[Dict] = []
    for key, t in entries:
        rec = _record_state(module_path, key, t, when)
        if not rec["is_finite"]:
            non_finite.append(rec)
            if jsonl_fp is not None:
                jsonl_fp.write(json.dumps(rec) + "\n")
    summary = {
        "when": when,
        "module_path": module_path,
        "param_or_buffer": "<summary>",
        "keys_total": len(entries),
        "keys_non_finite": len(non_finite),
        "is_finite": len(non_finite) == 0,
    }
    if jsonl_fp is not None:
        jsonl_fp.write(json.dumps(summary) + "\n")
        jsonl_fp.flush()

    if len(non_finite) == 0:
        print(f"[repro] {when} {module_path}: all_finite=True keys_total={len(entries)}")
        return None

    print(
        f"[repro] {when} {module_path}: all_finite=False "
        f"keys_non_finite={len(non_finite)} keys_total={len(entries)}"
    )
    for rec in non_finite[: max(1, int(max_keys))]:
        print(
            f"[repro] module_path={rec['module_path']} param_or_buffer={rec['param_or_buffer']} "
            f"dtype={rec['dtype']} device={rec['device']} shape={rec['shape']} "
            f"min={rec['min']:.6g} max={rec['max']:.6g} mean={rec['mean']:.6g} std={rec['std']:.6g} "
            f"isnan={rec['isnan']} isinf={rec['isinf']}"
        )
    return non_finite[0]


def first_non_finite_in_nested(name: str, value) -> Optional[str]:
    if isinstance(value, torch.Tensor):
        if torch.isfinite(value.detach()).all():
            return None
        return name
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            bad = first_non_finite_in_nested(f"{name}[{i}]", item)
            if bad is not None:
                return bad
    if isinstance(value, dict):
        for k, item in value.items():
            bad = first_non_finite_in_nested(f"{name}.{k}", item)
            if bad is not None:
                return bad
    return None


def run_pipeline(name: str, disc, images: torch.Tensor, jsonl_fp=None) -> Optional[str]:
    with torch.no_grad():
        cv_feat = disc.cv_ensemble(images)
    bad = first_non_finite_in_nested(f"{name}.cv_feat", cv_feat)
    if bad is not None:
        if jsonl_fp is not None:
            jsonl_fp.write(json.dumps({"when": name, "first_bad_path": bad}) + "\n")
            jsonl_fp.flush()
        return bad

    pred_mask = []
    with torch.no_grad():
        for i, feat in enumerate(cv_feat):
            pred_mask.append(disc.decoder[i](feat, None))
    bad = first_non_finite_in_nested(f"{name}.pred_mask", pred_mask)
    if bad is not None:
        if jsonl_fp is not None:
            jsonl_fp.write(json.dumps({"when": name, "first_bad_path": bad}) + "\n")
            jsonl_fp.flush()
        return bad

    with torch.no_grad():
        loss_for_g = disc.loss_type(pred_mask, for_G=True) if disc.loss_type is not None else pred_mask
    bad = first_non_finite_in_nested(f"{name}.loss_for_G", loss_for_g)
    if bad is not None:
        if jsonl_fp is not None:
            jsonl_fp.write(json.dumps({"when": name, "first_bad_path": bad}) + "\n")
            jsonl_fp.flush()
        return bad

    return None


def maybe_load_real_batch(args, device: torch.device) -> Optional[torch.Tensor]:
    if not args.dataset_folder:
        print("[repro] dataset_folder not provided: skipping real batch check")
        return None
    ds = UnpairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_img_prep,
        split="train",
        tokenizer=_DummyTokenizer(),
        return_paths=True,
    )
    if len(ds) == 0:
        print("[repro] dataset is empty: skipping real batch check")
        return None
    sample = ds[0]
    img = sample["pixel_values_tgt"].unsqueeze(0).float().to(device)
    print(f"[repro] loaded real batch path={sample.get('path_tgt')}")
    return img


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce and isolate vision-aided discriminator NaN.")
    parser.add_argument("--gan_output_type", type=str, default="conv_multi_level", choices=["conv_multi_level", "pool"])
    parser.add_argument("--gan_loss_type", type=str, default="multilevel_sigmoid")
    parser.add_argument("--gan_diffaug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gan_disable_spectral_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dataset_folder", type=str, default=None)
    parser.add_argument("--train_img_prep", type=str, default="resize_286_randomcrop_256x256_hflip")
    parser.add_argument("--scan_max_keys", type=int, default=20)
    parser.add_argument("--output_jsonl", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[repro] CUDA is not available, fallback to cpu")
        device = torch.device("cpu")

    jsonl_fp = None
    if args.output_jsonl:
        out_dir = os.path.dirname(args.output_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        jsonl_fp = open(args.output_jsonl, "w", encoding="utf-8")

    try:
        disc = vision_aided_loss.Discriminator(
            cv_type="clip",
            output_type=args.gan_output_type,
            loss_type=args.gan_loss_type,
            diffaug=args.gan_diffaug,
            device=str(device),
        )
        disc.cv_ensemble.requires_grad_(False)
        disc.to(device)
        disc.eval()

        if args.gan_disable_spectral_norm:
            disable_spectral_norm_in_decoder(disc)

        first_bad = scan_disc_state("disc", disc, "post_discriminator_init", args.scan_max_keys, jsonl_fp)
        if first_bad is not None:
            print(f"[repro][FAIL] non-finite state at init: {first_bad['param_or_buffer']}")
            sys.exit(1)

        x_rand = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device).mul_(2.0).sub_(1.0)
        bad = run_pipeline("random_input", disc, x_rand, jsonl_fp)
        if bad is not None:
            print(f"[repro][FAIL] non-finite output path on random input: {bad}")
            sys.exit(1)

        first_bad = scan_disc_state("disc", disc, "post_random_forward", args.scan_max_keys, jsonl_fp)
        if first_bad is not None:
            print(f"[repro][FAIL] non-finite state after random forward: {first_bad['param_or_buffer']}")
            sys.exit(1)

        real_batch = maybe_load_real_batch(args, device)
        if real_batch is not None:
            bad = run_pipeline("real_input", disc, real_batch, jsonl_fp)
            if bad is not None:
                print(f"[repro][FAIL] non-finite output path on real input: {bad}")
                sys.exit(1)

            first_bad = scan_disc_state("disc", disc, "post_real_forward", args.scan_max_keys, jsonl_fp)
            if first_bad is not None:
                print(f"[repro][FAIL] non-finite state after real forward: {first_bad['param_or_buffer']}")
                sys.exit(1)

        print("[repro][PASS] discriminator state/output are finite for all executed checks")
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()


if __name__ == "__main__":
    main()
