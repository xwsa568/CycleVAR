import gc
import json
import os
import atexit
import importlib
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import lpips
import numpy as np
import torch
import vision_aided_loss
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from cleanfid.fid import build_feature_extractor, frechet_distance, get_folder_features
from diffusers.optimization import get_scheduler
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from cyclevar import CycleVAR
from my_utils.dino_struct import DinoStructureLoss
from my_utils.training_utils import UnpairedDataset, build_transform, parse_args_cyclevar_training


class _DummyTokenizer:
    model_max_length = 1

    def __call__(self, *args, **kwargs):
        class _Out:
            input_ids = torch.zeros((1, 1), dtype=torch.long)

        return _Out()


def _to_scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def check_finite(name, tensor, step, accelerator, abort_on_non_finite=True):
    if tensor is None:
        return True
    with torch.no_grad():
        t = tensor.detach()
        finite_mask = torch.isfinite(t)
        is_ok = bool(finite_mask.all().item())
        if is_ok:
            return True

        t_float = t.float()
        has_nan = bool(torch.isnan(t_float).any().item())
        has_inf = bool(torch.isinf(t_float).any().item())
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

    msg = (
        f"[nan-guard] step={step} name={name} non-finite detected "
        f"(nan={has_nan}, inf={has_inf}, min={min_v:.6g}, max={max_v:.6g}, "
        f"mean={mean_v:.6g}, std={std_v:.6g})"
    )
    accelerator.print(msg)
    if abort_on_non_finite:
        raise RuntimeError(msg)
    return False


def check_nested_finite(name, value, step, accelerator, abort_on_non_finite=True):
    if isinstance(value, torch.Tensor):
        return check_finite(name, value, step, accelerator, abort_on_non_finite)
    if isinstance(value, (list, tuple)):
        ok = True
        for idx, item in enumerate(value):
            if not check_nested_finite(f"{name}[{idx}]", item, step, accelerator, abort_on_non_finite):
                ok = False
        return ok
    accelerator.print(f"[nan-guard] step={step} name={name} unsupported type={type(value)}")
    return True


def _state_tensor_record(module_path: str, param_or_buffer: str, tensor: torch.Tensor, when: str, step: int) -> Dict:
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
        "step": int(step),
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


def scan_module_state(
    module_path: str,
    module,
    when: str,
    step: int,
    accelerator,
    abort_on_non_finite: bool,
    max_keys: int,
    jsonl_fp: Optional[TextIO] = None,
) -> Optional[Dict]:
    entries = _iter_named_params_and_buffers(module)
    non_finite: List[Dict] = []
    for key, tensor in entries:
        rec = _state_tensor_record(module_path, key, tensor, when, step)
        if not rec["is_finite"]:
            non_finite.append(rec)
            if jsonl_fp is not None:
                jsonl_fp.write(json.dumps(rec) + "\n")

    summary = {
        "when": when,
        "step": int(step),
        "module_path": module_path,
        "param_or_buffer": "<summary>",
        "dtype": "",
        "device": "",
        "shape": [],
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "isnan": len(non_finite) > 0,
        "isinf": False,
        "is_finite": len(non_finite) == 0,
        "keys_total": len(entries),
        "keys_non_finite": len(non_finite),
    }
    if jsonl_fp is not None:
        jsonl_fp.write(json.dumps(summary) + "\n")
        jsonl_fp.flush()

    if len(non_finite) == 0:
        accelerator.print(
            f"[disc-scan] when={when} step={step} module={module_path} all_finite=True keys_total={len(entries)}"
        )
        return None

    accelerator.print(
        f"[disc-scan] when={when} step={step} module={module_path} "
        f"all_finite=False keys_non_finite={len(non_finite)} keys_total={len(entries)}"
    )
    for rec in non_finite[:max(1, int(max_keys))]:
        accelerator.print(
            f"[disc-scan] module_path={rec['module_path']} param_or_buffer={rec['param_or_buffer']} "
            f"dtype={rec['dtype']} device={rec['device']} shape={rec['shape']} "
            f"min={rec['min']:.6g} max={rec['max']:.6g} mean={rec['mean']:.6g} std={rec['std']:.6g} "
            f"isnan={rec['isnan']} isinf={rec['isinf']}"
        )
    if len(non_finite) > max(1, int(max_keys)):
        accelerator.print(
            f"[disc-scan] ... truncated {len(non_finite) - max(1, int(max_keys))} additional non-finite keys"
        )

    first_bad = non_finite[0]
    if abort_on_non_finite:
        raise RuntimeError(
            f"[disc-scan] first_non_finite when={when} step={step} "
            f"module_path={first_bad['module_path']} param_or_buffer={first_bad['param_or_buffer']}"
        )
    return first_bad


def disable_spectral_norm_in_decoder(discriminator, disc_name: str, accelerator):
    removed = 0
    for module_name, module in discriminator.decoder.named_modules():
        try:
            torch.nn.utils.remove_spectral_norm(module)
            removed += 1
            accelerator.print(
                f"[disc-config] removed spectral_norm from {disc_name}.decoder.{module_name or '<root>'}"
            )
        except Exception:
            continue
    accelerator.print(f"[disc-config] {disc_name}: total_spectral_norm_removed={removed}")


def log_tensor_stats(name, tensor, step, accelerator):
    with torch.no_grad():
        t = tensor.detach().float()
        accelerator.print(
            f"[nan-guard] step={step} {name}: "
            f"min={float(t.min().item()):.6g}, max={float(t.max().item()):.6g}, "
            f"mean={float(t.mean().item()):.6g}, std={float(t.std(unbiased=False).item()):.6g}, "
            f"absmax={float(t.abs().max().item()):.6g}"
        )


def check_input_quality(name, tensor, step, accelerator, paths=None):
    with torch.no_grad():
        t = tensor.detach().float()
        min_v = float(t.min().item())
        max_v = float(t.max().item())
        out_of_range = bool((t < -1.0).any().item() or (t > 1.0).any().item())
        all_zero = bool((t == 0).all().item())
        t_flat = t.view(t.shape[0], -1)
        per_sample_const = (t_flat.max(dim=1).values - t_flat.min(dim=1).values) == 0
        has_constant = bool(per_sample_const.any().item())
        has_non_finite = bool((~torch.isfinite(t)).any().item())

    if out_of_range or all_zero or has_constant or has_non_finite:
        path_str = ""
        if paths is not None:
            path_str = f", paths={paths}"
        accelerator.print(
            f"[nan-guard] step={step} input={name} suspicious "
            f"(min={min_v:.6g}, max={max_v:.6g}, out_of_range={out_of_range}, "
            f"all_zero={all_zero}, constant_sample={has_constant}, non_finite={has_non_finite}{path_str})"
        )


def debug_discriminator_pipeline(name, disc, images, step, accelerator, abort_on_non_finite=True):
    with torch.no_grad():
        cv_feat = disc.cv_ensemble(images)
    check_nested_finite(f"{name}.cv_feat", cv_feat, step, accelerator, abort_on_non_finite)
    if isinstance(cv_feat, (list, tuple)):
        for i, feat in enumerate(cv_feat):
            log_tensor_stats(f"{name}.cv_feat[{i}]", feat, step, accelerator)
    elif isinstance(cv_feat, torch.Tensor):
        log_tensor_stats(f"{name}.cv_feat", cv_feat, step, accelerator)

    pred_mask = []
    with torch.no_grad():
        for i, feat in enumerate(cv_feat):
            pred_i = disc.decoder[i](feat, None)
            pred_mask.append(pred_i)
    check_nested_finite(f"{name}.pred_mask", pred_mask, step, accelerator, abort_on_non_finite)

    if hasattr(disc, "loss_type") and disc.loss_type is not None:
        with torch.no_grad():
            loss_for_g = disc.loss_type(pred_mask, for_G=True)
            loss_for_fake = disc.loss_type(pred_mask, for_real=False)
        check_nested_finite(f"{name}.loss_for_G", loss_for_g, step, accelerator, abort_on_non_finite)
        check_nested_finite(f"{name}.loss_for_fake", loss_for_fake, step, accelerator, abort_on_non_finite)


def _read_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _resolve_cycle_prompts(args) -> Tuple[str, str]:
    prompt_a = args.prompt_a
    prompt_b = args.prompt_b
    if prompt_a is None:
        prompt_a = _read_prompt_file(os.path.join(args.dataset_folder, "fixed_prompt_a.txt"))
    if prompt_b is None:
        prompt_b = _read_prompt_file(os.path.join(args.dataset_folder, "fixed_prompt_b.txt"))
    prompt_a = str(prompt_a).strip()
    prompt_b = str(prompt_b).strip()
    if not prompt_a or not prompt_b:
        raise ValueError("Both prompt_a and prompt_b must be non-empty.")
    return prompt_a, prompt_b


def _default_infinity_root() -> Path:
    env_root = os.environ.get("INFINITY_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # /workspace/CycleVAR/src/train_cyclevar.py -> /workspace/Infinity
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
        raise FileNotFoundError(
            f"Infinity repository not found at {infinity_root}. "
            "Set INFINITY_ROOT if your Infinity checkout is elsewhere."
        )
    root_str = str(infinity_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        importlib.import_module("flash_attn")
    except Exception as exc:
        raise RuntimeError(
            "flash_attn import failed. Install FlashAttention before using --generator_backbone infinity."
        ) from exc

    try:
        importlib.import_module("infinity")
    except Exception as exc:
        raise RuntimeError(
            f"infinity import failed from root={infinity_root}. Check Infinity dependencies and INFINITY_ROOT."
        ) from exc


def main(args):
    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "log_with": args.report_to,
    }
    if args.mixed_precision is not None:
        accelerator_kwargs["mixed_precision"] = args.mixed_precision
    accelerator = Accelerator(**accelerator_kwargs)
    set_seed(args.seed)
    if args.debug_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        accelerator.print("[nan-guard] torch.autograd.set_detect_anomaly(True)")

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    disc_scan_enabled = bool(args.debug_disc_state_scan and accelerator.is_main_process)
    disc_scan_fp: Optional[TextIO] = None
    if disc_scan_enabled and args.debug_disc_scan_save_jsonl:
        disc_scan_path = os.path.join(args.output_dir, "debug_disc_state.jsonl")
        disc_scan_fp = open(disc_scan_path, "w", encoding="utf-8")
        atexit.register(disc_scan_fp.close)
        accelerator.print(f"[disc-scan] writing JSONL to {disc_scan_path}")

    prompt_a = None
    prompt_b = None
    if args.generator_backbone == "infinity":
        _preflight_infinity(args)
        prompt_a, prompt_b = _resolve_cycle_prompts(args)
        from cycleinfinity import CycleInfinity

        generator = CycleInfinity(
            infinity_model_path=args.infinity_model_path,
            infinity_vae_path=args.infinity_vae_path,
            infinity_text_encoder_ckpt=args.infinity_text_encoder_ckpt,
            cycleinfinity_ckpt_path=args.resume_cyclevar_ckpt,
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
            debug_nan_guard=args.debug_nan_guard,
            debug_nan_abort=args.debug_nan_abort,
        )
        accelerator.print(
            f"[CycleInfinity] prompts loaded: prompt_a='{prompt_a}', prompt_b='{prompt_b}', pn={args.infinity_pn}"
        )
    elif args.generator_backbone == "var":
        generator = CycleVAR(
            vqvae_ckpt_path=args.vqvae_ckpt_path,
            var_ckpt_path=args.var_ckpt_path,
            cyclevar_ckpt_path=args.resume_cyclevar_ckpt,
            patch_nums=args.var_patch_nums,
            var_depth=args.var_depth,
            num_classes=args.var_num_classes,
            label_a=args.label_a,
            label_b=args.label_b,
            srq_temperature=args.srq_temperature,
            source_temperature=args.source_temperature,
            use_srq_gumbel=args.use_srq_gumbel,
            use_source_ste=args.use_source_ste,
            src_fusion_alpha=args.src_fusion_alpha,
            debug_nan_guard=args.debug_nan_guard,
            debug_nan_abort=args.debug_nan_abort,
        )
    else:
        raise ValueError(f"Unsupported --generator_backbone: {args.generator_backbone}")
    generator.train()

    if args.gan_disc_type != "vagan_clip":
        raise ValueError("CycleVAR reimplementation currently supports --gan_disc_type vagan_clip only.")

    net_disc_a = vision_aided_loss.Discriminator(
        cv_type="clip",
        output_type=args.gan_output_type,
        loss_type=args.gan_loss_type,
        diffaug=args.gan_diffaug,
        device="cuda",
    )
    net_disc_b = vision_aided_loss.Discriminator(
        cv_type="clip",
        output_type=args.gan_output_type,
        loss_type=args.gan_loss_type,
        diffaug=args.gan_diffaug,
        device="cuda",
    )
    net_disc_a.cv_ensemble.requires_grad_(False)
    net_disc_b.cv_ensemble.requires_grad_(False)

    if disc_scan_enabled:
        scan_module_state(
            module_path="net_disc_a",
            module=net_disc_a,
            when="post_discriminator_init",
            step=0,
            accelerator=accelerator,
            abort_on_non_finite=args.debug_nan_abort,
            max_keys=args.debug_disc_scan_max_keys,
            jsonl_fp=disc_scan_fp,
        )
        scan_module_state(
            module_path="net_disc_b",
            module=net_disc_b,
            when="post_discriminator_init",
            step=0,
            accelerator=accelerator,
            abort_on_non_finite=args.debug_nan_abort,
            max_keys=args.debug_disc_scan_max_keys,
            jsonl_fp=disc_scan_fp,
        )

    if args.gan_disable_spectral_norm:
        disable_spectral_norm_in_decoder(net_disc_a, "net_disc_a", accelerator)
        disable_spectral_norm_in_decoder(net_disc_b, "net_disc_b", accelerator)

    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    params_gen = generator.get_trainable_params()
    optimizer_gen = torch.optim.AdamW(
        params_gen,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(
        params_disc,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if disc_scan_enabled:
        scan_module_state(
            module_path="net_disc_a",
            module=net_disc_a,
            when="post_optimizer_init",
            step=0,
            accelerator=accelerator,
            abort_on_non_finite=args.debug_nan_abort,
            max_keys=args.debug_disc_scan_max_keys,
            jsonl_fp=disc_scan_fp,
        )
        scan_module_state(
            module_path="net_disc_b",
            module=net_disc_b,
            when="post_optimizer_init",
            step=0,
            accelerator=accelerator,
            abort_on_non_finite=args.debug_nan_abort,
            max_keys=args.debug_disc_scan_max_keys,
            jsonl_fp=disc_scan_fp,
        )

    dataset_train = UnpairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_img_prep,
        split="train",
        tokenizer=_DummyTokenizer(),
        return_paths=args.debug_nan_guard,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    T_val = build_transform(args.val_img_prep)
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(args.dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)

    if accelerator.is_main_process:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        output_dir_ref = os.path.join(args.output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)
        for _path in tqdm(l_images_tgt_test):
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert("RGB"))
                _img.save(outf)
        ref_features = get_folder_features(
            output_dir_ref,
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device("cuda"),
            mode="clean",
            custom_fn_resize=None,
            description="",
            verbose=True,
            custom_image_tranform=None,
        )
        a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)

        output_dir_ref = os.path.join(args.output_dir, "fid_reference_b2a")
        os.makedirs(output_dir_ref, exist_ok=True)
        for _path in tqdm(l_images_src_test):
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert("RGB"))
                _img.save(outf)
        ref_features = get_folder_features(
            output_dir_ref,
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device("cuda"),
            mode="clean",
            custom_fn_resize=None,
            description="",
            verbose=True,
            custom_image_tranform=None,
        )
        b2a_ref_mu, b2a_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)

    if args.max_train_steps is None:
        args.max_train_steps = args.max_train_epochs * len(train_dataloader)

    lr_scheduler_gen = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    net_lpips = lpips.LPIPS(net="vgg")
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    generator, net_disc_a, net_disc_b = accelerator.prepare(generator, net_disc_a, net_disc_b)
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips,
        optimizer_gen,
        optimizer_disc,
        train_dataloader,
        lr_scheduler_gen,
        lr_scheduler_disc,
    )

    if disc_scan_enabled:
        scan_module_state(
            module_path="net_disc_a",
            module=accelerator.unwrap_model(net_disc_a),
            when="post_accelerator_prepare",
            step=0,
            accelerator=accelerator,
            abort_on_non_finite=args.debug_nan_abort,
            max_keys=args.debug_disc_scan_max_keys,
            jsonl_fp=disc_scan_fp,
        )
        scan_module_state(
            module_path="net_disc_b",
            module=accelerator.unwrap_model(net_disc_b),
            when="post_accelerator_prepare",
            step=0,
            accelerator=accelerator,
            abort_on_non_finite=args.debug_nan_abort,
            max_keys=args.debug_disc_scan_max_keys,
            jsonl_fp=disc_scan_fp,
        )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))
    if args.debug_nan_guard:
        accelerator.print(
            "[nan-guard] enabled "
            f"(abort={args.debug_nan_abort}, every={max(1, args.debug_nan_every)}, "
            f"generator_backbone={args.generator_backbone}, "
            f"srq_temperature={args.srq_temperature}, source_temperature={args.source_temperature}, "
            f"gan_diffaug={args.gan_diffaug}, gan_output_type={args.gan_output_type}, "
            f"gan_disable_spectral_norm={args.gan_disable_spectral_norm})"
        )

    global_step = 0
    pre_step0_disc_scan_done = False
    debug_every = max(1, int(args.debug_nan_every))
    use_gan = float(args.lambda_gan) > 0.0
    if accelerator.is_main_process and not use_gan:
        accelerator.print("[nan-guard] lambda_gan <= 0: skipping GAN/discriminator optimization paths.")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for _epoch in range(args.max_train_epochs):
        for _step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(generator, net_disc_a, net_disc_b):
                img_a = batch["pixel_values_src"].float()
                img_b = batch["pixel_values_tgt"].float()
                bsz = img_a.shape[0]
                should_debug = args.debug_nan_guard and (global_step % debug_every == 0)

                if disc_scan_enabled and global_step == 0 and not pre_step0_disc_scan_done:
                    scan_module_state(
                        module_path="net_disc_a",
                        module=accelerator.unwrap_model(net_disc_a),
                        when="pre_step0_forward",
                        step=global_step,
                        accelerator=accelerator,
                        abort_on_non_finite=args.debug_nan_abort,
                        max_keys=args.debug_disc_scan_max_keys,
                        jsonl_fp=disc_scan_fp,
                    )
                    scan_module_state(
                        module_path="net_disc_b",
                        module=accelerator.unwrap_model(net_disc_b),
                        when="pre_step0_forward",
                        step=global_step,
                        accelerator=accelerator,
                        abort_on_non_finite=args.debug_nan_abort,
                        max_keys=args.debug_disc_scan_max_keys,
                        jsonl_fp=disc_scan_fp,
                    )
                    pre_step0_disc_scan_done = True

                if should_debug:
                    check_finite("img_a", img_a, global_step, accelerator, args.debug_nan_abort)
                    check_finite("img_b", img_b, global_step, accelerator, args.debug_nan_abort)
                    if global_step < 2:
                        check_input_quality("img_a", img_a, global_step, accelerator, batch.get("path_src"))
                        check_input_quality("img_b", img_b, global_step, accelerator, batch.get("path_tgt"))
                        if use_gan:
                            debug_discriminator_pipeline(
                                "disc_a_on_real_b",
                                net_disc_a,
                                img_b,
                                global_step,
                                accelerator,
                                args.debug_nan_abort,
                            )
                            debug_discriminator_pipeline(
                                "disc_b_on_real_a",
                                net_disc_b,
                                img_a,
                                global_step,
                                accelerator,
                                args.debug_nan_abort,
                            )

                # Cycle consistency
                cyc_fake_b = generator(img_a, "a2b")
                cyc_rec_a = generator(cyc_fake_b, "b2a")
                if should_debug:
                    check_finite("cyc_fake_b", cyc_fake_b, global_step, accelerator, args.debug_nan_abort)
                    check_finite("cyc_rec_a", cyc_rec_a, global_step, accelerator, args.debug_nan_abort)
                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                loss_cycle_a = loss_cycle_a + net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips

                cyc_fake_a = generator(img_b, "b2a")
                cyc_rec_b = generator(cyc_fake_a, "a2b")
                if should_debug:
                    check_finite("cyc_fake_a", cyc_fake_a, global_step, accelerator, args.debug_nan_abort)
                    check_finite("cyc_rec_b", cyc_rec_b, global_step, accelerator, args.debug_nan_abort)
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                loss_cycle_b = loss_cycle_b + net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips
                loss_cycle = loss_cycle_a + loss_cycle_b
                if should_debug:
                    check_finite("loss_cycle_a", loss_cycle_a, global_step, accelerator, args.debug_nan_abort)
                    check_finite("loss_cycle_b", loss_cycle_b, global_step, accelerator, args.debug_nan_abort)
                    check_finite("loss_cycle", loss_cycle, global_step, accelerator, args.debug_nan_abort)

                accelerator.backward(loss_cycle, retain_graph=False)
                if accelerator.sync_gradients:
                    gen_cycle_grad_norm = accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                    if should_debug:
                        check_finite(
                            "grad_norm_gen_cycle",
                            torch.as_tensor(gen_cycle_grad_norm),
                            global_step,
                            accelerator,
                            args.debug_nan_abort,
                        )
                        accelerator.print(
                            f"[nan-guard] step={global_step} grad_norm_gen_cycle={_to_scalar(gen_cycle_grad_norm):.6g}"
                        )
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                # GAN loss (generator)
                fake_a = generator(img_b, "b2a")
                fake_b = generator(img_a, "a2b")
                if should_debug:
                    check_finite("fake_a", fake_a, global_step, accelerator, args.debug_nan_abort)
                    check_finite("fake_b", fake_b, global_step, accelerator, args.debug_nan_abort)
                    if use_gan:
                        debug_discriminator_pipeline(
                            "disc_a_on_fake_b",
                            net_disc_a,
                            fake_b,
                            global_step,
                            accelerator,
                            args.debug_nan_abort,
                        )
                        debug_discriminator_pipeline(
                            "disc_b_on_fake_a",
                            net_disc_b,
                            fake_a,
                            global_step,
                            accelerator,
                            args.debug_nan_abort,
                        )
                if use_gan:
                    loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                    loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                    loss_gan = loss_gan_a + loss_gan_b
                    if should_debug:
                        check_finite("loss_gan_a", loss_gan_a, global_step, accelerator, args.debug_nan_abort)
                        check_finite("loss_gan_b", loss_gan_b, global_step, accelerator, args.debug_nan_abort)
                        check_finite("loss_gan", loss_gan, global_step, accelerator, args.debug_nan_abort)
                    accelerator.backward(loss_gan, retain_graph=False)
                    if accelerator.sync_gradients:
                        gen_gan_grad_norm = accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                        if should_debug:
                            check_finite(
                                "grad_norm_gen_gan",
                                torch.as_tensor(gen_gan_grad_norm),
                                global_step,
                                accelerator,
                                args.debug_nan_abort,
                            )
                            accelerator.print(
                                f"[nan-guard] step={global_step} grad_norm_gen_gan={_to_scalar(gen_gan_grad_norm):.6g}"
                            )
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()
                    optimizer_disc.zero_grad()
                else:
                    loss_gan_a = torch.zeros((), device=img_a.device)
                    loss_gan_b = torch.zeros((), device=img_a.device)
                    loss_gan = torch.zeros((), device=img_a.device)

                # Identity loss
                idt_a = generator(img_b, "a2b")
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                loss_idt_a = loss_idt_a + net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips

                idt_b = generator(img_a, "b2a")
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                loss_idt_b = loss_idt_b + net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips
                if should_debug:
                    check_finite("idt_a", idt_a, global_step, accelerator, args.debug_nan_abort)
                    check_finite("idt_b", idt_b, global_step, accelerator, args.debug_nan_abort)
                    check_finite("loss_idt_a", loss_idt_a, global_step, accelerator, args.debug_nan_abort)
                    check_finite("loss_idt_b", loss_idt_b, global_step, accelerator, args.debug_nan_abort)

                loss_g_idt = loss_idt_a + loss_idt_b
                if should_debug:
                    check_finite("loss_g_idt", loss_g_idt, global_step, accelerator, args.debug_nan_abort)
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    gen_idt_grad_norm = accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                    if should_debug:
                        check_finite(
                            "grad_norm_gen_idt",
                            torch.as_tensor(gen_idt_grad_norm),
                            global_step,
                            accelerator,
                            args.debug_nan_abort,
                        )
                        accelerator.print(
                            f"[nan-guard] step={global_step} grad_norm_gen_idt={_to_scalar(gen_idt_grad_norm):.6g}"
                        )
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                # Discriminator (fake)
                if use_gan:
                    loss_d_a_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                    loss_d_b_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                    loss_d_fake = (loss_d_a_fake + loss_d_b_fake) * 0.5
                    if should_debug:
                        check_finite("loss_d_a_fake", loss_d_a_fake, global_step, accelerator, args.debug_nan_abort)
                        check_finite("loss_d_b_fake", loss_d_b_fake, global_step, accelerator, args.debug_nan_abort)
                        check_finite("loss_d_fake", loss_d_fake, global_step, accelerator, args.debug_nan_abort)
                    accelerator.backward(loss_d_fake, retain_graph=False)
                    if accelerator.sync_gradients:
                        params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                        disc_fake_grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        if should_debug:
                            check_finite(
                                "grad_norm_disc_fake",
                                torch.as_tensor(disc_fake_grad_norm),
                                global_step,
                                accelerator,
                                args.debug_nan_abort,
                            )
                            accelerator.print(
                                f"[nan-guard] step={global_step} grad_norm_disc_fake={_to_scalar(disc_fake_grad_norm):.6g}"
                            )
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad()
                else:
                    loss_d_a_fake = torch.zeros((), device=img_a.device)
                    loss_d_b_fake = torch.zeros((), device=img_a.device)
                    loss_d_fake = torch.zeros((), device=img_a.device)

                # Discriminator (real)
                if use_gan:
                    loss_d_a_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                    loss_d_b_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                    loss_d_real = (loss_d_a_real + loss_d_b_real) * 0.5
                    if should_debug:
                        check_finite("loss_d_a_real", loss_d_a_real, global_step, accelerator, args.debug_nan_abort)
                        check_finite("loss_d_b_real", loss_d_b_real, global_step, accelerator, args.debug_nan_abort)
                        check_finite("loss_d_real", loss_d_real, global_step, accelerator, args.debug_nan_abort)
                    accelerator.backward(loss_d_real, retain_graph=False)
                    if accelerator.sync_gradients:
                        params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                        disc_real_grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        if should_debug:
                            check_finite(
                                "grad_norm_disc_real",
                                torch.as_tensor(disc_real_grad_norm),
                                global_step,
                                accelerator,
                                args.debug_nan_abort,
                            )
                            accelerator.print(
                                f"[nan-guard] step={global_step} grad_norm_disc_real={_to_scalar(disc_real_grad_norm):.6g}"
                            )
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad()
                else:
                    loss_d_a_real = torch.zeros((), device=img_a.device)
                    loss_d_b_real = torch.zeros((), device=img_a.device)
                    loss_d_real = torch.zeros((), device=img_a.device)

            logs = {
                "cycle_a": loss_cycle_a.detach().item(),
                "cycle_b": loss_cycle_b.detach().item(),
                "gan_a": loss_gan_a.detach().item(),
                "gan_b": loss_gan_b.detach().item(),
                "disc_a": loss_d_a_fake.detach().item() + loss_d_a_real.detach().item(),
                "disc_b": loss_d_b_fake.detach().item() + loss_d_b_real.detach().item(),
                "idt_a": loss_idt_a.detach().item(),
                "idt_b": loss_idt_b.detach().item(),
            }

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_generator = accelerator.unwrap_model(generator)
                    if global_step % args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                log_dict = {
                                    "train/real_a": [wandb.Image(img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/real_b": [wandb.Image(img_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/rec_a": [wandb.Image(cyc_rec_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/rec_b": [wandb.Image(cyc_rec_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/fake_b": [wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/fake_a": [wandb.Image(fake_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                }
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        ckpt = eval_generator.export_checkpoint()
                        ckpt["generator_backbone"] = args.generator_backbone
                        if args.generator_backbone == "var":
                            ckpt["vqvae_ckpt_path"] = args.vqvae_ckpt_path
                            ckpt["var_ckpt_path"] = args.var_ckpt_path
                        elif args.generator_backbone == "infinity":
                            ckpt["infinity_model_path"] = args.infinity_model_path
                            ckpt["infinity_vae_path"] = args.infinity_vae_path
                            ckpt["infinity_text_encoder_ckpt"] = args.infinity_text_encoder_ckpt
                            ckpt["infinity_model_type"] = args.infinity_model_type
                            ckpt["infinity_pn"] = args.infinity_pn
                            ckpt["infinity_use_bit_label"] = args.infinity_use_bit_label
                            ckpt["infinity_apply_spatial_patchify"] = args.infinity_apply_spatial_patchify
                            ckpt["prompt_a"] = prompt_a
                            ckpt["prompt_b"] = prompt_b
                        torch.save(ckpt, outf)
                        gc.collect()
                        torch.cuda.empty_cache()

                    if global_step % args.validation_steps == 1:
                        was_training = eval_generator.training
                        eval_generator.eval()
                        net_dino = DinoStructureLoss()

                        # A -> B
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_a2b")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_a2b = []
                        for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
                            if idx > args.validation_num_images and args.validation_num_images > 0:
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_a_eval = transforms.ToTensor()(input_img)
                                img_a_eval = transforms.Normalize([0.5], [0.5])(img_a_eval).unsqueeze(0).cuda()
                                eval_fake_b = eval_generator(img_a_eval, "a2b")
                                eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                                eval_fake_b_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
                                l_dino_scores_a2b.append(net_dino.calculate_global_ssim_loss(a, b).item())
                        dino_score_a2b = np.mean(l_dino_scores_a2b)
                        gen_features = get_folder_features(
                            fid_output_dir,
                            model=feat_model,
                            num_workers=0,
                            num=None,
                            shuffle=False,
                            seed=0,
                            batch_size=8,
                            device=torch.device("cuda"),
                            mode="clean",
                            custom_fn_resize=None,
                            description="",
                            verbose=True,
                            custom_image_tranform=None,
                        )
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
                        print(f"step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}")

                        # B -> A
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_b2a")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_b2a = []
                        for idx, input_img_path in enumerate(tqdm(l_images_tgt_test)):
                            if idx > args.validation_num_images and args.validation_num_images > 0:
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_b_eval = transforms.ToTensor()(input_img)
                                img_b_eval = transforms.Normalize([0.5], [0.5])(img_b_eval).unsqueeze(0).cuda()
                                eval_fake_a = eval_generator(img_b_eval, "b2a")
                                eval_fake_a_pil = transforms.ToPILImage()(eval_fake_a[0] * 0.5 + 0.5)
                                eval_fake_a_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0).cuda()
                                l_dino_scores_b2a.append(net_dino.calculate_global_ssim_loss(a, b).item())
                        dino_score_b2a = np.mean(l_dino_scores_b2a)
                        gen_features = get_folder_features(
                            fid_output_dir,
                            model=feat_model,
                            num_workers=0,
                            num=None,
                            shuffle=False,
                            seed=0,
                            batch_size=8,
                            device=torch.device("cuda"),
                            mode="clean",
                            custom_fn_resize=None,
                            description="",
                            verbose=True,
                            custom_image_tranform=None,
                        )
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_b2a = frechet_distance(b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma)
                        print(f"step={global_step}, fid(b2a)={score_fid_b2a:.2f}, dino(b2a)={dino_score_b2a:.3f}")

                        logs["val/fid_a2b"], logs["val/fid_b2a"] = score_fid_a2b, score_fid_b2a
                        logs["val/dino_struct_a2b"], logs["val/dino_struct_b2a"] = dino_score_a2b, dino_score_b2a
                        del net_dino
                        if was_training:
                            eval_generator.train()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    if disc_scan_fp is not None:
        disc_scan_fp.close()


if __name__ == "__main__":
    args = parse_args_cyclevar_training()
    main(args)
