import gc
import os
from glob import glob

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


def _ensure_finite(name: str, tensor: torch.Tensor):
    if torch.isfinite(tensor).all():
        return
    bad = (~torch.isfinite(tensor)).sum().item()
    total = tensor.numel()
    safe = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    raise FloatingPointError(
        f"[non-finite] {name}: {bad}/{total} values are NaN/Inf "
        f"(min={safe.min().item():.5f}, max={safe.max().item():.5f})"
    )


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

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
    )
    generator.train()

    if args.gan_disc_type != "vagan_clip":
        raise ValueError("CycleVAR reimplementation currently supports --gan_disc_type vagan_clip only.")

    net_disc_a = vision_aided_loss.Discriminator(cv_type="clip", loss_type=args.gan_loss_type, device="cuda")
    net_disc_b = vision_aided_loss.Discriminator(cv_type="clip", loss_type=args.gan_loss_type, device="cuda")
    net_disc_a.cv_ensemble.requires_grad_(False)
    net_disc_b.cv_ensemble.requires_grad_(False)

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

    dataset_train = UnpairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_img_prep,
        split="train",
        tokenizer=_DummyTokenizer(),
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

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    global_step = 0
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
                _ensure_finite("img_a", img_a)
                _ensure_finite("img_b", img_b)

                # Cycle consistency
                cyc_fake_b = generator(img_a, "a2b")
                cyc_rec_a = generator(cyc_fake_b, "b2a")
                _ensure_finite("cyc_fake_b", cyc_fake_b)
                _ensure_finite("cyc_rec_a", cyc_rec_a)
                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                loss_cycle_a = loss_cycle_a + net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips

                cyc_fake_a = generator(img_b, "b2a")
                cyc_rec_b = generator(cyc_fake_a, "a2b")
                _ensure_finite("cyc_fake_a", cyc_fake_a)
                _ensure_finite("cyc_rec_b", cyc_rec_b)
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                loss_cycle_b = loss_cycle_b + net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips
                _ensure_finite("loss_cycle_a", loss_cycle_a)
                _ensure_finite("loss_cycle_b", loss_cycle_b)

                accelerator.backward(loss_cycle_a + loss_cycle_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                # GAN loss (generator)
                fake_a = generator(img_b, "b2a")
                fake_b = generator(img_a, "a2b")
                _ensure_finite("fake_a", fake_a)
                _ensure_finite("fake_b", fake_b)
                loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                _ensure_finite("loss_gan_a", loss_gan_a)
                _ensure_finite("loss_gan_b", loss_gan_b)
                accelerator.backward(loss_gan_a + loss_gan_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                # Identity loss
                idt_a = generator(img_b, "a2b")
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                loss_idt_a = loss_idt_a + net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips
                _ensure_finite("idt_a", idt_a)
                _ensure_finite("loss_idt_a", loss_idt_a)

                idt_b = generator(img_a, "b2a")
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                loss_idt_b = loss_idt_b + net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips
                _ensure_finite("idt_b", idt_b)
                _ensure_finite("loss_idt_b", loss_idt_b)

                loss_g_idt = loss_idt_a + loss_idt_b
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                # Discriminator (fake)
                loss_d_a_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                loss_d_b_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                loss_d_fake = (loss_d_a_fake + loss_d_b_fake) * 0.5
                _ensure_finite("loss_d_fake", loss_d_fake)
                accelerator.backward(loss_d_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                # Discriminator (real)
                loss_d_a_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                loss_d_b_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                loss_d_real = (loss_d_a_real + loss_d_b_real) * 0.5
                _ensure_finite("loss_d_real", loss_d_real)
                accelerator.backward(loss_d_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

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
                        ckpt["vqvae_ckpt_path"] = args.vqvae_ckpt_path
                        ckpt["var_ckpt_path"] = args.var_ckpt_path
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


if __name__ == "__main__":
    args = parse_args_cyclevar_training()
    main(args)
