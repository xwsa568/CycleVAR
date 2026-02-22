from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import VAR, VQVAE


Tensor = torch.Tensor


@dataclass
class CycleVAROutput:
    image: Tensor
    logits_per_scale: List[Tensor]
    pred_embeds_per_scale: List[Tensor]
    source_context_per_scale: List[Tensor]
    fused_latent: Tensor


class CycleVAR(nn.Module):
    """
    Re-implementation of CycleVAR core generation mechanics:
    1) Multi-scale source-token prefill.
    2) Softmax Relaxed Quantization (SRQ).
    3) Parallel one-step and serial multi-step generation modes.
    """

    def __init__(
        self,
        vae: VQVAE,
        var: VAR,
        alpha: float = 0.7,
        srq_temperature: float = 2.0,
        tokenize_temperature: float = 1.0,
        use_tokenizer_ste: bool = True,
        freeze_tokenizer: bool = True,
    ):
        super().__init__()
        self.vae = vae
        self.var = var
        self.alpha = alpha
        self.srq_temperature = srq_temperature
        self.tokenize_temperature = tokenize_temperature
        self.use_tokenizer_ste = use_tokenizer_ste

        if freeze_tokenizer:
            for p in self.vae.parameters():
                p.requires_grad_(False)

        self.patch_nums: Tuple[int, ...] = tuple(self.var.patch_nums)
        self.num_scales = len(self.patch_nums)
        self.max_pn = self.patch_nums[-1]
        self.codebook = self.vae.quantize.embedding
        self.quant_resi = self.vae.quantize.quant_resi

    def _prepare_labels(self, labels: Union[int, Tensor], batch: int, device: torch.device) -> Tensor:
        if isinstance(labels, int):
            return torch.full((batch,), labels, device=device, dtype=torch.long)
        if labels.ndim != 1:
            raise ValueError(f"labels must be shape [B], got {tuple(labels.shape)}")
        return labels.to(device=device, dtype=torch.long)

    @staticmethod
    def _pairwise_sq_dist(x: Tensor, codebook: Tensor) -> Tensor:
        # x: [B, N, C], codebook: [V, C] -> [B, N, V]
        x2 = (x * x).sum(dim=-1, keepdim=True)
        c2 = (codebook * codebook).sum(dim=-1).view(1, 1, -1)
        xc = torch.matmul(x, codebook.t())
        return x2 + c2 - 2.0 * xc

    def _sample_codebook_tokens(
        self,
        token_features: Tensor,
        temperature: float,
        differentiable: bool,
        hard: bool,
    ) -> Tuple[Tensor, Tensor]:
        # token_features: [B, N, Cvae]
        logits = -self._pairwise_sq_dist(token_features, self.codebook.weight)

        if hard and not differentiable:
            idx = logits.argmax(dim=-1)
            token_embed = F.embedding(idx, self.codebook.weight)
            return token_embed, logits

        t = max(float(temperature), 1e-6)
        probs = torch.softmax(logits / t, dim=-1)
        soft_embed = torch.matmul(probs, self.codebook.weight)

        if hard and differentiable and self.use_tokenizer_ste:
            idx = probs.argmax(dim=-1)
            hard_embed = F.embedding(idx, self.codebook.weight)
            # Forward uses hard tokens, backward flows through the soft mixture.
            token_embed = hard_embed + (soft_embed - soft_embed.detach())
            return token_embed, logits

        if hard:
            idx = probs.argmax(dim=-1)
            token_embed = F.embedding(idx, self.codebook.weight)
            return token_embed, logits

        return soft_embed, logits

    def tokenize_to_multiscale_context(
        self,
        images: Tensor,
        differentiable: bool = True,
    ) -> List[Tensor]:
        """
        Returns:
            List of K tensors, each [B, p_k^2, Cvae], corresponding to F_k.
        """
        if images.ndim != 4:
            raise ValueError(f"images must be [B,3,H,W], got {tuple(images.shape)}")

        feat = self.vae.quant_conv(self.vae.encoder(images))
        bsz, cvae, _, _ = feat.shape

        f_rest = feat
        f_hat = torch.zeros_like(feat)
        contexts: List[Tensor] = []
        denom = max(self.num_scales - 1, 1)

        for si, pn in enumerate(self.patch_nums):
            if si < self.num_scales - 1:
                z = F.interpolate(f_rest, size=(pn, pn), mode="area")
            else:
                z = f_rest

            z_tokens = z.permute(0, 2, 3, 1).reshape(bsz, pn * pn, cvae)
            token_embed, _ = self._sample_codebook_tokens(
                token_features=z_tokens,
                temperature=self.tokenize_temperature,
                differentiable=differentiable,
                hard=True,
            )

            h = token_embed.view(bsz, pn, pn, cvae).permute(0, 3, 1, 2).contiguous()
            if si < self.num_scales - 1:
                h_up = F.interpolate(h, size=(self.max_pn, self.max_pn), mode="bicubic")
            else:
                h_up = h

            h_up = self.quant_resi[si / denom](h_up)
            f_hat = f_hat + h_up
            f_rest = f_rest - h_up

            fk = F.interpolate(f_hat, size=(pn, pn), mode="area")
            fk = fk.view(bsz, cvae, pn * pn).transpose(1, 2).contiguous()
            contexts.append(fk)

        return contexts

    def _var_forward_with_context(self, label_B: Tensor, context_scales: Sequence[Tensor]) -> Tensor:
        # context_scales: sequence of [B, p_k^2, Cvae]
        x_prompt = torch.cat(context_scales, dim=1)
        bsz, cur_l, _ = x_prompt.shape

        if cur_l > self.var.L:
            raise ValueError(f"context length {cur_l} exceeds VAR length {self.var.L}")

        cond_bd = self.var.class_emb(label_B)
        x = self.var.word_embed(x_prompt.float())

        first_l = min(self.var.first_l, cur_l)
        if first_l > 0:
            x[:, :first_l] = x[:, :first_l] + cond_bd.unsqueeze(1) + self.var.pos_start[:, :first_l]

        x = x + self.var.lvl_embed(self.var.lvl_1L[:, :cur_l].expand(bsz, -1)) + self.var.pos_1LC[:, :cur_l]
        attn_bias = self.var.attn_bias_for_masking[:, :, :cur_l, :cur_l]
        cond_bd_or_gss = self.var.shared_ada_lin(cond_bd)

        # Match VAR forward precision behavior.
        probe = x.new_ones(2, 2)
        main_type = torch.matmul(probe, probe).dtype
        x = x.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        cond_bd_or_gss = cond_bd_or_gss.to(dtype=main_type)

        for block in self.var.blocks:
            x = block(x=x, cond_BD=cond_bd_or_gss, attn_bias=attn_bias)

        return self.var.get_logits(x.float(), cond_bd)

    def _srq_logits_to_embeds(self, logits: Tensor, hard: bool = False) -> Tensor:
        # logits: [B, N, V] -> embeds: [B, N, Cvae]
        if hard:
            idx = logits.argmax(dim=-1)
            return F.embedding(idx, self.codebook.weight)
        t = max(float(self.srq_temperature), 1e-6)
        probs = torch.softmax(logits / t, dim=-1)
        return torch.matmul(probs, self.codebook.weight)

    def _decode_from_predicted_scales(
        self,
        pred_embeds_per_scale: Sequence[Tensor],
        source_context_per_scale: Sequence[Tensor],
    ) -> Tensor:
        # Scale-wise residual prediction -> fused latent -> decoded image.
        pred_maps = []
        for pn, emb in zip(self.patch_nums, pred_embeds_per_scale):
            m = emb.view(emb.shape[0], pn, pn, emb.shape[-1]).permute(0, 3, 1, 2).contiguous()
            pred_maps.append(m)

        pred_latent = self.vae.quantize.embed_to_fhat(pred_maps, all_to_max_scale=True, last_one=True)
        src_top = source_context_per_scale[-1].transpose(1, 2).reshape(
            source_context_per_scale[-1].shape[0], self.vae.Cvae, self.max_pn, self.max_pn
        )
        fused = self.alpha * pred_latent + (1.0 - self.alpha) * src_top
        return self.vae.fhat_to_img(fused)

    def _parallel_generate(
        self,
        images: Tensor,
        labels: Tensor,
        differentiable_tokenize: bool,
        hard_quantization: bool,
    ) -> CycleVAROutput:
        contexts = self.tokenize_to_multiscale_context(images, differentiable=differentiable_tokenize)
        logits_all = self._var_forward_with_context(labels, contexts)

        logits_per_scale: List[Tensor] = []
        embeds_per_scale: List[Tensor] = []
        for bg, ed in self.var.begin_ends:
            logits_k = logits_all[:, bg:ed]
            logits_per_scale.append(logits_k)
            embeds_per_scale.append(self._srq_logits_to_embeds(logits_k, hard=hard_quantization))

        out_img = self._decode_from_predicted_scales(embeds_per_scale, contexts)
        fused_latent = self.alpha * self.vae.quantize.embed_to_fhat(
            [
                e.view(e.shape[0], pn, pn, e.shape[-1]).permute(0, 3, 1, 2).contiguous()
                for pn, e in zip(self.patch_nums, embeds_per_scale)
            ],
            all_to_max_scale=True,
            last_one=True,
        ) + (1.0 - self.alpha) * contexts[-1].transpose(1, 2).reshape(
            contexts[-1].shape[0], self.vae.Cvae, self.max_pn, self.max_pn
        )
        return CycleVAROutput(
            image=out_img,
            logits_per_scale=logits_per_scale,
            pred_embeds_per_scale=embeds_per_scale,
            source_context_per_scale=contexts,
            fused_latent=fused_latent,
        )

    def _serial_generate(
        self,
        images: Tensor,
        labels: Tensor,
        differentiable_tokenize: bool,
        hard_quantization: bool,
    ) -> CycleVAROutput:
        src_contexts = self.tokenize_to_multiscale_context(images, differentiable=differentiable_tokenize)
        fused_contexts: List[Tensor] = []
        logits_per_scale: List[Tensor] = []
        pred_embeds_per_scale: List[Tensor] = []

        for k in range(self.num_scales):
            prefix_contexts = list(fused_contexts)
            prefix_contexts.append(src_contexts[k])
            logits_prefix = self._var_forward_with_context(labels, prefix_contexts)

            bg, ed = self.var.begin_ends[k]
            logits_k = logits_prefix[:, bg:ed]
            pred_k = self._srq_logits_to_embeds(logits_k, hard=hard_quantization)

            logits_per_scale.append(logits_k)
            pred_embeds_per_scale.append(pred_k)

            pn = self.patch_nums[k]
            pred_k_map = pred_k.view(pred_k.shape[0], pn, pn, pred_k.shape[-1]).permute(0, 3, 1, 2).contiguous()
            src_k_map = src_contexts[k].transpose(1, 2).reshape(src_contexts[k].shape[0], self.vae.Cvae, pn, pn)
            fused_k_map = self.alpha * pred_k_map + (1.0 - self.alpha) * src_k_map
            fused_contexts.append(fused_k_map.flatten(2).transpose(1, 2).contiguous())

        out_img = self._decode_from_predicted_scales(pred_embeds_per_scale, src_contexts)
        fused_latent = self.alpha * self.vae.quantize.embed_to_fhat(
            [
                e.view(e.shape[0], pn, pn, e.shape[-1]).permute(0, 3, 1, 2).contiguous()
                for pn, e in zip(self.patch_nums, pred_embeds_per_scale)
            ],
            all_to_max_scale=True,
            last_one=True,
        ) + (1.0 - self.alpha) * src_contexts[-1].transpose(1, 2).reshape(
            src_contexts[-1].shape[0], self.vae.Cvae, self.max_pn, self.max_pn
        )
        return CycleVAROutput(
            image=out_img,
            logits_per_scale=logits_per_scale,
            pred_embeds_per_scale=pred_embeds_per_scale,
            source_context_per_scale=src_contexts,
            fused_latent=fused_latent,
        )

    def forward(
        self,
        images: Tensor,
        labels: Union[int, Tensor],
        mode: str = "parallel",
        differentiable_tokenize: bool = True,
        hard_quantization: bool = False,
        return_details: bool = False,
    ) -> Union[Tensor, CycleVAROutput]:
        bsz = images.shape[0]
        label_B = self._prepare_labels(labels, bsz, images.device)

        mode = mode.lower().strip()
        if mode == "parallel":
            out = self._parallel_generate(
                images=images,
                labels=label_B,
                differentiable_tokenize=differentiable_tokenize,
                hard_quantization=hard_quantization,
            )
        elif mode == "serial":
            out = self._serial_generate(
                images=images,
                labels=label_B,
                differentiable_tokenize=differentiable_tokenize,
                hard_quantization=hard_quantization,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return out if return_details else out.image
