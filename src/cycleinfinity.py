import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel


def _default_infinity_root() -> Path:
    env_root = os.environ.get("INFINITY_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # /workspace/CycleVAR/src/cycleinfinity.py -> /workspace/Infinity
    return (Path(__file__).resolve().parents[2] / "Infinity").resolve()


def _ensure_infinity_on_path(infinity_root: Path) -> None:
    if not infinity_root.exists():
        raise FileNotFoundError(f"Infinity root not found: {infinity_root}")
    root_str = str(infinity_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _strip_prefix_if_needed(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    return sd


def _extract_infinity_state(sd: Dict) -> Dict:
    if not isinstance(sd, dict):
        return sd
    trainer = sd.get("trainer")
    if isinstance(trainer, dict):
        for key in ("gpt_fsdp", "gpt_ema_fsdp", "gpt_wo_ddp", "gpt"):
            if key in trainer and isinstance(trainer[key], dict):
                return trainer[key]
    for key in ("infinity_state_dict", "state_dict", "model", "gpt_wo_ddp", "gpt_fsdp"):
        if key in sd and isinstance(sd[key], dict):
            return sd[key]
    return sd


def _infinity_model_kwargs(model_type: str) -> Dict:
    table = {
        "infinity_2b": dict(depth=32, embed_dim=2048, num_heads=2048 // 128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
        "infinity_layer12": dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer16": dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer24": dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer32": dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer40": dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer48": dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
    }
    if model_type not in table:
        raise ValueError(f"Unsupported infinity model_type: {model_type}")
    return table[model_type]


class CycleInfinity(nn.Module):
    def __init__(
        self,
        *,
        infinity_model_path: str,
        infinity_vae_path: str,
        infinity_text_encoder_ckpt: str,
        cycleinfinity_ckpt_path: Optional[str] = None,
        model_type: str = "infinity_2b",
        pn: str = "0.06M",
        rope2d_each_sa_layer: int = 1,
        rope2d_normalized_by_hw: int = 2,
        add_lvl_embeding_only_first_block: int = 1,
        use_bit_label: int = 1,
        apply_spatial_patchify: int = 0,
        use_flex_attn: int = 0,
        prompt_a: Optional[str] = None,
        prompt_b: Optional[str] = None,
        srq_temperature: float = 2.0,
        source_temperature: float = 1.0,
        use_srq_gumbel: bool = False,
        src_fusion_alpha: float = 1.0,
        debug_nan_guard: bool = False,
        debug_nan_abort: bool = True,
    ):
        super().__init__()
        if int(use_bit_label) != 1:
            raise ValueError("CycleInfinity currently supports only --infinity_use_bit_label=1 (bit labels).")

        infinity_root = _default_infinity_root()
        _ensure_infinity_on_path(infinity_root)

        from infinity.models.bsq_vae.vae import vae_model
        from infinity.models.infinity import Infinity
        from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

        self._dynamic_resolution_h_w = dynamic_resolution_h_w
        self._h_div_w_templates = np.array(h_div_w_templates)

        self.infinity_model_type = str(model_type)
        self.infinity_pn = str(pn)
        self.infinity_rope2d_each_sa_layer = int(rope2d_each_sa_layer)
        self.infinity_rope2d_normalized_by_hw = int(rope2d_normalized_by_hw)
        self.infinity_add_lvl_embeding_only_first_block = int(add_lvl_embeding_only_first_block)
        self.infinity_use_bit_label = int(use_bit_label)
        self.infinity_apply_spatial_patchify = int(apply_spatial_patchify)
        self.infinity_use_flex_attn = int(use_flex_attn)

        self.srq_temperature = float(srq_temperature)
        self.source_temperature = float(source_temperature)
        self.use_srq_gumbel = bool(use_srq_gumbel)
        self.src_fusion_alpha = float(src_fusion_alpha)
        self.debug_nan_guard = bool(debug_nan_guard)
        self.debug_nan_abort = bool(debug_nan_abort)

        self.infinity_model_path = infinity_model_path
        self.infinity_vae_path = infinity_vae_path
        self.infinity_text_encoder_ckpt = infinity_text_encoder_ckpt

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_dtype = torch.float16 if device.type == "cuda" else torch.float32

        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(infinity_text_encoder_ckpt, revision=None, legacy=True)
        except TypeError:
            self.text_tokenizer = AutoTokenizer.from_pretrained(infinity_text_encoder_ckpt, revision=None)
        self.text_tokenizer.model_max_length = 512
        self.text_encoder = T5EncoderModel.from_pretrained(infinity_text_encoder_ckpt, torch_dtype=text_dtype)
        self.text_encoder.to(device=device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        codebook_dim = 32
        codebook_size = 2 ** codebook_dim
        if self.infinity_apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult = [1, 2, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult = [1, 2, 4, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4, 4]

        self.vae = vae_model(
            infinity_vae_path,
            schedule_mode="dynamic",
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            test_mode=True,
            patch_size=patch_size,
            encoder_ch_mult=encoder_ch_mult,
            decoder_ch_mult=decoder_ch_mult,
        ).to(device=device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        model_kwargs = _infinity_model_kwargs(self.infinity_model_type)
        base_schedule = self._dynamic_resolution_h_w[1.0][self.infinity_pn]["scales"]
        self.infinity = Infinity(
            vae_local=self.vae,
            text_channels=int(self.text_encoder.config.d_model),
            text_maxlen=512,
            shared_aln=True,
            raw_scale_schedule=base_schedule,
            checkpointing="full-block",
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=bool(self.infinity_use_flex_attn),
            add_lvl_embeding_only_first_block=self.infinity_add_lvl_embeding_only_first_block,
            use_bit_label=self.infinity_use_bit_label,
            rope2d_each_sa_layer=self.infinity_rope2d_each_sa_layer,
            rope2d_normalized_by_hw=self.infinity_rope2d_normalized_by_hw,
            pn=self.infinity_pn,
            apply_spatial_patchify=self.infinity_apply_spatial_patchify,
            inference_mode=False,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        self.infinity.cond_drop_rate = 0.0

        sd = torch.load(infinity_model_path, map_location="cpu")
        sd = _strip_prefix_if_needed(_extract_infinity_state(sd))
        missing, unexpected = self.infinity.load_state_dict(sd, strict=False)
        if missing:
            print(f"[CycleInfinity] Infinity missing keys: {len(missing)}")
        if unexpected:
            print(f"[CycleInfinity] Infinity unexpected keys: {len(unexpected)}")

        self.prompt_a: Optional[str] = None
        self.prompt_b: Optional[str] = None

        if cycleinfinity_ckpt_path is not None:
            self.load_cycleinfinity_ckpt(cycleinfinity_ckpt_path)

        if prompt_a is not None:
            self.prompt_a = prompt_a.strip()
        if prompt_b is not None:
            self.prompt_b = prompt_b.strip()
        if not self.prompt_a or not self.prompt_b:
            raise ValueError("CycleInfinity requires both prompt_a and prompt_b (via args or checkpoint).")

        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen conditioners/tokenizer path in eval mode.
        self.vae.eval()
        self.text_encoder.eval()
        return self

    def _check_finite(self, name: str, t: torch.Tensor) -> None:
        if not self.debug_nan_guard:
            return
        if torch.isfinite(t).all():
            return
        msg = f"[CycleInfinity nan-guard] non-finite tensor detected: {name}"
        if self.debug_nan_abort:
            raise RuntimeError(msg)
        print(msg)

    def _scale_schedule_for_input(self, h: int, w: int) -> List[Tuple[int, int, int]]:
        ratio = float(h) / float(w)
        template = float(self._h_div_w_templates[np.argmin(np.abs(self._h_div_w_templates - ratio))])
        raw = self._dynamic_resolution_h_w[template][self.infinity_pn]["scales"]
        return [(1, int(ph), int(pw)) for _, ph, pw in raw]

    def _vae_scale_schedule(self, scale_schedule: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        if self.infinity_apply_spatial_patchify:
            return [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
        return scale_schedule

    def _encode_prompt_batch(self, prompt: str, batch_size: int, device: torch.device):
        prompts = [prompt] * batch_size
        tokens = self.text_tokenizer(
            text=prompts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device=device, non_blocking=True)
        mask = tokens.attention_mask.to(device=device, non_blocking=True)

        with torch.no_grad():
            text_features = self.text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()

        lens = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        max_seqlen_k = max(lens)
        kv_compact = torch.cat([feat[:ln] for ln, feat in zip(lens, text_features.unbind(0))], dim=0)
        return kv_compact, lens, cu_seqlens_k, max_seqlen_k

    def _encode_source(self, x: torch.Tensor, scale_schedule: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        vae_scale_schedule = self._vae_scale_schedule(scale_schedule)
        with torch.no_grad():
            _, src_latent, _, _, _, var_input = self.vae.encode(x, scale_schedule=vae_scale_schedule)

        if src_latent.dim() == 5 and src_latent.shape[-3] == 1:
            src_latent = src_latent.squeeze(-3)
        src_latent = src_latent.float()

        x_parts: List[torch.Tensor] = []
        for item in var_input:
            if item.dim() == 5 and item.shape[-3] == 1:
                item = item.squeeze(-3)
            if item.dim() != 4:
                raise ValueError(f"Unexpected VAE var_input shape: {tuple(item.shape)}")
            if self.infinity_apply_spatial_patchify:
                item = torch.nn.functional.pixel_unshuffle(item, 2)
            x_parts.append(item.reshape(item.shape[0], item.shape[1], -1).permute(0, 2, 1).contiguous())

        if len(x_parts) == 0:
            raise RuntimeError("VAE returned empty var_input; cannot build Infinity teacher-forcing input.")
        x_blc_wo_prefix = torch.cat(x_parts, dim=1).float()
        return x_blc_wo_prefix, src_latent

    def _decode_logits_to_latent(
        self,
        logits_blv: torch.Tensor,
        scale_schedule: List[Tuple[int, int, int]],
        hard_decode: bool,
    ) -> torch.Tensor:
        expected_len = sum(int(pt * ph * pw) for pt, ph, pw in scale_schedule)
        if int(logits_blv.shape[1]) != expected_len:
            raise ValueError(f"Infinity logits sequence length mismatch: got {logits_blv.shape[1]}, expected {expected_len}")

        if logits_blv.shape[-1] % 2 != 0:
            raise ValueError(f"Infinity bitwise logits shape invalid: V={logits_blv.shape[-1]} (must be even)")

        vae_scale_schedule = self._vae_scale_schedule(scale_schedule)
        ptr = 0
        summed_codes = None
        tau = max(float(self.srq_temperature), 1e-6)

        for pt, ph, pw in scale_schedule:
            seq_len = int(pt * ph * pw)
            stage_logits = logits_blv[:, ptr:ptr + seq_len, :]
            ptr += seq_len
            stage_logits = stage_logits.reshape(stage_logits.shape[0], seq_len, -1, 2)

            if hard_decode:
                bit_prob = stage_logits.argmax(dim=-1).float()
            else:
                scaled = stage_logits.float() / tau
                if self.use_srq_gumbel:
                    probs = F.gumbel_softmax(scaled, tau=1.0, hard=False, dim=-1)
                else:
                    probs = torch.softmax(scaled, dim=-1)
                bit_prob = probs[..., 1]

            bits = bit_prob.reshape(bit_prob.shape[0], pt, ph, pw, bit_prob.shape[-1])
            if self.infinity_apply_spatial_patchify:
                if pt != 1:
                    raise ValueError("CycleInfinity currently supports image generation only (pt=1).")
                bits_2d = bits.squeeze(1).permute(0, 3, 1, 2).contiguous()
                bits_2d = torch.nn.functional.pixel_shuffle(bits_2d, 2)
                bits = bits_2d.permute(0, 2, 3, 1).unsqueeze(1).contiguous()

            stage_codes = self.vae.quantizer.lfq.indices_to_codes(bits, label_type="bit_label")
            stage_codes = F.interpolate(stage_codes, size=vae_scale_schedule[-1], mode=self.vae.quantizer.z_interplote_up)
            summed_codes = stage_codes if summed_codes is None else (summed_codes + stage_codes)

        if ptr != logits_blv.shape[1]:
            raise RuntimeError(f"Logit decode cursor mismatch: ptr={ptr}, total={logits_blv.shape[1]}")

        pred_latent = summed_codes.squeeze(-3) if summed_codes.dim() == 5 and summed_codes.shape[-3] == 1 else summed_codes
        return pred_latent.float()

    def _prompt_for_direction(self, direction: str) -> str:
        if direction == "a2b":
            return self.prompt_b
        if direction == "b2a":
            return self.prompt_a
        raise ValueError(f"direction must be one of ['a2b', 'b2a'], got {direction}")

    def forward(self, x: torch.Tensor, direction: str, hard_decode: bool = False) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"CycleInfinity expects BCHW inputs, got shape={tuple(x.shape)}")
        scale_schedule = self._scale_schedule_for_input(x.shape[-2], x.shape[-1])
        text_cond_tuple = self._encode_prompt_batch(self._prompt_for_direction(direction), x.shape[0], x.device)
        x_blc_wo_prefix, src_latent = self._encode_source(x, scale_schedule)

        logits_blv = self.infinity(text_cond_tuple, x_blc_wo_prefix, scale_schedule=scale_schedule)
        self._check_finite("logits_blv", logits_blv)
        pred_latent = self._decode_logits_to_latent(logits_blv, scale_schedule, hard_decode=hard_decode)
        self._check_finite("pred_latent", pred_latent)

        if self.src_fusion_alpha < 1.0:
            alpha = float(self.src_fusion_alpha)
            pred_latent = alpha * pred_latent + (1.0 - alpha) * src_latent

        out = self.vae.decode(pred_latent).clamp(-1, 1)
        self._check_finite("out", out)
        return out

    def load_cycleinfinity_ckpt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            return
        sd = ckpt.get("infinity_state_dict", ckpt.get("state_dict", None))
        if isinstance(sd, dict):
            sd = _strip_prefix_if_needed(sd)
            missing, unexpected = self.infinity.load_state_dict(sd, strict=False)
            if missing:
                print(f"[CycleInfinity] ckpt missing keys: {len(missing)}")
            if unexpected:
                print(f"[CycleInfinity] ckpt unexpected keys: {len(unexpected)}")

        if "prompt_a" in ckpt and ckpt["prompt_a"] is not None:
            self.prompt_a = str(ckpt["prompt_a"]).strip()
        if "prompt_b" in ckpt and ckpt["prompt_b"] is not None:
            self.prompt_b = str(ckpt["prompt_b"]).strip()

        self.srq_temperature = float(ckpt.get("srq_temperature", self.srq_temperature))
        self.source_temperature = float(ckpt.get("source_temperature", self.source_temperature))
        self.use_srq_gumbel = bool(ckpt.get("use_srq_gumbel", self.use_srq_gumbel))
        self.src_fusion_alpha = float(ckpt.get("src_fusion_alpha", self.src_fusion_alpha))

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.infinity.parameters() if p.requires_grad]

    def export_checkpoint(self) -> Dict:
        return {
            "generator_backbone": "infinity",
            "infinity_state_dict": self.infinity.state_dict(),
            "infinity_model_type": self.infinity_model_type,
            "infinity_pn": self.infinity_pn,
            "infinity_use_bit_label": self.infinity_use_bit_label,
            "infinity_apply_spatial_patchify": self.infinity_apply_spatial_patchify,
            "infinity_rope2d_each_sa_layer": self.infinity_rope2d_each_sa_layer,
            "infinity_rope2d_normalized_by_hw": self.infinity_rope2d_normalized_by_hw,
            "infinity_add_lvl_embeding_only_first_block": self.infinity_add_lvl_embeding_only_first_block,
            "infinity_use_flex_attn": self.infinity_use_flex_attn,
            "prompt_a": self.prompt_a,
            "prompt_b": self.prompt_b,
            "srq_temperature": self.srq_temperature,
            "source_temperature": self.source_temperature,
            "use_srq_gumbel": self.use_srq_gumbel,
            "src_fusion_alpha": self.src_fusion_alpha,
            "infinity_model_path": self.infinity_model_path,
            "infinity_vae_path": self.infinity_vae_path,
            "infinity_text_encoder_ckpt": self.infinity_text_encoder_ckpt,
        }
