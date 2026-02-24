import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 1) /workspace/CycleVAR/models
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models import build_vae_var
except ModuleNotFoundError:
    # 2) /workspace/CycleVAR/VAR/models
    var_root = PROJECT_ROOT / "VAR"
    if str(var_root) not in sys.path:
        sys.path.insert(0, str(var_root))
    from models import build_vae_var



def parse_patch_nums(patch_nums: Sequence[int] | str) -> Tuple[int, ...]:
    if isinstance(patch_nums, str):
        parsed = [int(tok.strip()) for tok in patch_nums.split(",") if tok.strip()]
        if not parsed:
            raise ValueError("patch_nums string must contain at least one integer")
        return tuple(parsed)
    return tuple(int(x) for x in patch_nums)


def _strip_prefix_if_needed(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    return sd


def _extract_sub_state(ckpt: Dict, candidates: Iterable[str]) -> Dict:
    if not isinstance(ckpt, dict):
        return ckpt
    for key in candidates:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


class SRQQuantizer(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, logits: torch.Tensor, temperature: float, use_gumbel: bool = False) -> torch.Tensor:
        # logits: B x L x V -> embeddings: B x L x C
        tau = max(float(temperature), 1e-6)
        logits_fp32 = logits.float()
        if use_gumbel:
            # Gumbel(0,1) noise, used in the paper's SRQ ablation.
            gumbel = -torch.empty_like(logits_fp32).exponential_().log()
            logits_fp32 = logits_fp32 + gumbel
        probs = torch.softmax(logits_fp32 / tau, dim=-1)
        return probs @ self.embedding.weight.float()


class CycleVAR(nn.Module):
    def __init__(
        self,
        vqvae_ckpt_path: Optional[str] = None,
        var_ckpt_path: Optional[str] = None,
        cyclevar_ckpt_path: Optional[str] = None,
        *,
        patch_nums: Sequence[int] | str = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        var_depth: int = 16,
        num_classes: int = 1000,
        label_a: int = 0,
        label_b: int = 1,
        srq_temperature: float = 2.0,
        source_temperature: float = 1.0,
        use_srq_gumbel: bool = False,
        use_source_ste: bool = True,
        src_fusion_alpha: float = 1.0,
        debug_nan_guard: bool = False,
        debug_nan_abort: bool = True,
    ):
        super().__init__()
        self.patch_nums = parse_patch_nums(patch_nums)

        vae_local, var_model = build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,
            device="cuda",
            patch_nums=self.patch_nums,
            num_classes=num_classes,
            depth=var_depth,
            shared_aln=False,
            attn_l2_norm=True,
            flash_if_available=False,
            fused_if_available=True,
            init_adaln=0.5,
            init_adaln_gamma=1e-5,
            init_head=0.02,
            init_std=-1,
        )

        self.vae = vae_local
        self.var = var_model
        # CycleVAR uses explicit target-domain conditions; disable classifier-free label dropout.
        self.var.cond_drop_rate = 0.0
        self.num_classes = int(num_classes)

        self.srq = SRQQuantizer(self.vae.quantize.embedding)
        self.srq_temperature = float(srq_temperature)
        self.source_temperature = float(source_temperature)
        self.use_srq_gumbel = bool(use_srq_gumbel)
        self.use_source_ste = bool(use_source_ste)
        self.src_fusion_alpha = float(src_fusion_alpha)
        self.debug_nan_guard = bool(debug_nan_guard)
        self.debug_nan_abort = bool(debug_nan_abort)

        self.label_a = int(label_a)
        self.label_b = int(label_b)
        self._has_loaded_vqvae = False

        if vqvae_ckpt_path is not None:
            self.load_vqvae_ckpt(vqvae_ckpt_path)
            self._has_loaded_vqvae = True
        if var_ckpt_path is not None:
            self.load_var_ckpt(var_ckpt_path)

        # Freeze visual tokenizer as in CycleVAR.
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        if cyclevar_ckpt_path is not None:
            self.load_cyclevar_ckpt(cyclevar_ckpt_path)

    @property
    def begin_ends(self) -> List[Tuple[int, int]]:
        return self.var.begin_ends

    @staticmethod
    def _pairwise_neg_sqdist(z_flat: torch.Tensor, emb_weight: torch.Tensor) -> torch.Tensor:
        z_sq = (z_flat ** 2).sum(dim=-1, keepdim=True)
        e_sq = (emb_weight ** 2).sum(dim=-1).unsqueeze(0)
        return -(z_sq + e_sq - 2.0 * (z_flat @ emb_weight.t()))

    def _stage_tokens_from_source(self, z_blc: torch.Tensor) -> torch.Tensor:
        emb = self.vae.quantize.embedding.weight.float()
        z_flat = z_blc.reshape(-1, z_blc.shape[-1]).float()

        neg_d = self._pairwise_neg_sqdist(z_flat, emb)
        hard_idx = torch.argmax(neg_d, dim=-1)
        hard_embed = emb.index_select(0, hard_idx)

        soft_embed = self.srq(neg_d.unsqueeze(0), temperature=self.source_temperature, use_gumbel=False).squeeze(0)

        if self.use_source_ste:
            # Straight-through estimator: hard in forward, soft in backward.
            out = soft_embed + (hard_embed - soft_embed).detach()
        else:
            out = hard_embed

        return out.view(z_blc.shape[0], z_blc.shape[1], z_blc.shape[2])

    def _encode_source_to_var_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_src = self.vae.quant_conv(self.vae.encoder(x))
        bsz, cvae, h, w = f_src.shape
        expected_hw = self.patch_nums[-1]
        if (h, w) != (expected_hw, expected_hw):
            raise ValueError(
                f"Tokenizer feature size mismatch: got {(h, w)}, expected {(expected_hw, expected_hw)}. "
                "Use a VAR checkpoint/patch schedule matching your input resolution."
            )

        sn = len(self.patch_nums)
        f_rest = f_src
        f_hat = torch.zeros_like(f_src)
        stage_maps: List[torch.Tensor] = []

        for si, pn in enumerate(self.patch_nums):
            if si != sn - 1:
                z = F.interpolate(f_rest, size=(pn, pn), mode="area")
            else:
                z = f_rest
            z_blc = z.flatten(2).transpose(1, 2)
            h_blc = self._stage_tokens_from_source(z_blc)
            h_bchw = h_blc.transpose(1, 2).reshape(bsz, cvae, pn, pn)
            stage_maps.append(h_bchw)

            h_up = F.interpolate(h_bchw, size=(h, w), mode="bicubic") if si != sn - 1 else h_bchw
            h_up = self.vae.quantize.quant_resi[si / max(sn - 1, 1)](h_up)
            f_hat = f_hat + h_up
            f_rest = f_rest - h_up

        # Build teacher-forcing input exactly as VAR does for next-scale prediction.
        tf_inputs: List[torch.Tensor] = []
        f_partial = torch.zeros_like(f_hat)
        for si in range(sn - 1):
            pn = self.patch_nums[si]
            pn_next = self.patch_nums[si + 1]
            h_si = stage_maps[si]
            h_up = F.interpolate(h_si, size=(h, w), mode="bicubic")
            h_up = self.vae.quantize.quant_resi[si / max(sn - 1, 1)](h_up)
            f_partial = f_partial + h_up
            next_scale = F.interpolate(f_partial, size=(pn_next, pn_next), mode="area")
            tf_inputs.append(next_scale.flatten(2).transpose(1, 2))

        x_var = torch.cat(tf_inputs, dim=1).float()
        return x_var, f_hat

    def _decode_from_logits(self, logits_blv: torch.Tensor, hard: bool = False) -> torch.Tensor:
        bsz = logits_blv.shape[0]
        cvae = self.vae.Cvae
        ms_h: List[torch.Tensor] = []

        for si, (bg, ed) in enumerate(self.begin_ends):
            pn = self.patch_nums[si]
            logits_stage = logits_blv[:, bg:ed, :]
            if hard:
                idx = logits_stage.argmax(dim=-1)
                h_blc = self.vae.quantize.embedding(idx)
            else:
                h_blc = self.srq(logits_stage, temperature=self.srq_temperature, use_gumbel=self.use_srq_gumbel)
            h_bchw = h_blc.transpose(1, 2).reshape(bsz, cvae, pn, pn)
            ms_h.append(h_bchw)

        f_hat = self.vae.quantize.embed_to_fhat(ms_h_BChw=ms_h, all_to_max_scale=True, last_one=True)
        return f_hat

    def _label_for_direction(self, direction: str, bsz: int, device: torch.device) -> torch.Tensor:
        if direction not in {"a2b", "b2a"}:
            raise ValueError(f"direction must be one of ['a2b', 'b2a'], got {direction}")
        if direction == "a2b":
            label = self.label_b
        else:
            label = self.label_a
        return torch.full((bsz,), fill_value=label, dtype=torch.long, device=device)

    def forward(self, x: torch.Tensor, direction: str, hard_decode: bool = False) -> torch.Tensor:
        bsz = x.shape[0]
        labels = self._label_for_direction(direction, bsz, x.device)
        x_var, src_f_hat = self._encode_source_to_var_input(x)
        logits = self.var(labels, x_var)
        pred_f_hat = self._decode_from_logits(logits, hard=hard_decode)

        if self.src_fusion_alpha < 1.0:
            alpha = self.src_fusion_alpha
            pred_f_hat = alpha * pred_f_hat + (1.0 - alpha) * src_f_hat

        out = self.vae.fhat_to_img(pred_f_hat).clamp(-1, 1)
        return out

    def load_vqvae_ckpt(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = _extract_sub_state(sd, candidates=("vae_local", "vae", "state_dict"))
        sd = _strip_prefix_if_needed(sd)
        missing, unexpected = self.vae.load_state_dict(sd, strict=False)
        if missing:
            print(f"[CycleVAR] VQVAE missing keys: {len(missing)}")
        if unexpected:
            print(f"[CycleVAR] VQVAE unexpected keys: {len(unexpected)}")

    def load_var_ckpt(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "trainer" in sd and isinstance(sd["trainer"], dict):
            sd = _extract_sub_state(sd["trainer"], candidates=("var_wo_ddp", "var"))
        else:
            sd = _extract_sub_state(sd, candidates=("var_wo_ddp", "var", "state_dict"))
        sd = _strip_prefix_if_needed(sd)
        missing, unexpected = self.var.load_state_dict(sd, strict=False)
        if missing:
            print(f"[CycleVAR] VAR missing keys: {len(missing)}")
        if unexpected:
            print(f"[CycleVAR] VAR unexpected keys: {len(unexpected)}")

    def load_cyclevar_ckpt(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict):
            if "vqvae_state_dict" in sd and isinstance(sd["vqvae_state_dict"], dict):
                vq_sd = _strip_prefix_if_needed(sd["vqvae_state_dict"])
                self.vae.load_state_dict(vq_sd, strict=False)
                self._has_loaded_vqvae = True
            elif (not self._has_loaded_vqvae) and isinstance(sd.get("vqvae_ckpt_path"), str):
                vq_path = sd.get("vqvae_ckpt_path")
                if vq_path and os.path.exists(vq_path):
                    self.load_vqvae_ckpt(vq_path)
                    self._has_loaded_vqvae = True

        if "var_state_dict" in sd:
            model_sd = sd["var_state_dict"]
        elif "state_dict" in sd:
            model_sd = sd["state_dict"]
        else:
            model_sd = sd
        model_sd = _strip_prefix_if_needed(model_sd)
        missing, unexpected = self.var.load_state_dict(model_sd, strict=False)
        if missing:
            print(f"[CycleVAR] CycleVAR ckpt missing keys: {len(missing)}")
        if unexpected:
            print(f"[CycleVAR] CycleVAR ckpt unexpected keys: {len(unexpected)}")

        self.label_a = int(sd.get("label_a", self.label_a)) if isinstance(sd, dict) else self.label_a
        self.label_b = int(sd.get("label_b", self.label_b)) if isinstance(sd, dict) else self.label_b
        self.srq_temperature = float(sd.get("srq_temperature", self.srq_temperature)) if isinstance(sd, dict) else self.srq_temperature
        self.source_temperature = float(sd.get("source_temperature", self.source_temperature)) if isinstance(sd, dict) else self.source_temperature
        self.src_fusion_alpha = float(sd.get("src_fusion_alpha", self.src_fusion_alpha)) if isinstance(sd, dict) else self.src_fusion_alpha

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.var.parameters() if p.requires_grad]

    def export_checkpoint(self) -> Dict:
        return {
            "var_state_dict": self.var.state_dict(),
            "label_a": self.label_a,
            "label_b": self.label_b,
            "patch_nums": list(self.patch_nums),
            "num_classes": self.num_classes,
            "var_depth": self.var.depth,
            "srq_temperature": self.srq_temperature,
            "source_temperature": self.source_temperature,
            "use_srq_gumbel": self.use_srq_gumbel,
            "use_source_ste": self.use_source_ste,
            "src_fusion_alpha": self.src_fusion_alpha,
        }
