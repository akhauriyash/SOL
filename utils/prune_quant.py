import math
from typing import Optional, Dict
import torch
import torch.nn.functional as F

# -------- Quantization helpers --------

def _qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1  # symmetric range [-2^(b-1), 2^(b-1)-1]

@torch.no_grad()
def quantize_weight_rowwise_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Row-wise (per-out-feature) symmetric quantization, dequantized back to original dtype.
    Returns a *tensor copy* (not a Parameter) you can pass to F.linear.
    """
    if bits >= 16:
        return w
    qmax = _qmax(bits)
    # scale per row: [out, 1]
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(w / scale).clamp_(-qmax - 1, qmax)
    return (q * scale).to(w.dtype)

@torch.no_grad()
def fake_quantize_activation_sym(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Per-tensor symmetric fake-quant on activations for numerical realism. No speedup intended.
    """
    if bits >= 16:
        return x
    qmax = _qmax(bits)
    s = x.detach().abs().amax().clamp_min(1e-8) / qmax
    q = torch.round(x / s).clamp_(-qmax - 1, qmax)
    return (q * s).to(x.dtype)

# -------- Activation pruning (magnitude top-k) --------

def prune_hidden_topk(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """
    Keep top-|.| channels along the last dim per token, zero out the rest.
    keep_ratio in (0,1]; keep_ratio=1.0 leaves x unchanged.
    """
    if keep_ratio >= 0.999:
        return x
    D = x.size(-1)
    k = max(1, int(math.ceil(keep_ratio * D)))
    # threshold via top-k
    thr = x.abs().kthvalue(D - k + 1, dim=-1).values  # kthvalue is stable & cheaper than topk for thresholds
    mask = (x.abs() >= thr.unsqueeze(-1))
    return x * mask.to(x.dtype)

# -------- Monkey-patch LlamaMLP to support runtime controls --------

def enable_prune_quant_controls(model) -> None:
    """
    Monkey-patch LlamaMLP.forward to honor:
      - module._rq_keep : float keep fraction in (0,1]
      - module._rq_bits : int {4,8,16}
    Exposes:
      model._rq_mlp_modules : list of patched MLP modules
    """
    from transformers.models.llama import modeling_llama as llama_mod
    LlamaMLP = llama_mod.LlamaMLP

    if not hasattr(llama_mod, "_rq_original_llama_mlp_forward"):
        llama_mod._rq_original_llama_mlp_forward = LlamaMLP.forward

        def forward_with_controls(self, x: torch.Tensor) -> torch.Tensor:
            enabled = bool(getattr(self, "_rq_enabled", False))
            if not enabled:
                return llama_mod._rq_original_llama_mlp_forward(self, x)

            keep = float(getattr(self, "_rq_keep", 1.0))
            bits = int(getattr(self, "_rq_bits", 16))

            # Choose weights (maybe quantized & cached)
            # Build (and memoize) per-bits dequantized copies to avoid recompute.
            cache: Dict[int, Dict[str, torch.Tensor]] = getattr(self, "_rq_qcache", None)
            if cache is None:
                cache = {}
                setattr(self, "_rq_qcache", cache)

            if bits >= 16:
                up_w   = self.up_proj.weight
                gate_w = self.gate_proj.weight
                down_w = self.down_proj.weight
            else:
                layer_cache = cache.get(bits)
                if layer_cache is None:
                    layer_cache = {
                        "up":   quantize_weight_rowwise_symmetric(self.up_proj.weight, bits),
                        "gate": quantize_weight_rowwise_symmetric(self.gate_proj.weight, bits),
                        "down": quantize_weight_rowwise_symmetric(self.down_proj.weight, bits),
                    }
                    cache[bits] = layer_cache
                up_w, gate_w, down_w = layer_cache["up"], layer_cache["gate"], layer_cache["down"]

            gate = F.linear(x, gate_w, self.gate_proj.bias)
            up   = F.linear(x,   up_w, self.up_proj.bias)
            hidden = F.silu(gate) * up

            # Activation fake-quant AFTER nonlinearity (matches many QAT recipes)
            if bits < 16:
                hidden = fake_quantize_activation_sym(hidden, bits)

            # Magnitude-based top-k pruning along channel dim
            if keep < 0.999:
                hidden = prune_hidden_topk(hidden, keep)

            out = F.linear(hidden, down_w, self.down_proj.bias)
            return out

        LlamaMLP.forward = forward_with_controls

    # Attach control attributes to each LlamaMLP
    mlps = []
    for m in model.modules():
        if isinstance(m, llama_mod.LlamaMLP):
            m._rq_enabled = True
            m._rq_keep = 1.0         # default: no pruning
            m._rq_bits = 16          # default: no quantization
            m._rq_qcache = {}        # per-bits quantized weights cache
            mlps.append(m)
    model._rq_mlp_modules = mlps

def set_runtime_prune_keep(model, keep: float) -> None:
    """Set the keep fraction for all patched MLPs."""
    if not getattr(model, "_rq_mlp_modules", None):
        return
    for m in model._rq_mlp_modules:
        m._rq_keep = float(keep)

def set_runtime_quant_bits(model, bits: int) -> None:
    """Set the quantization precision for all patched MLPs."""
    if not getattr(model, "_rq_mlp_modules", None):
        return
    for m in model._rq_mlp_modules:
        m._rq_bits = int(bits)

def clear_prune_quant(model) -> None:
    """Restore defaults."""
    set_runtime_prune_keep(model, 1.0)
    set_runtime_quant_bits(model, 16)
    
# --- Convenience wrappers to match your evaluator names ---
def enable_structured_controls(model) -> None:
    enable_prune_quant_controls(model)

@torch.no_grad()
def set_structured_action(model, prune_keep: float, quant_bits: int) -> None:
    set_runtime_prune_keep(model, prune_keep)
    set_runtime_quant_bits(model, quant_bits)

def clear_structured_action(model) -> None:
    clear_prune_quant(model)
