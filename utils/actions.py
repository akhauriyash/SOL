from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class ActionSpec:
    """Flattened grid of composite actions."""
    token_keep: List[float]   # len = N
    prune_keep: List[float]   # len = N  (fraction of channels to keep)
    q_bits:     List[int]     # len = N  (4, 8, 16 -> 16 means identity)
    tags:       List[str]     # human-readable tags like "k100-s100-q16"
    dense_idx:  int           # index of (keep=1.0, prune_keep=1.0, q_bits=16) or best available

    @property
    def n_actions(self) -> int:
        return len(self.token_keep)


def _parse_prune_choices(choices: Sequence[str]) -> List[float]:
    """
    Map strings to keep-fractions:
      s40 -> keep 0.60; s70 -> keep 0.30; s100 -> keep 1.0
    """
    out = []
    for c in choices:
        c = c.strip().lower()
        # out.append(1 - float(c.split("s")[-1])/100.0)
        out.append(float(c.split("s")[-1])/100.0)
    return out


def _parse_quant_choices(choices: Sequence[str]) -> List[int]:
    """
    Map strings to activation bitwidths for FFNs.
    """
    out = []
    for c in choices:
        s = c.strip().lower()
        if not (s.startswith("q") and s[1:].isdigit()):
            raise ValueError(f"Unknown quant choice '{c}' (expected q{N}).")
        bits = int(s[1:])
        if not (2 <= bits <= 16):
            raise ValueError(f"bits must be in [2,16], got {bits}.")
        out.append(bits)
    return out



def build_action_spec(
    *,
    keep_fracs: Sequence[float],
    prune_choices: Sequence[str] = ("s100",),     # default: no structural pruning
    quant_choices: Sequence[str] = ("q16",),     # default: no quantization
) -> ActionSpec:
    """
    Cartesian product of (token_keep) × (prune_keep) × (q_bits).
    """
    k_list = list(float(k) for k in keep_fracs)
    p_list = _parse_prune_choices(prune_choices)
    q_list = _parse_quant_choices(quant_choices)

    token_keep: List[float] = []
    prune_keep: List[float] = []
    q_bits:     List[int]   = []
    tags:       List[str]   = []

    for k in k_list:
        for p in p_list:
            for q in q_list:
                token_keep.append(k)
                prune_keep.append(p)
                q_bits.append(q)
                k_pct = int(round(k * 100))
                p_pct = int(round(p * 100))
                tags.append(f"k{k_pct:03d}-s{p_pct:03d}-q{q}")

    # dense = keep=1.0 AND prune_keep=1.0 AND q=16 (if present), otherwise best effort.
    dense_idx = 0
    for i, (k, p, q) in enumerate(zip(token_keep, prune_keep, q_bits)):
        if abs(k - 1.0) < 1e-6 and abs(p - 1.0) < 1e-6 and int(q) == 16:
            dense_idx = i
            break
    else:
        # fallback: pick any action with keep=1.0
        for i, k in enumerate(token_keep):
            if abs(k - 1.0) < 1e-6:
                dense_idx = i
                break

    return ActionSpec(token_keep, prune_keep, q_bits, tags, dense_idx)