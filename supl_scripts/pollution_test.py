
#!/usr/bin/env python3
"""
kv_pollution_impulse.py

A more convincing KV-pollution study, written for SOL-style “per-token efficiency actions”.

What this measures
------------------
For each sampled window:
  1) Dense prefill on P tokens to seed KV cache.
  2) Teacher-force C = t_corrupt tokens under a *single* aggressive approximation axis:
        - token sparsity (Quest or recency mask)
        - MLP activation pruning (structured channel keep-rate)
        - MLP activation quantization (fake-quant on MLP output)
  3) Switch back to fully dense decoding for the next T tail tokens.
  4) Measure tail-only degradation vs a fully-dense baseline using:
        - per-position ΔNLL (nats), and
        - per-position %Δppl = (exp(ΔNLL)-1)*100

This yields an “impulse response” curve: how pollution propagates downstream after we return to dense.

Figure
------
Produces 1 figure with 3 columns:
    [Token sparsity] [MLP pruning] [MLP quantization]
shared Y axis = %Δppl (tail-only), X axis = token offset after corruption.

It also prints and saves summary statistics per axis/level:
  - mean tail %Δppl
  - mean of worst-10% tail windows
  - p90/p95 of peak tail %Δppl

Assumptions / requirements
-------------------------
- Model is a HF LLaMA-family model where your monkey-patches apply.
- Quest mode requires eager attention implementation.
- Your helper functions exist:
    enable_quest_attention, build_sparse_attention_bias, clear_quest_token_budgets,
    enable_structured_controls, set_structured_action, clear_structured_action,
    clear_relevancy_keep (optional)
Adjust the imports below to match your repo.

Typical usage
-------------
python kv_pollution_impulse.py \
  --model meta-llama/Llama-3.2-1B \
  --dataset wikitext --dataset_config wikitext-2-raw-v1 --split test \
  --prefill_len 512 --t_corrupt 4 --tail_len 64 \
  --num_windows 64 --batch_size 4 \
  --sparsity_mode quest --quest_page_size 8 \
  --ratios 0.3 0.5 0.7 \
  --out_dir ./kv_pollution_out

If you want explicit axis levels instead of ratios:
  --kappa_list 0.15 0.3 0.6
  --prune_list 0.5 0.7 0.9
  --qbits_list 6 8 12
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


# -----------------------------
# Adjust these imports to your repo
# -----------------------------
try:
    # You already have these in your code snippets / utils
    from utils.masks import (
        build_sparse_attention_bias,
        enable_quest_attention,
        clear_quest_token_budgets,
        enable_structured_controls,
        set_structured_action,
        clear_structured_action,
        clear_relevancy_keep,
    )
except Exception as e:
    raise ImportError(
        "Could not import your utils.* patches. Update the imports in kv_pollution_impulse.py.\n"
        f"Original error: {e}"
    )


# -----------------------------
# Cache helpers
# -----------------------------
def _to_legacy_cache(cache):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def clone_cache_as_dynamic(cache) -> DynamicCache:
    """
    Deep-clone any HF cache (DynamicCache or legacy tuple) into a fresh DynamicCache.
    This is crucial: cache objects are mutated/appended during decoding.
    """
    legacy = _to_legacy_cache(cache)
    cloned_legacy = tuple(tuple(t.detach().clone() for t in layer) for layer in legacy)
    return DynamicCache.from_legacy_cache(cloned_legacy)


# -----------------------------
# Data helpers
# -----------------------------
def build_token_stream_from_dataset(
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_field: str = "text",
    max_chars: Optional[int] = None,
    add_eos_between_docs: bool = True,
    token_cache_path: Optional[str] = None,
) -> List[int]:
    """
    Tokenize a dataset split into one long stream of token IDs, optionally cached to disk.

    max_chars: if set, only tokenize up to this many characters total (for speed).
    """
    if token_cache_path is not None and os.path.exists(token_cache_path):
        arr = np.load(token_cache_path)
        return arr.astype(np.int64).tolist()

    ds = load_dataset(dataset_name, dataset_config, split=split)
    toks: List[int] = []
    eos = tokenizer.eos_token_id

    char_count = 0
    for ex in tqdm(ds, desc=f"Tokenizing {dataset_name}/{dataset_config}:{split}"):
        txt = ex.get(text_field, "")
        if not isinstance(txt, str) or len(txt) == 0:
            continue
        if max_chars is not None and char_count >= max_chars:
            break
        char_count += len(txt)

        ids = tokenizer(txt, add_special_tokens=False).input_ids
        if len(ids) == 0:
            continue
        toks.extend(ids)
        if add_eos_between_docs and eos is not None:
            toks.append(int(eos))

    if token_cache_path is not None:
        os.makedirs(os.path.dirname(token_cache_path) or ".", exist_ok=True)
        np.save(token_cache_path, np.array(toks, dtype=np.int64))

    return toks


def sample_fixed_windows(
    token_stream: List[int],
    window_len: int,
    num_windows: int,
    seed: int,
) -> np.ndarray:
    """
    Sample random contiguous windows of fixed length from a token stream.
    Returns array [N, window_len] of int64.
    """
    rng = np.random.default_rng(seed)
    max_start = len(token_stream) - window_len
    if max_start <= 0:
        raise ValueError(f"Token stream too short ({len(token_stream)}) for window_len={window_len}")

    starts = rng.integers(low=0, high=max_start, size=num_windows, endpoint=False)
    windows = np.stack([np.array(token_stream[s : s + window_len], dtype=np.int64) for s in starts], axis=0)
    return windows


# -----------------------------
# KV pollution measurement core
# -----------------------------
@dataclass
class CorruptSpec:
    axis: str  # "sparsity" | "prune" | "quant"
    level: Union[float, int]  # keep frac for sparsity/prune; bits for quant


@torch.inference_mode()
def run_dense_baseline(
    model,
    prefill_ids: torch.Tensor,  # [B, P]
    decode_ids: torch.Tensor,   # [B, D]
    label_ids: torch.Tensor,    # [B, D]
    Ts: int,
    Tw: int,
    sparsity_mode: str,
    relevancy_tier: str,
) -> torch.Tensor:
    """
    Fully dense teacher-forced decode (still goes through build_sparse_attention_bias with keep=1.0
    so you can keep the interface uniform).
    Returns nll: [D, B]
    """
    B, P = prefill_ids.shape
    _, D = decode_ids.shape
    device = prefill_ids.device
    dtype = next(model.parameters()).dtype

    # reset any leftover knobs
    clear_structured_action(model)
    try:
        clear_relevancy_keep(model)
    except Exception:
        pass
    try:
        clear_quest_token_budgets(model)
    except Exception:
        pass

    out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
    base_cache = out.past_key_values
    past_kv = clone_cache_as_dynamic(base_cache)
    kv_len = torch.full((B,), P + 1, device=device, dtype=torch.long)

    ones = torch.ones((B,), device=device, dtype=torch.float32)
    nll = torch.empty((D, B), device=device, dtype=torch.float32)

    for t in range(D):
        cur = decode_ids[:, t]
        lab = label_ids[:, t]
        pos_ids = (kv_len - 1).clamp_min(0).unsqueeze(1)

        attn_bias = build_sparse_attention_bias(
            model=model,
            past_kv_lens=kv_len,
            keep_fracs=ones,
            Ts=Ts,
            Tw=Tw,
            device=device,
            dtype=dtype,
            criteria=sparsity_mode,
            tier=relevancy_tier,
        )

        out_step = model(
            input_ids=cur.unsqueeze(1),
            past_key_values=past_kv,
            use_cache=True,
            position_ids=pos_ids,
            attention_mask=attn_bias,
            return_dict=True,
        )
        logits = out_step.logits[:, -1, :]
        nll[t] = F.cross_entropy(logits, lab, reduction="none").to(torch.float32)

        past_kv = out_step.past_key_values
        kv_len = kv_len + 1

    clear_structured_action(model)
    return nll


@torch.inference_mode()
def run_corrupt_then_dense_tail(
    model,
    prefill_ids: torch.Tensor,  # [B, P]
    decode_ids: torch.Tensor,   # [B, D]
    label_ids: torch.Tensor,    # [B, D]
    Ts: int,
    Tw: int,
    sparsity_mode: str,
    relevancy_tier: str,
    t_corrupt: int,
    corrupt: CorruptSpec,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher-forced decode where first t_corrupt steps are run with a single approximation axis,
    then everything is set back to dense.

    Returns:
      nll_all: [D, B]   (for direct + tail analysis)
      applied_mask: [D, B] bool, True where corruption was applied (first t_corrupt steps)
    """
    B, P = prefill_ids.shape
    _, D = decode_ids.shape
    device = prefill_ids.device
    dtype = next(model.parameters()).dtype

    # reset knobs
    clear_structured_action(model)
    try:
        clear_relevancy_keep(model)
    except Exception:
        pass
    try:
        clear_quest_token_budgets(model)
    except Exception:
        pass

    out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
    base_cache = out.past_key_values
    past_kv = clone_cache_as_dynamic(base_cache)
    kv_len = torch.full((B,), P + 1, device=device, dtype=torch.long)

    nll = torch.empty((D, B), device=device, dtype=torch.float32)
    applied = torch.zeros((D, B), device=device, dtype=torch.bool)

    # Dense defaults
    kappa_dense = torch.ones((B,), device=device, dtype=torch.float32)
    prune_dense = torch.ones((B,), device=device, dtype=torch.float32)
    qbits_dense = torch.full((B,), 16, device=device, dtype=torch.long)

    for t in range(D):
        cur = decode_ids[:, t]
        lab = label_ids[:, t]
        pos_ids = (kv_len - 1).clamp_min(0).unsqueeze(1)

        # Decide controls for this step
        if t < t_corrupt:
            applied[t] = True
            if corrupt.axis == "sparsity":
                kappa = torch.full((B,), float(corrupt.level), device=device, dtype=torch.float32)
                prune = prune_dense
                qbits = qbits_dense
            elif corrupt.axis == "prune":
                kappa = kappa_dense
                prune = torch.full((B,), float(corrupt.level), device=device, dtype=torch.float32)
                qbits = qbits_dense
            elif corrupt.axis == "quant":
                kappa = kappa_dense
                prune = prune_dense
                qbits = torch.full((B,), int(corrupt.level), device=device, dtype=torch.long)
            else:
                raise ValueError(f"Unknown corrupt.axis={corrupt.axis}")
        else:
            kappa = kappa_dense
            prune = prune_dense
            qbits = qbits_dense

        # Apply structured knobs (prune/quant) even if dense, to keep behavior explicit
        set_structured_action(model, prune_keep=prune, quant_bits=qbits)

        # Apply sparsity knob (Quest/recency/relevancy)
        attn_bias = build_sparse_attention_bias(
            model=model,
            past_kv_lens=kv_len,
            keep_fracs=kappa,
            Ts=Ts,
            Tw=Tw,
            device=device,
            dtype=dtype,
            criteria=sparsity_mode,
            tier=relevancy_tier,
        )

        out_step = model(
            input_ids=cur.unsqueeze(1),
            past_key_values=past_kv,
            use_cache=True,
            position_ids=pos_ids,
            attention_mask=attn_bias,
            return_dict=True,
        )
        logits = out_step.logits[:, -1, :]
        nll[t] = F.cross_entropy(logits, lab, reduction="none").to(torch.float32)

        past_kv = out_step.past_key_values
        kv_len = kv_len + 1

        clear_structured_action(model)

    clear_structured_action(model)
    return nll, applied


# -----------------------------
# Aggregation / plotting
# -----------------------------
def summarize_delta_tail(delta_nll_tail: np.ndarray) -> Dict[str, Union[float, List[float]]]:
    """
    delta_nll_tail: [N, T_tail] (nats)
    Returns summary stats in %Δppl space.
    """
    # per-token percent increase
    delta_pct = (np.exp(delta_nll_tail) - 1.0) * 100.0  # [N, T]

    # per-window mean tail increase
    mean_tail_pct_per_win = (np.exp(delta_nll_tail.mean(axis=1)) - 1.0) * 100.0  # [N]
    # per-window peak tail increase (max over tail positions)
    peak_tail_pct_per_win = delta_pct.max(axis=1)  # [N]

    # worst-10% mean tail (by mean_tail_pct)
    N = mean_tail_pct_per_win.shape[0]
    k = max(1, int(math.ceil(0.10 * N)))
    worst_idx = np.argsort(mean_tail_pct_per_win)[-k:]
    worst10_mean_tail = float(mean_tail_pct_per_win[worst_idx].mean())

    return {
        "mean_tail_pct": float(mean_tail_pct_per_win.mean()),
        "worst10_mean_tail_pct": worst10_mean_tail,
        "peak_tail_pct_p90": float(np.quantile(peak_tail_pct_per_win, 0.90)),
        "peak_tail_pct_p95": float(np.quantile(peak_tail_pct_per_win, 0.95)),
        "curve_median_pct": np.median(delta_pct, axis=0).tolist(),
        "curve_p90_pct": np.quantile(delta_pct, 0.90, axis=0).tolist(),
        "curve_p10_pct": np.quantile(delta_pct, 0.10, axis=0).tolist(),
    }


def plot_impulse_response_3col(
    stats_by_axis: Dict[str, Dict[str, Dict]],
    tail_len: int,
    out_path_png: str,
    out_path_pdf: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    stats_by_axis[axis][level_str] = summarize_delta_tail output dict, containing:
      curve_median_pct, curve_p10_pct, curve_p90_pct, plus summary scalars.

    Creates one 1x3 figure with shared Y.
    """
    axes_order = ["sparsity", "prune", "quant"]
    axis_titles = {
        "sparsity": "Token sparsity (KV pollution)",
        "prune": "MLP pruning (KV pollution)",
        "quant": "MLP quantization (KV pollution)",
    }

    x = np.arange(1, tail_len + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    if title is not None:
        fig.suptitle(title)

    for j, axis in enumerate(axes_order):
        ax = axs[j]
        level_dict = stats_by_axis.get(axis, {})

        # Sort levels for nicer legends:
        # - sparsity/prune: smaller keep => more aggressive
        # - quant: smaller bits => more aggressive
        def _sort_key(level_str):
            try:
                v = float(level_str)
            except Exception:
                v = 0.0
            return v

        levels_sorted = sorted(level_dict.keys(), key=_sort_key)

        for level_str in levels_sorted:
            st = level_dict[level_str]
            med = np.array(st["curve_median_pct"], dtype=np.float32)
            p10 = np.array(st["curve_p10_pct"], dtype=np.float32)
            p90 = np.array(st["curve_p90_pct"], dtype=np.float32)

            # plot median and a p10-p90 band
            ax.plot(x, med, label=level_str)
            ax.fill_between(x, p10, p90, alpha=0.15)

        ax.set_title(axis_titles[axis])
        ax.set_xlabel("Tokens after corruption (tail offset)")
        ax.grid(True, alpha=0.3)

        if j == 0:
            ax.set_ylabel("%Δ perplexity vs dense (tail-only)")

        ax.legend(title="Level", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=200)
    if out_path_pdf is not None:
        fig.savefig(out_path_pdf)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--token_cache_path", type=str, default=None)

    p.add_argument("--prefill_len", type=int, default=512)
    p.add_argument("--t_corrupt", type=int, default=4)
    p.add_argument("--tail_len", type=int, default=64)

    p.add_argument("--num_windows", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)

    # Attention sparsity backend
    p.add_argument("--sparsity_mode", type=str, default="quest", choices=["quest", "recency", "relevancy"])
    p.add_argument("--quest_page_size", type=int, default=8)
    p.add_argument("--Ts", type=int, default=4)
    p.add_argument("--Tw", type=int, default=2)
    p.add_argument("--relevancy_tier", type=str, default="per_head", choices=["per_head", "per_layer"])

    # Level specification:
    # Option A: ratios shared across axes (token keep, prune keep, qbits/16 mapped to bits)
    p.add_argument("--ratios", type=float, nargs="*", default=None)

    # Option B: explicit lists
    p.add_argument("--kappa_list", type=float, nargs="*", default=None)
    p.add_argument("--prune_list", type=float, nargs="*", default=None)
    p.add_argument("--qbits_list", type=int, nargs="*", default=None)

    p.add_argument("--out_dir", type=str, default="./kv_pollution_out")
    p.add_argument("--run_name", type=str, default="kv_pollution_impulse")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device / dtype
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # Load model/tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Quest patch needs eager attention to be active in transformers.
    # If your load_lm_and_tokenizer() already does this, you can remove this.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        attn_implementation="eager" if args.sparsity_mode == "quest" else None,
    ).to(device)
    model.eval()

    # Enable patches
    enable_structured_controls(model)
    if args.sparsity_mode == "quest":
        enable_quest_attention(model, page_size=args.quest_page_size)

    # Build token stream + windows
    window_len = args.prefill_len + (args.t_corrupt + args.tail_len) + 1
    token_stream = build_token_stream_from_dataset(
        tokenizer=tok,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        token_cache_path=args.token_cache_path,
    )
    windows = sample_fixed_windows(token_stream, window_len, args.num_windows, seed=args.seed)
    windows_t = torch.from_numpy(windows)  # [N, L]
    dl = DataLoader(TensorDataset(windows_t), batch_size=args.batch_size, shuffle=False)

    # Levels to test
    if args.ratios is not None and len(args.ratios) > 0:
        ratios = [float(r) for r in args.ratios]
        kappa_levels = ratios
        prune_levels = ratios
        # map ratio to bits in [4,16]
        qbits_levels = [int(np.clip(int(round(r * 16.0)), 4, 16)) for r in ratios]
    else:
        kappa_levels = args.kappa_list or [0.3, 0.5, 0.7]
        prune_levels = args.prune_list or [0.5, 0.7, 0.9]
        qbits_levels = args.qbits_list or [6, 8, 12]

    # Storage: raw per-example tail ΔNLL arrays
    raw_delta_tail: Dict[str, Dict[str, List[np.ndarray]]] = {
        "sparsity": {str(k): [] for k in kappa_levels},
        "prune": {str(p): [] for p in prune_levels},
        "quant": {str(q): [] for q in qbits_levels},
    }
    # Also store direct-region deltas (optional summary)
    raw_delta_direct: Dict[str, Dict[str, List[np.ndarray]]] = {
        "sparsity": {str(k): [] for k in kappa_levels},
        "prune": {str(p): [] for p in prune_levels},
        "quant": {str(q): [] for q in qbits_levels},
    }

    decode_len = args.t_corrupt + args.tail_len

    # Evaluate
    for (batch_windows,) in tqdm(dl, desc="KV pollution eval batches"):
        batch_windows = batch_windows.to(device)
        B = batch_windows.size(0)

        prefill = batch_windows[:, : args.prefill_len]
        decode_ids = batch_windows[:, args.prefill_len : args.prefill_len + decode_len]
        labels = batch_windows[:, args.prefill_len + 1 : args.prefill_len + decode_len + 1]

        # Dense baseline once per batch
        nll_dense = run_dense_baseline(
            model=model,
            prefill_ids=prefill,
            decode_ids=decode_ids,
            label_ids=labels,
            Ts=args.Ts,
            Tw=args.Tw,
            sparsity_mode=args.sparsity_mode,
            relevancy_tier=args.relevancy_tier,
        )  # [D,B]

        dense_direct = nll_dense[: args.t_corrupt]  # [C,B]
        dense_tail = nll_dense[args.t_corrupt :]    # [T,B]

        # --- Token sparsity corruption ---
        for kappa in kappa_levels:
            nll_corrupt, _ = run_corrupt_then_dense_tail(
                model=model,
                prefill_ids=prefill,
                decode_ids=decode_ids,
                label_ids=labels,
                Ts=args.Ts,
                Tw=args.Tw,
                sparsity_mode=args.sparsity_mode,
                relevancy_tier=args.relevancy_tier,
                t_corrupt=args.t_corrupt,
                corrupt=CorruptSpec(axis="sparsity", level=float(kappa)),
            )
            delta_direct = (nll_corrupt[: args.t_corrupt] - dense_direct).transpose(0, 1).contiguous()  # [B,C]
            delta_tail = (nll_corrupt[args.t_corrupt :] - dense_tail).transpose(0, 1).contiguous()      # [B,T]
            raw_delta_direct["sparsity"][str(kappa)].append(delta_direct.detach().cpu().numpy())
            raw_delta_tail["sparsity"][str(kappa)].append(delta_tail.detach().cpu().numpy())

        # --- Pruning corruption ---
        for pr in prune_levels:
            nll_corrupt, _ = run_corrupt_then_dense_tail(
                model=model,
                prefill_ids=prefill,
                decode_ids=decode_ids,
                label_ids=labels,
                Ts=args.Ts,
                Tw=args.Tw,
                sparsity_mode=args.sparsity_mode,
                relevancy_tier=args.relevancy_tier,
                t_corrupt=args.t_corrupt,
                corrupt=CorruptSpec(axis="prune", level=float(pr)),
            )
            delta_direct = (nll_corrupt[: args.t_corrupt] - dense_direct).transpose(0, 1).contiguous()
            delta_tail = (nll_corrupt[args.t_corrupt :] - dense_tail).transpose(0, 1).contiguous()
            raw_delta_direct["prune"][str(pr)].append(delta_direct.detach().cpu().numpy())
            raw_delta_tail["prune"][str(pr)].append(delta_tail.detach().cpu().numpy())

        # --- Quantization corruption ---
        for qb in qbits_levels:
            nll_corrupt, _ = run_corrupt_then_dense_tail(
                model=model,
                prefill_ids=prefill,
                decode_ids=decode_ids,
                label_ids=labels,
                Ts=args.Ts,
                Tw=args.Tw,
                sparsity_mode=args.sparsity_mode,
                relevancy_tier=args.relevancy_tier,
                t_corrupt=args.t_corrupt,
                corrupt=CorruptSpec(axis="quant", level=int(qb)),
            )
            delta_direct = (nll_corrupt[: args.t_corrupt] - dense_direct).transpose(0, 1).contiguous()
            delta_tail = (nll_corrupt[args.t_corrupt :] - dense_tail).transpose(0, 1).contiguous()
            raw_delta_direct["quant"][str(qb)].append(delta_direct.detach().cpu().numpy())
            raw_delta_tail["quant"][str(qb)].append(delta_tail.detach().cpu().numpy())

    # Summarize + plot
    stats_by_axis: Dict[str, Dict[str, Dict]] = {"sparsity": {}, "prune": {}, "quant": {}}
    summary_table: Dict[str, Dict[str, Dict]] = {"sparsity": {}, "prune": {}, "quant": {}}

    for axis in ["sparsity", "prune", "quant"]:
        for level_str, chunks in raw_delta_tail[axis].items():
            if len(chunks) == 0:
                continue
            # [num_examples, tail_len]
            delta_tail = np.concatenate(chunks, axis=0)
            st = summarize_delta_tail(delta_tail)
            stats_by_axis[axis][level_str] = st

            # Also compute direct-region average (to support the “quant worse but less pollution” narrative)
            direct_chunks = raw_delta_direct[axis][level_str]
            delta_direct = np.concatenate(direct_chunks, axis=0)  # [N, C]
            direct_mean_pct = float((np.exp(delta_direct.mean(axis=1)) - 1.0).mean() * 100.0)

            summary_table[axis][level_str] = {
                "direct_mean_pct": direct_mean_pct,
                "tail_mean_pct": st["mean_tail_pct"],
                "tail_worst10_mean_pct": st["worst10_mean_tail_pct"],
                "tail_peak_p90_pct": st["peak_tail_pct_p90"],
                "tail_peak_p95_pct": st["peak_tail_pct_p95"],
            }

    # Save JSON
    json_path = os.path.join(args.out_dir, f"{args.run_name}_summary.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "summary_table": summary_table,
            },
            f,
            indent=2,
        )

    # Print quick table
    print("\n=== KV Pollution Summary (percent Δppl) ===")
    for axis in ["sparsity", "prune", "quant"]:
        print(f"\n[{axis}]")
        for level_str, row in summary_table[axis].items():
            print(
                f"  level={level_str:>6} | direct_mean={row['direct_mean_pct']:+6.2f}% "
                f"| tail_mean={row['tail_mean_pct']:+6.2f}% "
                f"| tail_worst10_mean={row['tail_worst10_mean_pct']:+6.2f}% "
                f"| peak_p90={row['tail_peak_p90_pct']:+7.2f}% "
                f"| peak_p95={row['tail_peak_p95_pct']:+7.2f}%"
            )

    # Plot
    out_png = os.path.join(args.out_dir, f"{args.run_name}_impulse_3col.png")
    out_pdf = os.path.join(args.out_dir, f"{args.run_name}_impulse_3col.pdf")
    title = f"KV pollution impulse response (prefill={args.prefill_len}, corrupt={args.t_corrupt}, tail={args.tail_len}, mode={args.sparsity_mode})"
    plot_impulse_response_3col(
        stats_by_axis=stats_by_axis,
        tail_len=args.tail_len,
        out_path_png=out_png,
        out_path_pdf=out_pdf,
        title=title,
    )

    print(f"\nSaved summary JSON: {json_path}")
    print(f"Saved plot PNG:     {out_png}")
    print(f"Saved plot PDF:     {out_pdf}")


if __name__ == "__main__":
    main()



