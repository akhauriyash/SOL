import os
import json
from pprint import pprint
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import argparse
import torch

from utils.seeds import set_seed
from utils.model import load_lm_and_tokenizer, unwrap
from utils.data import make_dataloader, limited_dl
from utils.eval import (
    evaluate_sft_teacher_matched_keep,
    evaluate_stateful_policy_rollout
)
from utils.eval_baselines import (
    evaluate_fixed_matched_keep,
    evaluate_randomized_matched_sparsity,
    evaluate_dense_full,
    evaluate_drift_aware_matched_keep,
    evaluate_emc_matched_keep
)
from utils.config import Config
from predictor import RecurrentActorCriticPolicy
from utils.actions import build_action_spec
import torch
import math
import torch.backends.cuda as sdp
sdp.enable_flash_sdp(False)
sdp.enable_math_sdp(False)
sdp.enable_mem_efficient_sdp(True)  # SDPA only

@dataclass
class EvalCfg:
    CKPT_DIR: str = "/home/ya255/rl4e/checkpoints/GRPO_DKL_Relv-20251017-002523/"
    eval_batches: Optional[int] = None   # None = run full loader
    split: str = "validation"           # typically "validation"
    mode: str = "latest" # "latest" or "best" (if best.pt exists)
    # dataset_name: Optional[str] = "wikitext"
    # dataset_config: Optional[str] = "wikitext-2-raw-v1"
    dataset_name: Optional[str] = "allenai/c4"
    dataset_config: Optional[str] = "en"
    text_field: Optional[str] = "text"
    batch_size: Optional[int] = 16
    seed: int = 1234
    greedy: bool = True
    policy_temperature: float = 0.6
    sparsity_bias: float = 0.0
    quant_bias: float = 0.0
    prune_bias: float = 0.0


def find_latest_ckpt(ckpt_dir: str, mode: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    latest = os.path.join(ckpt_dir, f"policy_{mode}.pt")
    if os.path.exists(latest):
        return latest
    cands = [f for f in os.listdir(ckpt_dir) if f.startswith("policy_epoch") and f.endswith(".pt")]
    if not cands:
        return None
    cands.sort()
    return os.path.join(ckpt_dir, cands[-1])


def load_cfg_from_checkpoint_or_yaml(
    ckpt_dir: str,
    ckpt_path: str,
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    text_field: Optional[str],
    batch_size: Optional[int],
) -> Config:
    """
    Load the *training* config for evaluation.
    Priority:
      1) 'cfg' dict embedded in the checkpoint
      2) YAML recorded in 'meta.config_paths.base' -> ckpt_dir/code/<relpath> (or train_meta.json)
      3) Default Config()
    Then apply only dataset-related overrides.
    """
    cfg = Config()
    sd_cpu = torch.load(ckpt_path, map_location="cpu")
    sd_cfg = sd_cpu.get("cfg")
    if sd_cfg:
        print("[eval] Using training config embedded in checkpoint.")
        for k, v in sd_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        meta = sd_cpu.get("meta", None)
        if meta is None:
            meta_path = os.path.join(ckpt_dir, "train_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
        if meta is None:
            print("[eval] No 'meta' found in checkpoint or train_meta.json.")
        base_rel = None
        if meta is not None:
            base_rel = meta.get("config_paths", {}).get("base")
            # For visibility, print what we found
            kind = meta.get("kind", "unknown")
            print(f"[eval] meta.kind={kind}")
            print(f"[eval] meta.config_paths={meta.get('config_paths', {})}")
        if base_rel:
            yaml_path = os.path.join(ckpt_dir, "code", base_rel)
            if os.path.exists(yaml_path):
                from utils.config import apply_cfg_overrides_from_file
                apply_cfg_overrides_from_file(cfg, yaml_path, is_main=True)
                print(f"[eval] Applied base YAML overrides from snapshot: {yaml_path}")
            else:
                print(f"[eval] Saved base config not found at {yaml_path}; using defaults.")
        else:
            print("[eval] No training config in checkpoint or meta; using defaults.")

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dataset_name is not None:
        cfg.dataset_name = dataset_name
    if dataset_config is not None:
        cfg.dataset_config = dataset_config
    if text_field is not None:
        cfg.text_field = text_field
    if batch_size is not None:
        cfg.batch_size = batch_size
    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default=None)
parser.add_argument("--mode", type=str, default=None)
parser.add_argument("--tgt_keep", type=float, default=None)
parser.add_argument("--criteria", type=str, default=None)
parser.add_argument("--outp", type=str, default="")
parser.add_argument("--dataset_name", type=str, choices=["wikitext", "allenai/c4"], default=None)
parser.add_argument("--sparsity_bias", type=float, default=None,
                    help="Positive values bias the policy toward sparser actions during eval.")
parser.add_argument("--quant_bias", type=float, default=None,
                    help="Positive values bias the policy toward more quantization during eval.")
parser.add_argument("--prune_bias", type=float, default=None,
                    help="Positive values bias the policy toward more pruning during eval.")
args = parser.parse_args()

E = EvalCfg()
if args.ckpt_dir is not None:
    E.CKPT_DIR = args.ckpt_dir
if args.mode is not None:
    E.mode = args.mode
if args.dataset_name is not None:
    E.dataset_name = args.dataset_name
    E.dataset_config = "wikitext-2-raw-v1" if args.dataset_name == "wikitext" else "en"
if args.sparsity_bias is not None:
    E.sparsity_bias = args.sparsity_bias
if args.quant_bias is not None:
    E.quant_bias = args.quant_bias
if args.prune_bias is not None:
    E.prune_bias = args.prune_bias

set_seed(E.seed)
ckpt_path = find_latest_ckpt(E.CKPT_DIR, E.mode)
if ckpt_path is None:
    raise FileNotFoundError(f"No checkpoint found in {E.CKPT_DIR}")
cfg = load_cfg_from_checkpoint_or_yaml(
    ckpt_dir=E.CKPT_DIR,
    ckpt_path=ckpt_path,
    dataset_name=E.dataset_name,
    dataset_config=E.dataset_config,
    text_field=E.text_field,
    batch_size=E.batch_size,
)


cfg.eval_sparsity_bias = float(getattr(E, "sparsity_bias", getattr(cfg, "eval_sparsity_bias", 0.0)))
cfg.eval_quant_bias = float(getattr(E, "quant_bias", getattr(cfg, "eval_quant_bias", 0.0)))
cfg.eval_prune_bias = float(getattr(E, "prune_bias", getattr(cfg, "eval_prune_bias", 0.0)))

if args.criteria is not None:
    cfg.sparsity_criteria = args.criteria

tok, model = load_lm_and_tokenizer(cfg)

dl = make_dataloader(cfg, tok, split=E.split, shuffle=False, distributed=False)

base_model = unwrap(model)
hidden_size = getattr(base_model.config, "hidden_size", getattr(base_model.config, "n_embd", None))
if hidden_size is None:
    raise ValueError("Could not infer hidden size from model.config")

sd = torch.load(ckpt_path, map_location=cfg.device)
meta = sd.get("meta")
if meta:
    kind = meta.get("kind", "unknown")
    cmd = meta.get("command", "")
    paths = meta.get("config_paths", {})
    print(f"[ckpt] kind: {kind}")
    print(f"[ckpt] command: {cmd}")
    print(f"[ckpt] config_paths: base={paths.get('base')}, rl={paths.get('rl')}")
else:
    print("[ckpt] No meta found in checkpoint.")

 
sd_cfg = sd.get("cfg", None)
if sd_cfg is not None and "keep_fracs" in sd_cfg:
    cfg.keep_fracs = tuple(sd_cfg["keep_fracs"])

emb_layer = unwrap(model).get_input_embeddings()
embed_dim = getattr(emb_layer, "embedding_dim", emb_layer.weight.shape[1])
in_dim = int(hidden_size + embed_dim + 1)
pol_d_model  = int(getattr(cfg, "policy_d_model", 768))
pol_heads    = int(getattr(cfg, "policy_n_heads", 8))
pol_layers   = int(getattr(cfg, "policy_n_layers", 2))
pol_mlp_mult = float(getattr(cfg, "policy_mlp_ratio", 4.0))
pol_act_dim  = int(getattr(cfg, "policy_action_dim", 32))
pol_max_len  = int(getattr(cfg, "policy_max_len", max(1024, cfg.rollout_len + 8)))
lam = float(sd.get("global_step_state", {}).get("lambda_keep", 0.0))
lamprune = float(sd.get("global_step_state", {}).get("lambda_prune", 0.0))
lamquant = float(sd.get("global_step_state", {}).get("lambda_quant", 0.0))
SCALAR_D = int(getattr(cfg, "policy_scalar_dim", 8))

spec = build_action_spec(
    keep_fracs=cfg.keep_fracs,
    prune_choices=cfg.struct_prune_choices,
    quant_choices=cfg.quant_choices,
)
policy = RecurrentActorCriticPolicy(
    h_dim=int(hidden_size),
    e_dim=int(embed_dim),
    n_actions=spec.n_actions,
    d_model=pol_d_model,
    n_heads=pol_heads,
    n_layers=pol_layers,
    mlp_ratio=pol_mlp_mult,
    action_dim=pol_act_dim,
    max_len=pol_max_len,
    scalar_dim=SCALAR_D,
).to(cfg.device, dtype=torch.float32)
n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"Policy parameters: {n_params:,}")
state_key = "policy_state_dict" if "policy_state_dict" in sd else "state_dict"
policy.load_state_dict(sd[state_key], strict=True)
policy.eval()

print(f"\nLoaded checkpoint: {ckpt_path}")
print(f"Device={cfg.device}, dtype={cfg.dtype}, model={cfg.model_name}")
print(f"Eval split='{E.split}', batches={E.eval_batches}, batch_size={cfg.batch_size}")
print(f"Structure: Ts={cfg.Ts}  Tw={cfg.Tw}  keep_fracs={cfg.keep_fracs}")
print(f"context_len={cfg.context_len} rollout_len={cfg.rollout_len}")
print(f"RL policy eval settings: greedy={E.greedy}, temperature={E.policy_temperature}")

assert cfg.sparsity_criteria == "quest", "This eval script only supports 'quest' criteria. temporarily."

start_time = time.time()

greedy = evaluate_stateful_policy_rollout(
    cfg,
    model,
    policy,
    limited_dl(dl, E.eval_batches),
    Ts=cfg.Ts, Tw=cfg.Tw, keep_fracs=cfg.keep_fracs,
    context_len=cfg.context_len, rollout_len=cfg.rollout_len,
    device=cfg.device,
    greedy=E.greedy, temperature=E.policy_temperature,
    lambda_keep=lam,
    lambda_prune=lamprune,
    lambda_quant=lamquant,
    sparsity_bias=cfg.eval_sparsity_bias,
    prune_bias=cfg.eval_prune_bias,
    quant_bias=cfg.eval_quant_bias,
)
total_time = time.time() - start_time
print(
    f"\nPolicy (greedy={'yes' if E.greedy else 'no'})\t: ppl={greedy['ppl']:.3f}  "
    f"keep_all={greedy['avg_keep_all']:.3f}  "
    f"prune_keep={greedy['avg_prune_keep']:.3f}  "
    f"quant_ratio={16*greedy['avg_quant_ratio']:.3f}\t"
    f"tokens={greedy['tokens_effective']}/{greedy['tokens']}\n"
)
print(f"Policy rollout evaluation complete in {total_time:.2f} seconds.")
print(f"Policy Action Probs \t: {[f'{p:.3f}' for p in greedy['action_probs']]}")

if args.tgt_keep is not None:
    greedy["avg_keep_effective"] = args.tgt_keep



target_prune = greedy["avg_prune_keep"]
target_qratio = greedy["avg_quant_ratio"]

start_time = time.time()
fixed_matched = evaluate_fixed_matched_keep(
    cfg,
    model,
    limited_dl(make_dataloader(cfg, tok, split=E.split, shuffle=False, distributed=False), E.eval_batches),
    Ts=cfg.Ts,
    Tw=cfg.Tw,
    keep_fracs=cfg.keep_fracs,
    prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
    quant_choices=getattr(cfg, "quant_choices", ("q16",)),
    target_keep_effective=greedy["avg_keep_effective"],
    target_prune_keep=target_prune,
    target_quant_ratio=target_qratio,
    context_len=cfg.context_len,
    rollout_len=cfg.rollout_len,
    device=cfg.device,
    struct_on_non_eff=False,
)
total_time = time.time() - start_time
print(
    f"\nFixed (matched keep)  \t\t: ppl={fixed_matched['ppl']:.3f}  "
    f"keep_all={fixed_matched['avg_keep_all']:.3f}  "
    f"prune_keep={fixed_matched['avg_prune_keep']:.3f}  "
    f"quant_ratio={16*fixed_matched['avg_quant_ratio']:.3f}\t"
    f"tokens={fixed_matched['tokens_effective']}/{fixed_matched['tokens']}\n"
)
print(f"Fixed matched evaluation complete in {total_time:.2f} seconds.")
print(f"Fixed Action Probs \t: {[f'{p:.3f}' for p in fixed_matched['action_probs']]}")

# ---- Pretty print summary ----
print("\n\n=== Results (validation) ===\n")
print(
    f"Policy (greedy={'yes' if E.greedy else 'no'})\t\t: ppl={greedy['ppl']:.3f}  "
    f"keep_all={greedy['avg_keep_all']:.3f}  "
    f"keep_eff={greedy['avg_keep_effective']:.3f}\t"
    f"tokens={greedy['tokens_effective']}/{greedy['tokens']}"
)
print(
    f"Fixed (matched keep)  \t\t: ppl={fixed_matched['ppl']:.3f}  "
    f"keep_all={fixed_matched['avg_keep_all']:.3f}  "
    f"keep_eff={fixed_matched['avg_keep_effective']:.3f}\t"
    f"tokens={fixed_matched['tokens_effective']}/{fixed_matched['tokens']}"
)

if 'action_probs' in greedy:
    probs = ", ".join(f"{p:.2f}" for p in greedy['action_probs'])
    print(f"Sparsity levels (κ order)    : [{', '.join(f'{k:.2f}' for k in cfg.keep_fracs)}]")
    print(f"Policy action probs (κ order) : [{probs}]")

if 'action_probs' in fixed_matched:
    probs = ", ".join(f"{p:.2f}" for p in fixed_matched['action_probs'])
    print(f"Fixed action probs (κ order)  : [{probs}]")


import csv

csv_path = "multi_eff_ppl_scan.csv"
if args.outp != "":
    csv_path = args.outp + "_multi_eff_ppl_scan.csv"

ckpt_dir_last = os.path.basename(os.path.normpath(E.CKPT_DIR))

sparsity_levels = "|".join(f"{k:.6f}" for k in cfg.keep_fracs)
policy_action_probs = "|".join(f"{p:.6f}" for p in greedy.get("action_probs", []))
fixed_levels_kappa_order = "|".join(f"{k:.6f}" for k in fixed_matched.get("action_probs", []))
fixed_action_probs = "|".join(f"{p:.6f}" for p in fixed_matched.get("action_probs", []))

row = {
    "ckpt_dir": ckpt_dir_last,
    "mode": E.mode,
    "dataset_name": E.dataset_name,
    "sparsity_bias": E.sparsity_bias,
    "prune_bias": E.prune_bias,
    "quant_bias": E.quant_bias,

    "policy_ppl": float(greedy["ppl"]),
    "fixed_ppl": float(fixed_matched["ppl"]),

    "policy_keep_all": float(greedy["avg_keep_all"]),
    "policy_prune_keep": float(greedy["avg_prune_keep"]),
    "policy_quant_ratio": float(greedy["avg_quant_ratio"]),
    "fixed_keep_all": float(fixed_matched["avg_keep_all"]),
    "fixed_prune_keep": float(fixed_matched["avg_prune_keep"]),
    "fixed_quant_ratio": float(fixed_matched["avg_quant_ratio"]),

    "sparsity_levels_kappa_order": sparsity_levels,
    "policy_action_probs_kappa_order": policy_action_probs,

    "fixed_levels_kappa_order": fixed_levels_kappa_order,
    "fixed_action_probs_kappa_order": fixed_action_probs,
}

fieldnames = list(row.keys())

file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"[csv] Appended results to {csv_path}")
