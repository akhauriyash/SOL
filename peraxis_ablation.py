import os
import csv
import time
import json
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.backends.cuda as sdp

from utils.seeds import set_seed
from utils.model import load_lm_and_tokenizer, unwrap
from utils.data import make_dataloader, limited_dl
from utils.eval import (
    evaluate_stateful_policy_rollout,
)
from utils.eval_baselines import (
    evaluate_dense_full,
    evaluate_fixed_matched_keep,
    evaluate_randomized_matched_sparsity,
    evaluate_drift_aware_matched_keep,
    evaluate_emc_matched_keep,
    evaluate_lrm_tokens_matched_keep,
    evaluate_qnr_quant_matched_keep,
    evaluate_dynr_quant_matched_keep,
    evaluate_ecov_prune_matched_keep,
    evaluate_dcp_prune_matched_keep,
    evaluate_margin_prune_matched_keep,
)
from utilities import find_latest_ckpt, load_cfg_from_checkpoint_or_yaml
from utils.config import Config
from utils.actions import build_action_spec
from predictor import RecurrentActorCriticPolicy

# SDPA toggles
sdp.enable_flash_sdp(False)
sdp.enable_math_sdp(False)
sdp.enable_mem_efficient_sdp(True)


@dataclass
class EvalCfg:
    CKPT_DIR: str = ""
    eval_batches: Optional[int] = None
    split: str = "validation"
    mode: str = "latest"   # or "best"
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
    study_type: str = "sparsity"  # "sparsity" | "pruning" | "quantization"

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--mode", type=str, default="latest")
parser.add_argument("--dataset_name", type=str, choices=["wikitext", "allenai/c4"], default=None)
parser.add_argument("--sparsity_bias", type=float, default=None,
                    help="Positive values bias the policy toward sparser actions during eval.")
parser.add_argument("--quant_bias", type=float, default=None,
                    help="Positive values bias the policy toward more quantization during eval.")
parser.add_argument("--prune_bias", type=float, default=None,
                    help="Positive values bias the policy toward more pruning during eval.")
parser.add_argument("--eval_batches", type=int, default=None)
parser.add_argument("--study_type", type=str, required=True,
                    choices=["sparsity", "pruning", "quantization"])
args = parser.parse_args()

E = EvalCfg()
E.CKPT_DIR = args.ckpt_dir
E.mode = args.mode
E.study_type = args.study_type
if args.dataset_name is not None:
    E.dataset_name = args.dataset_name
    E.dataset_config = "wikitext-2-raw-v1" if args.dataset_name == "wikitext" else "en"
if args.sparsity_bias is not None:
    E.sparsity_bias = args.sparsity_bias
if args.quant_bias is not None:
    E.quant_bias = args.quant_bias
if args.prune_bias is not None:
    E.prune_bias = args.prune_bias
if args.eval_batches is not None:
    E.eval_batches = args.eval_batches

# ----------------- Setup -----------------

set_seed(E.seed)
ckpt_path = find_latest_ckpt(E.CKPT_DIR, E.mode)
if ckpt_path is None:
    raise FileNotFoundError(f"No checkpoint found in {E.CKPT_DIR}")

cfg: Config = load_cfg_from_checkpoint_or_yaml(
    ckpt_dir=E.CKPT_DIR,
    ckpt_path=ckpt_path,
    dataset_name=E.dataset_name,
    dataset_config=E.dataset_config,
)

# runtime steering biases (used by policy evaluator)
cfg.eval_sparsity_bias = float(getattr(E, "sparsity_bias", getattr(cfg, "eval_sparsity_bias", 0.0)))
cfg.eval_quant_bias = float(getattr(E, "quant_bias", getattr(cfg, "eval_quant_bias", 0.0)))
cfg.eval_prune_bias = float(getattr(E, "prune_bias", getattr(cfg, "eval_prune_bias", 0.0)))

tok, model = load_lm_and_tokenizer(cfg)
dl_full = make_dataloader(cfg, tok, split=E.split, shuffle=False, distributed=False)
dl = limited_dl(dl_full, E.eval_batches)

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

# Build policy
emb_layer = unwrap(model).get_input_embeddings()
embed_dim = getattr(emb_layer, "embedding_dim", emb_layer.weight.shape[1])

pol_d_model  = int(getattr(cfg, "policy_d_model", 768))
pol_heads    = int(getattr(cfg, "policy_n_heads", 8))
pol_layers   = int(getattr(cfg, "policy_n_layers", 2))
pol_mlp_mult = float(getattr(cfg, "policy_mlp_ratio", 4.0))
pol_act_dim  = int(getattr(cfg, "policy_action_dim", 32))
pol_max_len  = int(getattr(cfg, "policy_max_len", max(1024, cfg.rollout_len + 8)))
SCALAR_D     = int(getattr(cfg, "policy_scalar_dim", 8))

spec = build_action_spec(
    keep_fracs=cfg.keep_fracs,
    prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
    quant_choices=getattr(cfg, "quant_choices", ("q16",)),
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
state_key = "policy_state_dict" if "policy_state_dict" in sd else "state_dict"
policy.load_state_dict(sd[state_key], strict=True)
policy.eval()

lam      = float(sd.get("global_step_state", {}).get("lambda_keep", 0.0))
lamprune = float(sd.get("global_step_state", {}).get("lambda_prune", 0.0))
lamquant = float(sd.get("global_step_state", {}).get("lambda_quant", 0.0))

print(f"\nLoaded checkpoint: {ckpt_path}")
print(f"Device={cfg.device}, dtype={cfg.dtype}, model={cfg.model_name}")
print(f"Eval split='{E.split}', batches={E.eval_batches}, batch_size={cfg.batch_size}")
print(f"Structure: Ts={cfg.Ts}  Tw={cfg.Tw}  keep_fracs={cfg.keep_fracs}")
print(f"context_len={cfg.context_len} rollout_len={cfg.rollout_len}")
print(f"Study type: {E.study_type}")
print(f"Policy eval: greedy={E.greedy}, temperature={E.policy_temperature}, "
      f"sparsity_bias={cfg.eval_sparsity_bias}, prune_bias={cfg.eval_prune_bias}, quant_bias={cfg.eval_quant_bias}")

# ----------------- Dense baseline -----------------

t0 = time.time()
dense = evaluate_dense_full(
    model, limited_dl(make_dataloader(cfg, tok, split=E.split, shuffle=False, distributed=False), E.eval_batches),
    cfg.context_len, cfg.rollout_len, cfg.device
)
print(f"Dense (full teacher)\t\t: ppl={dense['ppl']:.3f}  tokens={dense['tokens']}")
print(f"Dense eval done in {time.time()-t0:.2f}s.")

# ----------------- Policy rollout -----------------

t0 = time.time()
policy_out = evaluate_stateful_policy_rollout(
    cfg,
    model,
    policy,
    limited_dl(dl_full, E.eval_batches),
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
print(
    f"\nPolicy\t\t\t\t: ppl={policy_out['ppl']:.3f}  "
    f"keep_all={policy_out.get('avg_keep_all', 1.0):.3f}  "
    f"prune_keep={policy_out.get('avg_prune_keep', 1.0):.3f}  "
    f"quant_ratio={16*policy_out.get('avg_quant_ratio', 1.0):.3f}  "
    f"tokens={policy_out['tokens_effective']}/{policy_out['tokens']}"
)
print(f"Policy eval done in {time.time()-t0:.2f}s.")
if 'action_probs' in policy_out:
    print(f"Policy Action Probs \t: {[f'{p:.3f}' for p in policy_out['action_probs']]}")

# Targets for matched baselines (take from policy outputs)
tgt_keep   = policy_out.get("avg_keep_effective", policy_out.get("avg_keep_all", 1.0))
tgt_prune  = policy_out.get("avg_prune_keep", 1.0)
tgt_qratio = policy_out.get("avg_quant_ratio", 1.0)

# ----------------- Study-specific evaluations -----------------

results = {
    "dense": dense,
    "policy": policy_out,
}

def _dl():
    return limited_dl(make_dataloader(cfg, tok, split=E.split, shuffle=False, distributed=False), E.eval_batches)

def _print_res(name, r):
    print(
        f"\n{name:<26}: ppl={r.get('ppl', float('nan')):.3f}  "
        f"keep_all={r.get('avg_keep_all', 1.0):.3f}  "
        f"prune_keep={r.get('avg_prune_keep', 1.0):.3f}  "
        f"quant_ratio={16*r.get('avg_quant_ratio', 1.0):.3f}  "
        f"tokens={r.get('tokens_effective', 0)}/{r.get('tokens', 0)}"
    )

if E.study_type == "sparsity":
    # Fixed
    t0 = time.time()
    fixed = evaluate_fixed_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_keep_effective=tgt_keep,
        target_prune_keep=tgt_prune,
        target_quant_ratio=tgt_qratio,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
    )
    _print_res("Fixed (matched keep)", fixed); results["fixed"] = fixed

    # Random
    rand = evaluate_randomized_matched_sparsity(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        target_keep_effective=tgt_keep,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_prune_keep=tgt_prune,
        target_quant_ratio=tgt_qratio,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
    )
    _print_res("Random (matched keep)", rand); results["random"] = rand

    # Drift-aware
    drift = evaluate_drift_aware_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        target_keep_effective=tgt_keep,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_prune_keep=getattr(cfg, "C_target_prune", 1.0),
        target_quant_ratio=float(getattr(cfg, "C_target_quant_bits", 16))/16.0,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
    )
    _print_res("Drift-aware (matched)", drift); results["drift_aware"] = drift

    # EMC
    emc = evaluate_emc_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        target_keep_effective=tgt_keep,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_prune_keep=getattr(cfg, "C_target_prune", 1.0),
        target_quant_ratio=float(getattr(cfg, "C_target_quant_bits", 16))/16.0,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
    )
    _print_res("EMC (matched)", emc); results["emc"] = emc

    # LRM (long-range mass)
    lrm = evaluate_lrm_tokens_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        target_keep_effective=tgt_keep,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_prune_keep=tgt_prune,
        target_quant_ratio=tgt_qratio,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
    )
    _print_res("LRM (matched keep)", lrm); results["lrm"] = lrm

elif E.study_type == "pruning":
    # Fixed
    fixed = evaluate_fixed_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_keep_effective=tgt_keep,
        target_prune_keep=tgt_prune,
        target_quant_ratio=tgt_qratio,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
    )
    _print_res("Fixed (matched prune)", fixed); results["fixed"] = fixed

    # Random (generic randomized matcher works; with κ & q fixed, it randomizes prune)
    rand = evaluate_randomized_matched_sparsity(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        target_keep_effective=tgt_keep,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_prune_keep=tgt_prune,
        target_quant_ratio=tgt_qratio,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
    )
    _print_res("Random (matched prune)", rand); results["random"] = rand

    # ECov
    ecov = evaluate_ecov_prune_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_prune_keep=tgt_prune,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        coverage_target=float(getattr(cfg, "coverage_target", 0.90)),
        struct_on_non_eff=False,
    )
    _print_res("ECov (matched prune)", ecov); results["ecov"] = ecov

    # DCP
    dcp = evaluate_dcp_prune_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_prune_keep=tgt_prune,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
        cov_target=float(getattr(cfg, "coverage_target", 0.90)),
        dcp_gamma=float(getattr(cfg, "dcp_gamma", 0.35)),
    )
    _print_res("DCP (matched prune)", dcp); results["dcp"] = dcp

    # Margin (top1-top2 logit margin controller; pruning only, κ=1.0, bits fixed)
    margin = evaluate_margin_prune_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,  # must be (1.0,) inside impl
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_prune_keep=tgt_prune,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
        margin_smooth=float(getattr(cfg, "margin_smooth", 0.05)),
        gamma=float(getattr(cfg, "margin_gamma", 0.35)),
    )
    _print_res("Margin (matched prune)", margin)
    results["margin"] = margin
elif E.study_type == "quantization":
    # Fixed
    fixed = evaluate_fixed_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_keep_effective=tgt_keep,
        target_prune_keep=tgt_prune,
        target_quant_ratio=tgt_qratio,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
    )
    _print_res("Fixed (matched quant)", fixed); results["fixed"] = fixed

    # Random (generic randomized matcher; with κ & prune fixed, it randomizes bits)
    rand = evaluate_randomized_matched_sparsity(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        target_keep_effective=tgt_keep,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_prune_keep=tgt_prune,
        target_quant_ratio=tgt_qratio,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
    )
    _print_res("Random (matched quant)", rand); results["random"] = rand

    # QNR
    qnr = evaluate_qnr_quant_matched_keep(
        cfg, model, _dl(),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_quant_ratio=tgt_qratio,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        Ts=cfg.Ts, Tw=cfg.Tw,
        struct_on_non_eff=False,
    )
    _print_res("QNR (matched quant)", qnr); results["qnr"] = qnr

    # DynR
    dynr = evaluate_dynr_quant_matched_keep(
        cfg, model, _dl(),
        Ts=cfg.Ts, Tw=cfg.Tw,
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_quant_ratio=tgt_qratio,
        context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
        struct_on_non_eff=False,
        keep_fracs=cfg.keep_fracs,
    )
    _print_res("DynR (matched quant)", dynr); results["dynr"] = dynr

else:
    raise ValueError(f"Unknown study_type: {E.study_type}")

# ----------------- Summary -----------------

print("\n\n=== Summary (validation) ===")
print(f"Dense\t\t\t\t: ppl={dense['ppl']:.3f}\ttokens={dense['tokens']}")
print(
    f"Policy\t\t\t\t: ppl={policy_out['ppl']:.3f}  "
    f"keep_all={policy_out.get('avg_keep_all', 1.0):.3f}  "
    f"keep_eff={policy_out.get('avg_keep_effective', policy_out.get('avg_keep_all', 1.0)):.3f}  "
    f"prune_keep={policy_out.get('avg_prune_keep', 1.0):.3f}  "
    f"quant_ratio={16*policy_out.get('avg_quant_ratio', 1.0):.3f}  "
    f"tokens={policy_out['tokens_effective']}/{policy_out['tokens']}"
)

def _maybe(name):
    if name in results:
        r = results[name]
        print(
            f"{name.replace('_',' ').title():<24}: ppl={r.get('ppl', float('nan')):.3f}  "
            f"keep_all={r.get('avg_keep_all', 1.0):.3f}  "
            f"keep_eff={r.get('avg_keep_effective', r.get('avg_keep_all', 1.0)):.3f}  "
            f"prune_keep={r.get('avg_prune_keep', 1.0):.3f}  "
            f"quant_ratio={16*r.get('avg_quant_ratio', 1.0):.3f}  "
            f"tokens={r.get('tokens_effective', 0)}/{r.get('tokens', 0)}"
        )

for key in ["fixed", "random", "drift_aware", "emc", "lrm", "ecov", "dcp", "margin", "qnr", "dynr"]:
    _maybe(key)

# ----------------- CSV -----------------

csv_path = f"{E.study_type}_11055_ckpt_perplexities.csv"
ckpt_dir_last = os.path.basename(os.path.normpath(E.CKPT_DIR))

# Assemble row with a superset of fields (missing baselines left blank)
def _val(d, k, default=""):
    return d.get(k, default) if isinstance(d, dict) else default

row = {
    "study_type": E.study_type,
    "ckpt_dir": ckpt_dir_last,
    "mode": E.mode,
    "dataset_name": E.dataset_name,
    "sparsity_bias": float(cfg.eval_sparsity_bias),
    "prune_bias": float(cfg.eval_prune_bias),
    "quant_bias": float(cfg.eval_quant_bias),

    "dense_ppl": float(dense["ppl"]),
    "policy_ppl": float(policy_out["ppl"]),
    "fixed_ppl": float(_val(results.get("fixed", {}), "ppl", float("nan"))),
    "random_ppl": float(_val(results.get("random", {}), "ppl", float("nan"))),
    "drift_aware_ppl": float(_val(results.get("drift_aware", {}), "ppl", float("nan"))),
    "emc_ppl": float(_val(results.get("emc", {}), "ppl", float("nan"))),
    "lrm_ppl": float(_val(results.get("lrm", {}), "ppl", float("nan"))),
    "ecov_ppl": float(_val(results.get("ecov", {}), "ppl", float("nan"))),
    "dcp_ppl": float(_val(results.get("dcp", {}), "ppl", float("nan"))),
    "margin_ppl": float(_val(results.get("margin", {}), "ppl", float("nan"))),
    "qnr_ppl": float(_val(results.get("qnr", {}), "ppl", float("nan"))),
    "dynr_ppl": float(_val(results.get("dynr", {}), "ppl", float("nan"))),

    "dense_keep_all": float(dense.get("avg_keep_all", 1.0)),
    "policy_keep_all": float(policy_out.get("avg_keep_all", 1.0)),
    "policy_keep_eff": float(policy_out.get("avg_keep_effective", policy_out.get("avg_keep_all", 1.0))),
    "policy_prune_keep": float(policy_out.get("avg_prune_keep", 1.0)),
    "policy_quant_ratio": float(policy_out.get("avg_quant_ratio", 1.0)),

    "fixed_keep_all": float(_val(results.get("fixed", {}), "avg_keep_all", 1.0) or 1.0),
    "fixed_prune_keep": float(_val(results.get("fixed", {}), "avg_prune_keep", 1.0) or 1.0),
    "fixed_quant_ratio": float(_val(results.get("fixed", {}), "avg_quant_ratio", 1.0) or 1.0),

    "random_keep_all": float(_val(results.get("random", {}), "avg_keep_all", 1.0) or 1.0),
    "random_prune_keep": float(_val(results.get("random", {}), "avg_prune_keep", 1.0) or 1.0),
    "random_quant_ratio": float(_val(results.get("random", {}), "avg_quant_ratio", 1.0) or 1.0),


    "drift_aware_prune_keep": float(_val(results.get("drift_aware", {}), "avg_prune_keep", 1.0) or 1.0),
    "drift_aware_quant_ratio": float(_val(results.get("drift_aware", {}), "avg_quant_ratio", 1.0) or 1.0),

    "emc_prune_keep": float(_val(results.get("emc", {}), "avg_prune_keep", 1.0) or 1.0),
    "emc_quant_ratio": float(_val(results.get("emc", {}), "avg_quant_ratio", 1.0) or 1.0),

    "lrm_prune_keep": float(_val(results.get("lrm", {}), "avg_prune_keep", 1.0) or 1.0),
    "lrm_quant_ratio": float(_val(results.get("lrm", {}), "avg_quant_ratio", 1.0) or 1.0),

    "ecov_prune_keep": float(_val(results.get("ecov", {}), "avg_prune_keep", 1.0) or 1.0),
    "ecov_quant_ratio": float(_val(results.get("ecov", {}), "avg_quant_ratio", 1.0) or 1.0),

    "dcp_prune_keep": float(_val(results.get("dcp", {}), "avg_prune_keep", 1.0) or 1.0),
    "dcp_quant_ratio": float(_val(results.get("dcp", {}), "avg_quant_ratio", 1.0) or 1.0),

    "margin_prune_keep": float(_val(results.get("margin", {}), "avg_prune_keep", 1.0) or 1.0),
    "margin_quant_ratio": float(_val(results.get("margin", {}), "avg_quant_ratio", 1.0) or 1.0),

    "qnr_prune_keep": float(_val(results.get("qnr", {}), "avg_prune_keep", 1.0) or 1.0),
    "qnr_quant_ratio": float(_val(results.get("qnr", {}), "avg_quant_ratio", 1.0) or 1.0),

    "dynr_prune_keep": float(_val(results.get("dynr", {}), "avg_prune_keep", 1.0) or 1.0),
    "dynr_quant_ratio": float(_val(results.get("dynr", {}), "avg_quant_ratio", 1.0) or 1.0),
}

fieldnames = list(row.keys())
file_exists = os.path.exists(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"\n[csv] Appended results to {csv_path}")
