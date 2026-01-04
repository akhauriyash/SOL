import os
import json
import time
import csv
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.backends.cuda as sdp

from utils.seeds import set_seed
from utils.model import load_lm_and_tokenizer, unwrap
from utils.data import make_dataloader, limited_dl
from utils.eval import evaluate_stateful_policy_rollout
from utils.eval_baselines import (
    evaluate_fixed_matched_keep,
    evaluate_randomized_matched_sparsity,
    evaluate_emc_matched_structured,
    evaluate_drift_aware_matched_structured,
)
from utils.config import Config
from predictor import RecurrentActorCriticPolicy
from utils.actions import build_action_spec

# SDPA: mem-efficient only
sdp.enable_flash_sdp(False)
sdp.enable_math_sdp(False)
sdp.enable_mem_efficient_sdp(True)


@dataclass
class EvalCfg:
    ckpt_dir: str = ""
    eval_batches: Optional[int] = None   # None = run full loader
    split: str = "validation"
    mode: str = "latest"                # "latest" or "best"
    dataset_name: Optional[str] = "wikitext"
    dataset_config: Optional[str] = "en"
    text_field: Optional[str] = "text"
    batch_size: Optional[int] = 16
    seed: int = 1234
    greedy: bool = True
    policy_temperature: float = 0.6


def find_latest_ckpt(ckpt_dir: str, mode: str) -> Optional[str]:
    """
    Prefer policy_{mode}.pt, otherwise fall back to the latest policy_epoch*.pt.
    """
    if not os.path.isdir(ckpt_dir):
        return None

    preferred = os.path.join(ckpt_dir, f"policy_{mode}.pt")
    if os.path.exists(preferred):
        return preferred

    cands = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith("policy_epoch") and f.endswith(".pt")
    ]
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
      2) YAML recorded in 'meta.config_paths.base' -> ckpt_dir/code/<relpath>
         (or train_meta.json)
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

    # Device / dtype
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Dataset overrides (optional)
    if dataset_name is not None:
        cfg.dataset_name = dataset_name
    if dataset_config is not None:
        cfg.dataset_config = dataset_config
    if text_field is not None:
        cfg.text_field = text_field
    if batch_size is not None:
        cfg.batch_size = batch_size

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL policy (prune/quant/toksparse) and log results."
    )

    # Checkpoint location
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Directory containing policy checkpoints (policy_latest.pt, etc.).")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Optional explicit path to a checkpoint .pt file. "
                             "If set, overrides --ckpt_dir/--mode.")
    parser.add_argument("--mode", type=str, default="latest", choices=["latest", "best"])

    # Data options
    parser.add_argument("--dataset_name", type=str,
                        choices=["wikitext", "allenai/c4"], default=None)
    parser.add_argument("--eval_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=1234)

    # Targets (budgets)
    parser.add_argument(
        "--tgt_keep",
        type=float,
        default=None,
        help="Target effective keep ratio (tokens_effective / tokens) in [0,1].",
    )
    parser.add_argument(
        "--tgt_prune_keep",
        type=float,
        default=None,
        help="Target structured prune keep ratio in [0,1]. "
             "For your pruning experiments, use 0.6.",
    )
    parser.add_argument(
        "--tgt_quant_ratio",
        type=float,
        default=None,
        help="Target quantization ratio in [0,1], where bits = 16 * ratio. "
             "Alternative to --tgt_quant_bits.",
    )
    parser.add_argument(
        "--tgt_quant_bits",
        type=float,
        default=None,
        help="Target quantization *bit* budget. E.g. 7 -> ratio 7/16 ≈ 0.4375 "
             "for your quantization experiments.",
    )

    # Optional eval biases
    parser.add_argument("--sparsity_bias", type=float, default=None,
                        help="Optional eval sparsity bias.")
    parser.add_argument("--quant_bias", type=float, default=None,
                        help="Optional eval quantization bias.")
    parser.add_argument("--prune_bias", type=float, default=None,
                        help="Optional eval pruning bias.")

    # CSV logging
    parser.add_argument("--csv_path", type=str, default="policy_eval_results.csv",
                        help="CSV file to append results to.")
    parser.add_argument(
        "--num_trials",
        type=int,
        default=0,
        help="Number of random allocation trials to run as a baseline (0 = skip).",
    )
    parser.add_argument(
        "--do_emc_and_driftaware",
        action="store_true",
        help="If set, also evaluate EMC and Drift-Aware (DAC) matched baselines at the achieved "
             "policy budgets (keep/prune/quant) and write results to the CSV.",
    )
    args = parser.parse_args()

    # -------------------- Build EvalCfg -------------------- #
    if args.ckpt_path is None and args.ckpt_dir is None:
        parser.error("You must specify either --ckpt_dir or --ckpt_path.")

    E = EvalCfg()
    E.split = args.split
    E.mode = args.mode
    if args.eval_batches is not None:
        E.eval_batches = args.eval_batches
    E.seed = args.seed

    if args.dataset_name is not None:
        E.dataset_name = args.dataset_name
        # follow your pattern: wikitext vs c4
        E.dataset_config = (
            "wikitext-2-raw-v1" if args.dataset_name == "wikitext" else "en"
        )

    if args.batch_size is not None:
        E.batch_size = args.batch_size

    # Resolve checkpoint path
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
        if args.ckpt_dir is not None:
            E.ckpt_dir = args.ckpt_dir
        else:
            E.ckpt_dir = os.path.dirname(ckpt_path)
    else:
        E.ckpt_dir = args.ckpt_dir
        ckpt_path = find_latest_ckpt(E.ckpt_dir, E.mode)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {E.ckpt_dir}")

    print(f"[eval] Using checkpoint: {ckpt_path}")

    # -------------------- Config / model / data -------------------- #
    set_seed(E.seed)

    cfg = load_cfg_from_checkpoint_or_yaml(
        ckpt_dir=E.ckpt_dir,
        ckpt_path=ckpt_path,
        dataset_name=E.dataset_name,
        dataset_config=E.dataset_config,
        text_field=E.text_field,
        batch_size=E.batch_size,
    )

    # Biases for eval
    cfg.eval_sparsity_bias = float(getattr(cfg, "eval_sparsity_bias", 0.0))
    cfg.eval_quant_bias = float(getattr(cfg, "eval_quant_bias", 0.0))
    cfg.eval_prune_bias = float(getattr(cfg, "eval_prune_bias", 0.0))

    if args.sparsity_bias is not None:
        cfg.eval_sparsity_bias = float(args.sparsity_bias)
    if args.quant_bias is not None:
        cfg.eval_quant_bias = float(args.quant_bias)
    if args.prune_bias is not None:
        cfg.eval_prune_bias = float(args.prune_bias)

    # Optionally assert criteria if you want to keep it strict
    # assert getattr(cfg, "sparsity_criteria", "quest") == "quest", \
    #     "This eval script currently assumes 'quest' sparsity criteria."

    # Load LM + tokenizer
    tok, model = load_lm_and_tokenizer(cfg)

    # Data loader
    dl = make_dataloader(
        cfg,
        tok,
        split=E.split,
        shuffle=False,
        distributed=False,
    )

    # Infer model dims
    base_model = unwrap(model)
    hidden_size = getattr(base_model.config, "hidden_size",
                          getattr(base_model.config, "n_embd", None))
    if hidden_size is None:
        raise ValueError("Could not infer hidden size from model.config")

    # Re-load checkpoint on correct device for policy weights / meta
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

    # Embedding dim
    emb_layer = unwrap(model).get_input_embeddings()
    embed_dim = getattr(emb_layer, "embedding_dim", emb_layer.weight.shape[1])

    # Policy dimensions
    pol_d_model = int(getattr(cfg, "policy_d_model", 768))
    pol_heads = int(getattr(cfg, "policy_n_heads", 8))
    pol_layers = int(getattr(cfg, "policy_n_layers", 2))
    pol_mlp_mult = float(getattr(cfg, "policy_mlp_ratio", 4.0))
    pol_act_dim = int(getattr(cfg, "policy_action_dim", 32))
    pol_max_len = int(
        getattr(cfg, "policy_max_len", max(1024, cfg.rollout_len + 8))
    )
    SCALAR_D = int(getattr(cfg, "policy_scalar_dim", 8))

    # Action spec (uses the model's default actions)
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

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy parameters: {n_params:,}")

    # Load policy state dict
    state_key = "policy_state_dict" if "policy_state_dict" in sd else "state_dict"
    policy.load_state_dict(sd[state_key], strict=True)
    policy.eval()

    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"Device={cfg.device}, dtype={cfg.dtype}, model={cfg.model_name}")
    print(f"Eval split='{E.split}', batches={E.eval_batches}, batch_size={cfg.batch_size}")
    print(f"Structure: Ts={cfg.Ts}  Tw={cfg.Tw}  keep_fracs={cfg.keep_fracs}")
    print(f"context_len={cfg.context_len} rollout_len={cfg.rollout_len}")
    print(f"RL policy eval settings: greedy={E.greedy}, temperature={E.policy_temperature}")

    # -------------------- Targets / budgets -------------------- #
    eval_C_tok = args.tgt_keep          # effective keep ratio in [0,1] or None
    eval_C_pru = args.tgt_prune_keep    # structured prune keep ratio or None

    # quant bits -> quant ratio -> bits for evaluator
    if args.tgt_quant_bits is not None:
        eval_C_qbits = float(args.tgt_quant_bits)
    elif args.tgt_quant_ratio is not None:
        eval_C_qbits = 16.0 * float(args.tgt_quant_ratio)
    else:
        eval_C_qbits = None

    # -------------------- Policy rollout -------------------- #
    start_time = time.time()
    greedy = evaluate_stateful_policy_rollout(
        cfg,
        model,
        policy,
        limited_dl(dl, E.eval_batches),
        Ts=cfg.Ts,
        Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        context_len=cfg.context_len,
        rollout_len=cfg.rollout_len,
        device=cfg.device,
        greedy=E.greedy,
        temperature=E.policy_temperature,
        target_C_tok=eval_C_tok,
        target_C_pru=eval_C_pru,
        target_C_qbits=eval_C_qbits,
        sparsity_bias=cfg.eval_sparsity_bias,
        prune_bias=cfg.eval_prune_bias,
        quant_bias=cfg.eval_quant_bias,
    )
    total_time = time.time() - start_time

    print(
        f"\nPolicy (greedy={'yes' if E.greedy else 'no'})\t: "
        f"ppl={greedy['ppl']:.3f}  "
        f"keep_all={greedy['avg_keep_all']:.3f}  "
        f"prune_keep={greedy['avg_prune_keep']:.3f}  "
        f"quant_ratio={16*greedy['avg_quant_ratio']:.3f}\t"
        f"tokens={greedy['tokens_effective']}/{greedy['tokens']}\n"
    )
    print(f"Policy rollout evaluation complete in {total_time:.2f} seconds.")
    print(f"Policy Action Probs \t: {[f'{p:.3f}' for p in greedy['action_probs']]}")

    # Actual policy budgets (what it really chose)
    policy_keep_eff_actual = float(greedy["avg_keep_effective"])
    policy_prune_keep_actual = float(greedy["avg_prune_keep"])
    policy_quant_ratio_actual = float(greedy["avg_quant_ratio"])

    # Baseline targets = actual policy budgets
    target_keep_effective = policy_keep_eff_actual
    target_prune_keep = policy_prune_keep_actual
    target_quant_ratio = policy_quant_ratio_actual

    print(
        f"\n[targets] keep_eff target={target_keep_effective:.3f} "
        f"(policy_actual={policy_keep_eff_actual:.3f})"
    )
    print(
        f"[targets] prune_keep target={target_prune_keep:.3f} "
        f"(policy_actual={policy_prune_keep_actual:.3f})"
    )
    print(
        f"[targets] quant_ratio target={target_quant_ratio:.3f} "
        f"(policy_actual={policy_quant_ratio_actual:.3f})"
    )

    # -------------------- Fixed matched baseline -------------------- #
    start_time = time.time()
    fixed_matched = evaluate_fixed_matched_keep(
        cfg,
        model,
        limited_dl(
            make_dataloader(cfg, tok, split=E.split, shuffle=False, distributed=False),
            E.eval_batches,
        ),
        Ts=cfg.Ts,
        Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_keep_effective=target_keep_effective,
        target_prune_keep=target_prune_keep,
        target_quant_ratio=target_quant_ratio,
        context_len=cfg.context_len,
        rollout_len=cfg.rollout_len,
        device=cfg.device,
        struct_on_non_eff=False,
    )
    total_time = time.time() - start_time

    print(
        f"\nFixed (matched keep)  \t\t: "
        f"ppl={fixed_matched['ppl']:.3f}  "
        f"keep_all={fixed_matched['avg_keep_all']:.3f}  "
        f"prune_keep={fixed_matched['avg_prune_keep']:.3f}  "
        f"quant_ratio={16*fixed_matched['avg_quant_ratio']:.3f}\t"
        f"tokens={fixed_matched['tokens_effective']}/{fixed_matched['tokens']}\n"
    )
    print(f"Fixed matched evaluation complete in {total_time:.2f} seconds.")
    print(f"Fixed Action Probs \t: {[f'{p:.3f}' for p in fixed_matched['action_probs']]}")
    # -------------------- EMC + Drift-Aware baselines (optional) -------------------- #
    emc_matched = None
    dac_matched = None
    if args.do_emc_and_driftaware:
        # EMC
        start_time = time.time()
        emc_matched = evaluate_emc_matched_structured(
            cfg,
            model,
            limited_dl(
                make_dataloader(cfg, tok, split=E.split, shuffle=False, distributed=False),
                E.eval_batches,
            ),
            Ts=cfg.Ts,
            Tw=cfg.Tw,
            keep_fracs=cfg.keep_fracs,
            prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
            quant_choices=getattr(cfg, "quant_choices", ("q16",)),
            target_keep_effective=target_keep_effective,
            target_prune_keep=target_prune_keep,
            target_quant_ratio=target_quant_ratio,
            context_len=cfg.context_len,
            rollout_len=cfg.rollout_len,
            device=cfg.device,
            struct_on_non_eff=False,
        )
        total_time = time.time() - start_time
        print(
            f"\nEMC (matched targets)\t\t: "
            f"ppl={emc_matched['ppl']:.3f}  "
            f"keep_all={emc_matched['avg_keep_all']:.3f}  "
            f"keep_eff={emc_matched['avg_keep_effective']:.3f}  "
            f"prune_keep={emc_matched['avg_prune_keep']:.3f}  "
            f"quant_ratio={16*emc_matched['avg_quant_ratio']:.3f}\t"
            f"tokens={emc_matched['tokens_effective']}/{emc_matched['tokens']}  "
            f"(time={total_time:.2f}s)\n"
        )

        # Drift-Aware (DAC)
        start_time = time.time()
        dac_matched = evaluate_drift_aware_matched_structured(
            cfg,
            model,
            limited_dl(
                make_dataloader(cfg, tok, split=E.split, shuffle=False, distributed=False),
                E.eval_batches,
            ),
            Ts=cfg.Ts,
            Tw=cfg.Tw,
            keep_fracs=cfg.keep_fracs,
            prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
            quant_choices=getattr(cfg, "quant_choices", ("q16",)),
            target_keep_effective=target_keep_effective,
            target_prune_keep=target_prune_keep,
            target_quant_ratio=target_quant_ratio,
            context_len=cfg.context_len,
            rollout_len=cfg.rollout_len,
            device=cfg.device,
            struct_on_non_eff=False,
        )
        total_time = time.time() - start_time
        print(
            f"\nDAC (matched targets)\t\t: "
            f"ppl={dac_matched['ppl']:.3f}  "
            f"keep_all={dac_matched['avg_keep_all']:.3f}  "
            f"keep_eff={dac_matched['avg_keep_effective']:.3f}  "
            f"prune_keep={dac_matched['avg_prune_keep']:.3f}  "
            f"quant_ratio={16*dac_matched['avg_quant_ratio']:.3f}\t"
            f"tokens={dac_matched['tokens_effective']}/{dac_matched['tokens']}  "
            f"(time={total_time:.2f}s)\n"
        )
    rand_ppl_trials = []
    rand_keep_all_trials = []
    rand_keep_eff_trials = []
    rand_prune_trials = []
    rand_quant_trials = []
    rand_tokens_eff_trials = []
    rand_tokens_trials = []
    rand_seeds = []

    if args.num_trials and args.num_trials > 0:
        print(
            f"\n[Random baseline] Running {args.num_trials} randomized trials "
            f"matched to policy budgets:"
        )
        print(
            f"    keep_eff={target_keep_effective:.3f}, "
            f"prune_keep={target_prune_keep:.3f}, "
            f"quant_ratio={target_quant_ratio:.3f}"
        )

        for i in range(args.num_trials):
            trial_seed = args.seed + i
            rand_seeds.append(trial_seed)
            set_seed(trial_seed)

            print(f"\n[rand trial {i+1}/{args.num_trials}] seed={trial_seed}")
            t0 = time.time()
            rand_res = evaluate_randomized_matched_sparsity(
                cfg,
                model,
                limited_dl(dl, E.eval_batches),
                Ts=cfg.Ts,
                Tw=cfg.Tw,
                keep_fracs=cfg.keep_fracs,
                prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
                quant_choices=getattr(cfg, "quant_choices", ("q16",)),
                target_keep_effective=target_keep_effective,
                target_prune_keep=target_prune_keep,
                target_quant_ratio=target_quant_ratio,
                context_len=cfg.context_len,
                rollout_len=cfg.rollout_len,
                device=cfg.device,
                struct_on_non_eff=False,
            )
            dt = time.time() - t0

            ppl_r = float(rand_res["ppl"])
            rand_ppl_trials.append(ppl_r)
            rand_keep_all_trials.append(float(rand_res["avg_keep_all"]))
            rand_keep_eff_trials.append(float(rand_res["avg_keep_effective"]))
            rand_prune_trials.append(float(rand_res["avg_prune_keep"]))
            rand_quant_trials.append(float(rand_res["avg_quant_ratio"]))
            rand_tokens_eff_trials.append(int(rand_res["tokens_effective"]))
            rand_tokens_trials.append(int(rand_res["tokens"]))

            print(
                f"[rand trial {i+1}] ppl={ppl_r:.3f}  "
                f"keep_all={rand_res['avg_keep_all']:.3f}  "
                f"keep_eff={rand_res['avg_keep_effective']:.3f}  "
                f"prune_keep={rand_res['avg_prune_keep']:.3f}  "
                f"quant_ratio={16 * rand_res['avg_quant_ratio']:.3f}  "
                f"tokens={rand_res['tokens_effective']}/{rand_res['tokens']}  "
                f"(time={dt:.2f}s)"
            )
    else:
        # no random trials requested
        rand_ppl_trials = []
        rand_keep_all_trials = []
        rand_keep_eff_trials = []
        rand_prune_trials = []
        rand_quant_trials = []
        rand_tokens_eff_trials = []
        rand_tokens_trials = []
        rand_seeds = []
    # -------------------- Pretty-printed summary -------------------- #
    print("\n\n=== Results (validation) ===\n")
    print(
        f"Policy (greedy={'yes' if E.greedy else 'no'})\t\t: "
        f"ppl={greedy['ppl']:.3f}  "
        f"keep_all={greedy['avg_keep_all']:.3f}  "
        f"keep_eff_actual={policy_keep_eff_actual:.3f}\t"
        f"tokens={greedy['tokens_effective']}/{greedy['tokens']}"
    )
    print(
        f"Fixed (matched keep @ target)\t: "
        f"ppl={fixed_matched['ppl']:.3f}  "
        f"keep_all={fixed_matched['avg_keep_all']:.3f}  "
        f"keep_eff_target={target_keep_effective:.3f}\t"
        f"tokens={fixed_matched['tokens_effective']}/{fixed_matched['tokens']}"
    )
    if args.do_emc_and_driftaware and emc_matched is not None:
        print(
            f"EMC (matched targets)\t\t: "
            f"ppl={emc_matched['ppl']:.3f}  keep_eff={emc_matched['avg_keep_effective']:.3f}  "
            f"prune_keep={emc_matched['avg_prune_keep']:.3f}  quant_bits={16*emc_matched['avg_quant_ratio']:.3f}"
        )
    if args.do_emc_and_driftaware and dac_matched is not None:
        print(
            f"DAC (matched targets)\t\t: "
            f"ppl={dac_matched['ppl']:.3f}  keep_eff={dac_matched['avg_keep_effective']:.3f}  "
            f"prune_keep={dac_matched['avg_prune_keep']:.3f}  quant_bits={16*dac_matched['avg_quant_ratio']:.3f}"
        )

    if "action_probs" in greedy:
        probs = ", ".join(f"{p:.2f}" for p in greedy["action_probs"])
        print(f"Sparsity levels (κ order)    : "
              f"[{', '.join(f'{k:.2f}' for k in cfg.keep_fracs)}]")
        print(f"Policy action probs (κ order) : [{probs}]")

    if "action_probs" in fixed_matched:
        probs = ", ".join(f"{p:.2f}" for p in fixed_matched["action_probs"])
        print(f"Fixed action probs (κ order)  : [{probs}]")

    # -------------------- CSV logging -------------------- #
    csv_path = args.csv_path
    ckpt_dir_last = os.path.basename(os.path.normpath(E.ckpt_dir))

    sparsity_levels = "|".join(f"{k:.6f}" for k in cfg.keep_fracs)
    policy_action_probs = "|".join(f"{p:.6f}" for p in greedy.get("action_probs", []))
    fixed_action_probs = "|".join(f"{p:.6f}" for p in fixed_matched.get("action_probs", []))
    emc_action_probs = (
        "|".join(f"{p:.6f}" for p in emc_matched.get("action_probs", [])) if emc_matched is not None else ""
    )
    dac_action_probs = (
        "|".join(f"{p:.6f}" for p in dac_matched.get("action_probs", [])) if dac_matched is not None else ""
    )
    row = {
        "ckpt_dir": ckpt_dir_last,
        "ckpt_path": ckpt_path,
        "mode": E.mode,
        "dataset_name": cfg.dataset_name,
        "dataset_config": getattr(cfg, "dataset_config", None),
        "split": E.split,
        "eval_batches": E.eval_batches,

        # Targets used for the fixed baseline (matched to actual policy usage)
        "target_keep_effective": float(target_keep_effective),
        "target_prune_keep": float(target_prune_keep),
        "target_quant_ratio": float(target_quant_ratio),

        # Policy metrics
        "policy_ppl": float(greedy["ppl"]),
        "policy_keep_all": float(greedy["avg_keep_all"]),
        "policy_keep_effective_actual": float(policy_keep_eff_actual),
        "policy_prune_keep_actual": float(policy_prune_keep_actual),
        "policy_quant_ratio_actual": float(policy_quant_ratio_actual),

        # Fixed baseline metrics
        "fixed_ppl": float(fixed_matched["ppl"]),
        "fixed_keep_all": float(fixed_matched["avg_keep_all"]),
        "fixed_prune_keep": float(fixed_matched["avg_prune_keep"]),
        "fixed_quant_ratio": float(fixed_matched["avg_quant_ratio"]),

        "sparsity_levels_kappa_order": sparsity_levels,
        "policy_action_probs_kappa_order": policy_action_probs,
        "fixed_action_probs_kappa_order": fixed_action_probs,

        "do_emc_and_driftaware": bool(args.do_emc_and_driftaware),

        # EMC baseline metrics (optional)
        "emc_ppl": float(emc_matched["ppl"]) if emc_matched is not None else None,
        "emc_keep_all": float(emc_matched["avg_keep_all"]) if emc_matched is not None else None,
        "emc_keep_effective": float(emc_matched["avg_keep_effective"]) if emc_matched is not None else None,
        "emc_prune_keep": float(emc_matched["avg_prune_keep"]) if emc_matched is not None else None,
        "emc_quant_ratio": float(emc_matched["avg_quant_ratio"]) if emc_matched is not None else None,
        "emc_action_probs": emc_action_probs,

        # Drift-Aware (DAC) baseline metrics (optional)
        "dac_ppl": float(dac_matched["ppl"]) if dac_matched is not None else None,
        "dac_keep_all": float(dac_matched["avg_keep_all"]) if dac_matched is not None else None,
        "dac_keep_effective": float(dac_matched["avg_keep_effective"]) if dac_matched is not None else None,
        "dac_prune_keep": float(dac_matched["avg_prune_keep"]) if dac_matched is not None else None,
        "dac_quant_ratio": float(dac_matched["avg_quant_ratio"]) if dac_matched is not None else None,
        "dac_action_probs": dac_action_probs,

        "sparsity_criteria": getattr(cfg, "sparsity_criteria", None),

        "sparsity_bias": cfg.eval_sparsity_bias,
        "prune_bias": cfg.eval_prune_bias,
        "quant_bias": cfg.eval_quant_bias,

        "tgt_keep_cli": float(eval_C_tok) if eval_C_tok is not None else None,
        "tgt_prune_keep_cli": float(eval_C_pru) if eval_C_pru is not None else None,
        "tgt_quant_ratio_cli": float(eval_C_qbits / 16.0) if eval_C_qbits is not None else None,
        "tgt_quant_bits_cli": float(eval_C_qbits) if eval_C_qbits is not None else None,
    
        "rand_num_trials": int(args.num_trials),
        "rand_seeds": "|".join(str(s) for s in rand_seeds),
        "rand_ppl_trials": "|".join(f"{x:.6f}" for x in rand_ppl_trials),
        "rand_keep_all_trials": "|".join(f"{x:.6f}" for x in rand_keep_all_trials),
        "rand_keep_eff_trials": "|".join(f"{x:.6f}" for x in rand_keep_eff_trials),
        "rand_prune_keep_trials": "|".join(f"{x:.6f}" for x in rand_prune_trials),
        "rand_quant_ratio_trials": "|".join(f"{x:.6f}" for x in rand_quant_trials),
        "rand_tokens_eff_trials": "|".join(str(x) for x in rand_tokens_eff_trials),
        "rand_tokens_trials": "|".join(str(x) for x in rand_tokens_trials),

}

    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n[csv] Appended results to {csv_path}")


if __name__ == "__main__":
    main()
