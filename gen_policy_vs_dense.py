#!/usr/bin/env python3
"""
Example:
  python gen_policy_vs_dense.py \
    --ckpt_dir /path/to/ckpts \
    --input_sentence "The quick brown fox jumps over the lazy dog." \
    --generation_tokens 256 \
    --temperature 0.0

"""

import argparse
import random
import numpy as np
from pprint import pprint
import torch
from policy_harness import PolicyHarnessLM
import torch, torch.nn.functional as F

def avg_ll_under_dense(dense_model: PolicyHarnessLM, prompt: str, continuation: str) -> float:
    tok = dense_model.tok
    ctx_ids = tok.encode(prompt, add_special_tokens=False)
    cont_ids = tok.encode(continuation, add_special_tokens=False)
    if not cont_ids:
        return float('nan')

    device = dense_model.runner.device
    m = dense_model.runner.m

    full = torch.tensor([ctx_ids + cont_ids], device=device)
    with torch.no_grad():
        out = m(input_ids=full, return_dict=True)
        start = len(ctx_ids) - 1
        end = start + len(cont_ids)
        logits = out.logits[:, start:end, :]

        targets = torch.tensor([cont_ids], device=device)
        logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(logprobs.mean().item())

def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_once_true_dense_from_cfg(
    cfg,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Minimal, vanilla load directly from cfg.model_name
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)

    load_kwargs = {"trust_remote_code": True}
    dt = getattr(cfg, "dtype", None)
    if dt is not None:
        load_kwargs["torch_dtype"] = dt
    m = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)

    device = getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
    m.to(device).eval()

    # Encode exactly the prompt (no BOS/EOS injection, no harness preprocessing)
    input_ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)

    gen_kwargs = {"max_new_tokens": int(max_new_tokens), "do_sample": (temperature > 0.0)}
    if temperature and temperature > 0.0:
        gen_kwargs["temperature"] = float(temperature)
    if top_p is not None:
        gen_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)

    # Pass EOS/PAD ids if present, without mutating the tokenizer
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    with torch.no_grad():
        out = m.generate(input_ids, **gen_kwargs)

    # Return only the continuation
    new_ids = out[0].tolist()[input_ids.shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True)

def generate_once_policy(
    model: PolicyHarnessLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> str:
    ctx_ids = model.tok.encode(prompt, add_special_tokens=False)
    gen_ids = model.runner.generate_with_policy(
        ctx_ids=ctx_ids,
        max_new_tokens=max_new_tokens,
        until=None,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        return_stats=True,
    )
    return model.tok.decode(gen_ids[0], skip_special_tokens=True), gen_ids[1]

def generate_once(
    model: PolicyHarnessLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> str:
    ctx_ids = model.tok.encode(prompt, add_special_tokens=False)
    gen_ids = model.runner.generate_with_policy(
        ctx_ids=ctx_ids,
        max_new_tokens=max_new_tokens,
        until=None,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return model.tok.decode(gen_ids, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="Directory with policy_*.pt")
    p.add_argument("--mode", type=str, default="latest", choices=["latest", "best"])
    p.add_argument(
        "--input_sentence",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Prompt to seed generation."
    )
    p.add_argument(
        "--generation_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate for each model."
    )
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation")
    p.add_argument("--top_p", type=float, default=None, help="Top-p nucleus filtering (optional)")
    p.add_argument("--top_k", type=int, default=None, help="Top-k filtering (optional)")
    p.add_argument("--policy_temperature", type=float, default=0.6, help="Temperature for policy action selection")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--greedy_policy", dest="greedy_policy", action="store_true", help="Use argmax for κ actions")
    g.add_argument("--stochastic_policy", dest="greedy_policy", action="store_false", help="Sample κ actions")
    p.set_defaults(greedy_policy=True)

    p.add_argument("--episode_len", type=int, default=None, help="Override episode length (default cfg.rollout_len)")
    p.add_argument("--dense_refresh_tail", type=int, default=10**9, help="[disabled] Tail tokens to dense-prefill between episodes")
    p.add_argument("--batch_size", type=int, default=1, help="Internal LM-Eval wrapper batch size (safe to keep at 1)")
    p.add_argument("--seed", type=int, default=None, help="Set RNG seed for reproducibility")
    p.add_argument("--sparsity_bias", type=float, default=0,
                        help="Positive values bias the policy toward sparser actions during eval.")
    p.add_argument("--quant_bias", type=float, default=0,
                        help="Positive values bias the policy toward more quantization during eval.")
    p.add_argument("--prune_bias", type=float, default=0,
                        help="Positive values bias the policy toward more pruning during eval.")

    args = p.parse_args()
    set_seed(args.seed)

    policy_model = PolicyHarnessLM(
        ckpt_dir=args.ckpt_dir,
        mode=args.mode,
        greedy_policy=args.greedy_policy,
        policy_temperature=args.policy_temperature,
        episode_len=args.episode_len,
        dense_refresh_tail=args.dense_refresh_tail,
        dense_only=False,
        max_batch=args.batch_size,
        sparsity_bias=args.sparsity_bias,
        prune_bias=args.prune_bias,
        quant_bias=args.quant_bias,
    )

    dense_model = PolicyHarnessLM(
        ckpt_dir=args.ckpt_dir,
        mode=args.mode,
        greedy_policy=True,
        policy_temperature=args.policy_temperature,
        episode_len=args.episode_len,
        dense_refresh_tail=args.dense_refresh_tail,
        dense_only=True,
        max_batch=args.batch_size,
    )

    prompt = args.input_sentence
    print("\n============================")
    print("Prompt:\n")
    print(prompt)
    print("\n============================")

    true_dense_text = generate_once_true_dense_from_cfg(
        policy_model.cfg,  # or dense_model.cfg — either has the same base cfg
        prompt=prompt,
        max_new_tokens=args.generation_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print("\n--- True Dense continuation ---\n")
    print(true_dense_text)
    print(f"\n[chars: {len(true_dense_text)}]\n")


    policy_text, stats = generate_once_policy(
        policy_model,
        prompt=prompt,
        max_new_tokens=args.generation_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print("\n--- Policy (sparse) continuation ---\n")
    print(policy_text)
    print(f"\n[chars: {len(policy_text)}]")

    dense_text = generate_once(
        dense_model,
        prompt=prompt,
        max_new_tokens=args.generation_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print("\n--- Dense baseline continuation ---\n")
    print(dense_text)
    print(f"\n[chars: {len(dense_text)}]\n")
    policy_ll = avg_ll_under_dense(dense_model, prompt, policy_text)
    dense_ll  = avg_ll_under_dense(dense_model, prompt, dense_text)
    print(f"Mean logprob under dense LM -> policy: {policy_ll:.4f}, dense: {dense_ll:.4f}")
    # pprint(stats)
    print("============================\n")
    print(f"Achieved Token-Keep-Rate\t{100*stats['keep_avg_eff']}%")
    print(f"Achieved Prune-Keep\t{100*stats['prune_avg_eff']}%")
    print(f"Achieved Quant-Ratio\t{stats['quant_ratio_avg_eff']} (1 = full 16-bit)\n")
    print("============================\n")

if __name__ == "__main__":
    main()
