#!/usr/bin/env python3
"""
gen_policy_vs_dense.py

Minimal, no-lm-eval script to sanity-check continuous generation from:
  1) Policy-controlled sparse decoding (episodic refresh)
  2) Dense baseline (no policy, no sparse masks)

Requires your repo modules and eval_policy_lmeval.PolicyHarnessLM.

Example:
  python gen_policy_vs_dense.py \
    --ckpt_dir /path/to/ckpts \
    --input_sentence "The quick brown fox jumps over the lazy dog." \
    --generation_tokens 256 \
    --temperature 0.0

Notes:
- Uses the same tokenizer/model/policy loading as your eval script.
- Prints just the continuations (not the prompt) for easy visual compare.
"""

import argparse
import random
import numpy as np
import torch

# Pull the existing, battle-tested wrapper & runner from your provided file.
# This keeps the script tiny and avoids duplicating logic.
from eval_policy_lmeval import PolicyHarnessLM  # noqa: E402
import torch, torch.nn.functional as F

def avg_ll_under_dense(dense_model: PolicyHarnessLM, prompt: str, continuation: str) -> float:
    tok = dense_model.tok
    ctx_ids = tok.encode(prompt, add_special_tokens=False)
    cont_ids = tok.encode(continuation, add_special_tokens=False)
    if not cont_ids:
        return float('nan')

    device = dense_model.runner.device
    m = dense_model.runner.m

    # one pass over prompt+continuation
    full = torch.tensor([ctx_ids + cont_ids], device=device)
    with torch.no_grad():
        out = m(input_ids=full, return_dict=True)
        # logits at positions that predict the continuation tokens
        start = len(ctx_ids) - 1                     # position that predicts cont[0]
        end = start + len(cont_ids)                  # exclusive
        logits = out.logits[:, start:end, :]         # [1, len(cont), V]

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


def generate_once(
    model: PolicyHarnessLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> str:
    """Generate a single continuation using the already-initialized model."""
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

    # Policy controls (mirrors your original defaults/knobs)
    p.add_argument("--policy_temperature", type=float, default=0.6, help="Temperature for policy action selection")
    # Default greedy; allow flipping to stochastic policy with --stochastic_policy
    g = p.add_mutually_exclusive_group()
    g.add_argument("--greedy_policy", dest="greedy_policy", action="store_true", help="Use argmax for κ actions")
    g.add_argument("--stochastic_policy", dest="greedy_policy", action="store_false", help="Sample κ actions")
    p.set_defaults(greedy_policy=True)

    p.add_argument("--episode_len", type=int, default=None, help="Override episode length (default cfg.rollout_len)")
    p.add_argument("--dense_refresh_tail", type=int, default=None, help="Tail tokens to dense-prefill between episodes")
    p.add_argument("--batch_size", type=int, default=1, help="Internal LM-Eval wrapper batch size (safe to keep at 1)")
    p.add_argument("--seed", type=int, default=None, help="Set RNG seed for reproducibility")

    args = p.parse_args()
    set_seed(args.seed)

    # Load the policy-driven model (sparse path)
    policy_model = PolicyHarnessLM(
        ckpt_dir=args.ckpt_dir,
        mode=args.mode,
        greedy_policy=args.greedy_policy,
        policy_temperature=args.policy_temperature,
        episode_len=args.episode_len,
        dense_refresh_tail=args.dense_refresh_tail,
        dense_only=False,
        max_batch=args.batch_size,
    )

    # Load a dense-only baseline model (same weights/tokenizer, no policy/masks)
    dense_model = PolicyHarnessLM(
        ckpt_dir=args.ckpt_dir,
        mode=args.mode,
        greedy_policy=True,  # ignored in dense mode
        policy_temperature=args.policy_temperature,  # ignored in dense mode
        episode_len=args.episode_len,               # ignored in dense mode
        dense_refresh_tail=args.dense_refresh_tail, # ignored in dense mode
        dense_only=True,
        max_batch=args.batch_size,
    )

    prompt = args.input_sentence
    print("\n============================")
    print("Prompt:\n")
    print(prompt)
    print("\n============================")

    # Policy (sparse) generation
    policy_text = generate_once(
        policy_model,
        prompt=prompt,
        max_new_tokens=args.generation_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Dense baseline generation
    dense_text = generate_once(
        dense_model,
        prompt=prompt,
        max_new_tokens=args.generation_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Show results
    print("\n--- Policy (sparse) continuation ---\n")
    print(policy_text)
    print(f"\n[chars: {len(policy_text)}]")

    print("\n--- Dense baseline continuation ---\n")
    print(dense_text)
    print(f"\n[chars: {len(dense_text)}]\n")
    policy_ll = avg_ll_under_dense(dense_model, prompt, policy_text)
    dense_ll  = avg_ll_under_dense(dense_model, prompt, dense_text)
    print(f"Mean logprob under dense LM -> policy: {policy_ll:.4f}, dense: {dense_ll:.4f}")

if __name__ == "__main__":
    main()
