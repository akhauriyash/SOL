# policy_harness.py
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.config import Config
from utils.model import load_lm_and_tokenizer, unwrap
from utils.masks import (
    build_sparse_attention_bias, enable_structured_controls, set_structured_action, clear_structured_action, clear_relevancy_keep, clear_quest_token_budgets
)
from policy_runtime import PolicyLMRunner
from utilities import find_latest_ckpt, load_cfg_from_checkpoint_or_yaml
from predictor import RecurrentActorCriticPolicy
from utils.actions import build_action_spec
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval import evaluator
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import numpy as np


class PolicyHarnessLM(LM):
    """
    LM-Eval-compatible wrapper that routes scoring & generation through the policy runner.
    """
    SUPPORTED_TASKS = None  # accept all
    REQ_CHUNK_SIZE = 1

    def __init__(
        self,
        ckpt_dir: str,
        mode: str = "latest",
        greedy_policy: bool = True,
        policy_temperature: float = 0.6,
        episode_len: Optional[int] = None,
        dense_refresh_tail: Optional[int] = None,
        dense_only: bool = False,
        max_batch: int = 4,
        sparsity_bias: float = 0.0,
        prune_bias: float = 0.0,
        quant_bias: float = 0.0,
    ):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = find_latest_ckpt(ckpt_dir, mode)
        if self.ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

        self.cfg = load_cfg_from_checkpoint_or_yaml(ckpt_dir, self.ckpt_path)
        self.tok, self.model = load_lm_and_tokenizer(self.cfg, dense_only=dense_only)

        base_model = unwrap(self.model)
        hidden_size = getattr(base_model.config, "hidden_size", getattr(base_model.config, "n_embd", None))
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model.config")
        emb_layer = unwrap(self.model).get_input_embeddings()
        embed_dim = getattr(emb_layer, "embedding_dim", emb_layer.weight.shape[1])

        pol_d_model  = int(getattr(self.cfg, "policy_d_model", 768))
        pol_heads    = int(getattr(self.cfg, "policy_n_heads", 8))
        pol_layers   = int(getattr(self.cfg, "policy_n_layers", 2))
        pol_mlp_mult = float(getattr(self.cfg, "policy_mlp_ratio", 4.0))
        pol_act_dim  = int(getattr(self.cfg, "policy_action_dim", 32))
        pol_max_len  = int(getattr(self.cfg, "policy_max_len", max(1024, self.cfg.rollout_len + 8)))
        SCALAR_D     = int(getattr(self.cfg, "policy_scalar_dim", 12))

        spec = build_action_spec(
            keep_fracs=self.cfg.keep_fracs,
            prune_choices=getattr(self.cfg, "struct_prune_choices", ("s100",)),
            quant_choices=getattr(self.cfg, "quant_choices", ("q16",)),
        )
        self.policy = RecurrentActorCriticPolicy(
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
        ).to(self.cfg.device, dtype=torch.float32)
        self.spec = spec
        sd = torch.load(self.ckpt_path, map_location=self.cfg.device)
        state_key = "policy_state_dict" if "policy_state_dict" in sd else "state_dict"
        self.policy.load_state_dict(sd[state_key], strict=True)
        self.policy.eval()
        gs = sd.get("global_step_state", {}) or {}
        lam_keep  = float(gs.get("lambda_keep",  0.0))
        lam_prune = float(gs.get("lambda_prune", 0.0))
        lam_quant = float(gs.get("lambda_quant", 0.0))

        # value = self.tripwire_mask_changes_logits(self.model, self.cfg)
        # print(f"Tripwire check: max logits change from mask hook = {value:.6f}. Success.")

        self.runner = PolicyLMRunner(
            cfg=self.cfg,
            model=self.model,
            policy=self.policy,
            tokenizer=self.tok,
            greedy_policy=greedy_policy,
            policy_temperature=policy_temperature,
            episode_len=episode_len if episode_len is not None else int(getattr(self.cfg, "rollout_len", 16)),
            dense_refresh_tail=dense_refresh_tail if dense_refresh_tail is not None else int(getattr(self.cfg, "Ts",0) + getattr(self.cfg, "Tw",0) + 1),
            dense_only=dense_only,
            lambda_keep=lam_keep,
            lambda_prune=lam_prune,
            lambda_quant=lam_quant,
            sparsity_bias=sparsity_bias,
            prune_bias=prune_bias,
            quant_bias=quant_bias,
        )

        self._max_batch = max_batch
        self.vocab_size = unwrap(self.model).get_output_embeddings().weight.size(0)

        self._eos = self.tok.eos_token_id

        self.per_request_stats = []
        self._agg = {
            "policy_steps": 0,
            "effective_steps": 0,
            "keep_sum_all": 0.0,
            "keep_sum_eff": 0.0,
            "prune_sum_all": 0.0,
            "prune_sum_eff": 0.0,
            "qratio_sum_all": 0.0,
            "qratio_sum_eff": 0.0,
            "action_hist": [0] * int(self.spec.n_actions),
        }

    @torch.no_grad()
    def tripwire_mask_changes_logits(self, model, cfg):
        model.eval()
        B = 1
        prelen = max(8, cfg.Ts + cfg.Tw + 3)
        prefix = torch.randint(low=0, high=512, size=(B, prelen), device=cfg.device)

        out = model(input_ids=prefix, use_cache=True, return_dict=True, output_hidden_states=True)
        past_kv_ref = out.past_key_values

        x = torch.randint(low=0, high=512, size=(B, 1), device=cfg.device)
        kv_len = torch.full((B,), prelen + 1, device=cfg.device, dtype=torch.long)
        pos_ids = (kv_len - 1).unsqueeze(1)  # continue after prefix

        keep_hi = torch.tensor([1.0], device=cfg.device)
        keep_lo = torch.tensor([0.2], device=cfg.device)

        crit = getattr(cfg, "sparsity_criteria", "recency").lower()

        def forward_with_keep(keep_fracs):
            clear_relevancy_keep(model)
            clear_quest_token_budgets(model)

            bias = build_sparse_attention_bias(
                model=model,
                past_kv_lens=kv_len,
                keep_fracs=keep_fracs,
                Ts=cfg.Ts,
                Tw=cfg.Tw,
                device=cfg.device,
                dtype=model.dtype,
                criteria=crit,
                tier=getattr(cfg, "relevancy_tier", "per_head"),
            )
            kwargs = dict(
                input_ids=x,
                use_cache=True,
                past_key_values=past_kv_ref,
                position_ids=pos_ids,
                return_dict=True,
            )
            if bias is not None:
                kwargs["attention_mask"] = bias

            return model(**kwargs).logits

        out_hi = forward_with_keep(keep_hi)
        out_lo = forward_with_keep(keep_lo)

        if torch.allclose(out_hi, out_lo, atol=0, rtol=0):
            raise AssertionError(
                f"Mask hook seems inactive for criteria='{crit}'. "
                "If using Relevancy/Quest, ensure attention impl is 'eager' and the hook is installed, "
                "and that a causal default is applied when attention_mask is None."
            )
        return float((out_hi - out_lo).abs().max().item())

    def _record_request_stats(self, req, stats: dict):
        task = getattr(req, "task_name", "unknown")
        index = getattr(req, "index", None)
        doc = getattr(req, "doc", None)
        doc_id = None
        if isinstance(doc, dict):
            for k in ("id", "doc_id", "sample_id", "query_id"):
                if k in doc: doc_id = doc[k]; break
        entry = {
            "task": task,
            "index": index,
            "doc_id": doc_id,
            "keep_fracs": list(self.spec.token_keep),
            "action_tags": list(self.spec.tags),
            **stats,
        }
        self.per_request_stats.append(entry)
        self._agg["policy_steps"] += stats.get("policy_steps", 0)
        self._agg["effective_steps"] += stats.get("effective_steps", 0)
        self._agg["keep_sum_all"] += stats.get("keep_sum_all", 0.0)
        self._agg["keep_sum_eff"] += stats.get("keep_sum_eff", 0.0)
        self._agg["prune_sum_all"] += stats.get("prune_sum_all", 0.0)
        self._agg["prune_sum_eff"] += stats.get("prune_sum_eff", 0.0)
        self._agg["qratio_sum_all"] += stats.get("qratio_sum_all", 0.0)
        self._agg["qratio_sum_eff"] += stats.get("qratio_sum_eff", 0.0)
        ah = stats.get("action_hist", [])
        for i, c in enumerate(ah):
            self._agg["action_hist"][i] += int(c)

    def export_sparsity_stats(self):
        g = self._agg
        total_actions = max(1, sum(g["action_hist"]))
        return {
            "global": {
                "keep_fracs": list(self.spec.token_keep),
                "action_tags": list(self.spec.tags),
                "policy_steps": g["policy_steps"],
                "effective_steps": g["effective_steps"],
                "keep_avg_all": (g["keep_sum_all"] / max(1, g["policy_steps"])),
                "keep_avg_eff": (g["keep_sum_eff"] / max(1, g["effective_steps"])),
                "prune_avg_all": (g["prune_sum_all"] / max(1, g["policy_steps"])),
                "prune_avg_eff": (g["prune_sum_eff"] / max(1, g["effective_steps"])),
                "quant_ratio_avg_all": (g["qratio_sum_all"] / max(1, g["policy_steps"])),
                "quant_ratio_avg_eff": (g["qratio_sum_eff"] / max(1, g["effective_steps"])),
                "avg_prune_keep": (g["prune_sum_eff"] / max(1, g["effective_steps"])),
                "avg_quant_ratio": (g["qratio_sum_eff"] / max(1, g["effective_steps"])),
                "action_hist": g["action_hist"],
                "action_probs": [c / total_actions for c in g["action_hist"]],
            },
            "per_request": self.per_request_stats,
        }

    def max_length(self):
        return int(getattr(unwrap(self.model).config, "max_position_embeddings", 4096))

    def max_gen_toks(self):
        return 256

    def batch_size(self):
        return self._max_batch

    @property
    def eot_token_id(self):
        return self._eos

    def tok_encode(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def tok_decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def loglikelihood(self, requests):
        """
        Each request: (context, continuation)
        Return: list of (sum_logprobs, is_greedy)
        """
        out = []
        for req in tqdm(requests, desc="loglikelihood", total=len(requests)):
            ctx, cont = req.args
            ctx_ids  = self.tok.encode(ctx, add_special_tokens=False)
            cont_ids = self.tok.encode(cont, add_special_tokens=False)
            max_ctx = self.max_length() - max(2, len(cont_ids)) - 4
            if len(ctx_ids) > max_ctx:
                ctx_ids = ctx_ids[-max_ctx:]

            sum_lp, is_greedy, s = self.runner.score_continuation_with_policy(
                ctx_ids=ctx_ids,
                cont_ids=cont_ids,
                greedy_actions=True,
                policy_temperature=self.runner.pi_temperature,
            )
            self._record_request_stats(req, s)
            out.append((sum_lp, is_greedy))
        return out

    def loglikelihood_rolling(self, requests):
        out = []
        for req in tqdm(requests, desc="loglikelihood_rolling", total=len(requests)):
            (text,) = req.args if isinstance(req.args, (list, tuple)) else (req.args,)
            ids = self.tok.encode(text, add_special_tokens=False)
            if len(ids) <= 1:
                out.append(0.0)
                continue
            split = min(64, max(1, len(ids)//20))
            ctx_ids, cont_ids = ids[:split], ids[split:]
            max_ctx = self.max_length() - max(2, len(cont_ids)) - 4
            if len(ctx_ids) > max_ctx:
                ctx_ids = ctx_ids[-max_ctx:]
            sum_lp, _ = self.runner.score_continuation_with_policy(ctx_ids, cont_ids)
            out.append(sum_lp)
        return out

    def generate_until(self, requests):
        """
        Each request: (context, gen_kwargs_dict)
        Return: list[str] of generations
        """
        gens = []
        for req in tqdm(requests, desc="generate_until", total=len(requests)):
            args = req.args if isinstance(req.args, (list, tuple)) else (req.args,)
            if len(args) == 2 and isinstance(args[1], dict):
                ctx, gen_kwargs = args
            else:
                ctx, until_list = args
                gen_kwargs = {"until": until_list}

            until = gen_kwargs.get("until", None)
            max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks()))
            temperature = float(gen_kwargs.get("temperature", 0.0))
            top_p = gen_kwargs.get("top_p", None)
            top_k = gen_kwargs.get("top_k", None)

            ctx_ids = self.tok.encode(ctx, add_special_tokens=False)
            max_ctx = self.max_length() - 128
            if len(ctx_ids) > max_ctx:
                ctx_ids = ctx_ids[-max_ctx:]
            gen_ids, s = self.runner.generate_with_policy(
                ctx_ids=ctx_ids,
                max_new_tokens=max_new,
                until=until,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                return_stats=True,
            )
            self._record_request_stats(req, s)
            gens.append(self.tok.decode(gen_ids, skip_special_tokens=True))
        return gens




# ---- NEW: LM-Eval wrapper for deterministic fixed baseline ----
from policy_runtime import FixedLMRunner

class FixedHarnessLM(LM):
    """
    LM-Eval-compatible wrapper that uses FixedLMRunner (no policy) to score/generate
    with deterministic matched κ/ρ/bits.
    """
    SUPPORTED_TASKS = None
    REQ_CHUNK_SIZE = 1

    def __init__(
        self,
        ckpt_dir: str,
        mode: str = "latest",
        target_keep_effective: Optional[float] = None,
        target_prune_keep: float = 1.0,
        target_quant_ratio: float = 1.0,
        struct_on_non_eff: bool = False,
        episode_len: Optional[int] = None,
        dense_refresh_tail: Optional[int] = None,
        max_batch: int = 4,
    ):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = find_latest_ckpt(ckpt_dir, mode)
        self.cfg = load_cfg_from_checkpoint_or_yaml(ckpt_dir, self.ckpt_path)
        self.tok, self.model = load_lm_and_tokenizer(self.cfg)

        if target_keep_effective is None:
            target_keep_effective = float(getattr(self.cfg, "C_target",
                                         getattr(self.cfg, "keep_target", 1.0)))

        self.runner = FixedLMRunner(
            cfg=self.cfg,
            model=self.model,
            tokenizer=self.tok,
            target_keep_effective=float(target_keep_effective),
            target_prune_keep=float(target_prune_keep),
            target_quant_ratio=float(target_quant_ratio),
            struct_on_non_eff=bool(struct_on_non_eff),
            episode_len=episode_len if episode_len is not None else int(getattr(self.cfg, "rollout_len", 16)),
            dense_refresh_tail=dense_refresh_tail if dense_refresh_tail is not None else int(getattr(self.cfg, "Ts",0) + getattr(self.cfg, "Tw",0) + 1),
        )

        self._max_batch = max_batch
        self.vocab_size = unwrap(self.model).get_output_embeddings().weight.size(0)
        self._eos = self.tok.eos_token_id

        self.per_request_stats = []
        self._agg = {
            "policy_steps": 0,
            "effective_steps": 0,
            "keep_sum_all": 0.0,
            "keep_sum_eff": 0.0,
            "prune_sum_all": 0.0,
            "prune_sum_eff": 0.0,
            "qratio_sum_all": 0.0,
            "qratio_sum_eff": 0.0,
            "action_hist": [0] * int(self.runner.spec.n_actions),
        }

    def _record_request_stats(self, req, stats: dict):
        task = getattr(req, "task_name", "unknown")
        index = getattr(req, "index", None)
        doc = getattr(req, "doc", None)
        doc_id = None
        if isinstance(doc, dict):
            for k in ("id", "doc_id", "sample_id", "query_id"):
                if k in doc: doc_id = doc[k]; break
        entry = {
            "task": task,
            "index": index,
            "doc_id": doc_id,
            "keep_fracs": list(self.runner.spec.token_keep),
            "action_tags": list(self.runner.spec.tags),
            **stats,
        }
        self.per_request_stats.append(entry)
        self._agg["policy_steps"] += stats.get("policy_steps", 0)
        self._agg["effective_steps"] += stats.get("effective_steps", 0)
        self._agg["keep_sum_all"] += stats.get("keep_sum_all", 0.0)
        self._agg["keep_sum_eff"] += stats.get("keep_sum_eff", 0.0)
        self._agg["prune_sum_all"] += stats.get("prune_sum_all", 0.0)
        self._agg["prune_sum_eff"] += stats.get("prune_sum_eff", 0.0)
        self._agg["qratio_sum_all"] += stats.get("qratio_sum_all", 0.0)
        self._agg["qratio_sum_eff"] += stats.get("qratio_sum_eff", 0.0)
        ah = stats.get("action_hist", [])
        for i, c in enumerate(ah):
            self._agg["action_hist"][i] += int(c)

    def export_sparsity_stats(self):
        g = self._agg
        total_actions = max(1, sum(g["action_hist"]))
        return {
            "global": {
                "keep_fracs": list(self.runner.spec.token_keep),
                "action_tags": list(self.runner.spec.tags),
                "policy_steps": g["policy_steps"],
                "effective_steps": g["effective_steps"],
                "keep_avg_all": (g["keep_sum_all"] / max(1, g["policy_steps"])),
                "keep_avg_eff": (g["keep_sum_eff"] / max(1, g["effective_steps"])),
                "prune_avg_all": (g["prune_sum_all"] / max(1, g["policy_steps"])),
                "prune_avg_eff": (g["prune_sum_eff"] / max(1, g["effective_steps"])),
                "quant_ratio_avg_all": (g["qratio_sum_all"] / max(1, g["policy_steps"])),
                "quant_ratio_avg_eff": (g["qratio_sum_eff"] / max(1, g["effective_steps"])),
                "avg_prune_keep": (g["prune_sum_eff"] / max(1, g["effective_steps"])),
                "avg_quant_ratio": (g["qratio_sum_eff"] / max(1, g["effective_steps"])),
                "action_hist": g["action_hist"],
                "action_probs": [c / total_actions for c in g["action_hist"]],
            },
            "per_request": self.per_request_stats,
        }

    def max_length(self):
        return int(getattr(unwrap(self.model).config, "max_position_embeddings", 4096))

    def max_gen_toks(self):
        return 256

    def batch_size(self):
        return self._max_batch

    @property
    def eot_token_id(self):
        return self._eos

    def tok_encode(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def tok_decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def loglikelihood(self, requests):
        """
        Micro-batch the fixed scorer so we can mix κ/ρ/q *across* a batch per step.
        This lets short continuations (1–3 tokens) still meet the target on average
        across the batch, not just over time within one sample.
        """
        out = []
        N = len(requests)
        i = 0
        pbar = tqdm(total=N, desc="loglikelihood (fixed,microbatch)")
        while i < N:
            B = min(self._max_batch, N - i)
            chunk = requests[i : i + B]

            ctx_list = []
            cont_list = []
            for req in chunk:
                ctx, cont = req.args
                ctx_ids  = self.tok.encode(ctx, add_special_tokens=False)
                cont_ids = self.tok.encode(cont, add_special_tokens=False)
                max_ctx = self.max_length() - max(2, len(cont_ids)) - 4
                if len(ctx_ids) > max_ctx:
                    ctx_ids = ctx_ids[-max_ctx:]
                ctx_list.append(ctx_ids)
                cont_list.append(cont_ids)

            # NEW: batch-mixed fixed scorer (per-step mixing across the B requests)
            lp_list, greedy_list, stats_list = self.runner.score_continuation_fixed_batch(
                ctx_list, cont_list
            )

            for j, req in enumerate(chunk):
                self._record_request_stats(req, stats_list[j])
                out.append((lp_list[j], greedy_list[j]))

            i += B
            pbar.update(B)
        pbar.close()
        return out

    def loglikelihood_rolling(self, requests):
        out = []
        for req in tqdm(requests, desc="loglikelihood_rolling", total=len(requests)):
            (text,) = req.args if isinstance(req.args, (list, tuple)) else (req.args,)
            ids = self.tok.encode(text, add_special_tokens=False)
            if len(ids) <= 1:
                out.append(0.0)
                continue
            split = min(64, max(1, len(ids)//20))
            ctx_ids, cont_ids = ids[:split], ids[split:]
            max_ctx = self.max_length() - max(2, len(cont_ids)) - 4
            if len(ctx_ids) > max_ctx:
                ctx_ids = ctx_ids[-max_ctx:]
            sum_lp, _ = self.runner.score_continuation_fixed(ctx_ids, cont_ids)
            out.append(sum_lp)
        return out

    def generate_until(self, requests):
        gens = []
        for req in tqdm(requests, desc="generate_until", total=len(requests)):
            args = req.args if isinstance(req.args, (list, tuple)) else (req.args,)
            if len(args) == 2 and isinstance(args[1], dict):
                ctx, gen_kwargs = args
            else:
                ctx, until_list = args
                gen_kwargs = {"until": until_list}
            until = gen_kwargs.get("until", None)
            max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks()))
            temperature = float(gen_kwargs.get("temperature", 0.0))
            top_p = gen_kwargs.get("top_p", None)
            top_k = gen_kwargs.get("top_k", None)

            ctx_ids = self.tok.encode(ctx, add_special_tokens=False)
            max_ctx = self.max_length() - 128
            if len(ctx_ids) > max_ctx:
                ctx_ids = ctx_ids[-max_ctx:]
            gen_ids, s = self.runner.generate_fixed(
                ctx_ids=ctx_ids,
                max_new_tokens=max_new,
                until=until,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                return_stats=True,
            )
            self._record_request_stats(req, s)
            gens.append(self.tok.decode(gen_ids, skip_special_tokens=True))
        return gens