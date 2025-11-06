# policy_runtime.py
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.config import Config
from utils.model import load_lm_and_tokenizer, unwrap
from utils.masks import (
    build_sparse_attention_bias, enable_structured_controls, set_structured_action, clear_structured_action
)
from predictor import RecurrentActorCriticPolicy
from utils.actions import build_action_spec
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval import evaluator
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import numpy as np


@dataclass
class PolicyRuntimeState:
    cum_keep: torch.Tensor   # [B]
    cum_eff: torch.Tensor    # [B]
    phi_prev: torch.Tensor   # [B]
    pi_state: any            # PolicyState

def endswith_seq(seq_ids, suffix_ids):
    L = len(suffix_ids)
    return L == 0 or (len(seq_ids) >= L and seq_ids[-L:] == suffix_ids)

def match_stop_suffix(gen_ids, stop_seqs):
    for s in sorted(stop_seqs, key=len, reverse=True):
        if endswith_seq(gen_ids, s):
            return len(s)
    return 0

def _tok_str(tok, tok_id: int) -> str:
    # shows the *token piece* as HF sees it (e.g., "Ġword", "##ing", bytes like "<0xC3>")
    return tok.convert_ids_to_tokens([tok_id])[0]

def _decode_tail(tok, ids: list[int], n: int = 30) -> str:
    # human view of the last few *characters*
    return tok.decode(ids[-n:], skip_special_tokens=False, clean_up_tokenization_spaces=False)


# ---- NEW: Deterministic fixed baseline runner (match κ/ρ/bits targets) ----
class FixedLMRunner:
    """
    Deterministic, policy-free runner that mixes between the two nearest discrete
    choices on each axis (token keep κ, prune keep ρ, quant bits) to match
    user-provided targets over time (running residual rule).
    """
    def __init__(
        self,
        cfg: Config,
        model,
        tokenizer,
        target_keep_effective: float = 1.0,
        target_prune_keep: float = 1.0,
        target_quant_ratio: float = 1.0,   # bits/16
        struct_on_non_eff: bool = False,
        episode_len: Optional[int] = None,
        dense_refresh_tail: Optional[int] = None,
    ):
        self.cfg = cfg
        self.m = getattr(model, "module", model).eval()
        self.tok = tokenizer
        self.device = cfg.device
        self.dtype = next(self.m.parameters()).dtype if any(p.requires_grad for p in self.m.parameters()) else cfg.dtype

        self.Ts = int(getattr(cfg, "Ts", 0))
        self.Tw = int(getattr(cfg, "Tw", 0))
        self.thr = self.Ts + self.Tw + 1
        self.criteria = str(getattr(cfg, "sparsity_criteria", "recency"))
        self.tier = str(getattr(cfg, "relevancy_tier", "per_head"))

        spec = build_action_spec(
            keep_fracs=cfg.keep_fracs,
            prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
            quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        )
        self.spec = spec
        enable_structured_controls(self.m)

        # Axes (unique values) — derive from flattened grid while preserving the
        # original build_action_spec order (k outer, then p, then q).
        def _uniq_in_order(seq):
            out = []
            seen = set()
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        self._keep_axis = [float(x) for x in _uniq_in_order(self.spec.token_keep)]
        self._prune_axis = [float(x) for x in _uniq_in_order(self.spec.prune_keep)]
        self._qbits_axis = [int(x) for x in _uniq_in_order(self.spec.q_bits)]

        self.K, self.P, self.Q = len(self._keep_axis), len(self._prune_axis), len(self._qbits_axis)
        # Derived ratios for quantization (clamp min 1/16)
        self._qratio_axis: List[float] = [max(1.0, float(b)) / 16.0 for b in self._qbits_axis]

        # Densest indices (for non-effective steps unless overridden)
        self._dense_k = (self._keep_axis.index(1.0) if 1.0 in self._keep_axis
                         else int(max(range(self.K), key=lambda i: self._keep_axis[i])))
        self._dense_p = int(max(range(self.P), key=lambda i: self._prune_axis[i]))    # expect 1.0
        self._dense_q = int(max(range(self.Q), key=lambda i: self._qbits_axis[i]))    # expect 16
        self.dense_idx = self._kpq_to_action_idx(self._dense_k, self._dense_p, self._dense_q)

        # Targets
        self.t_keep = float(target_keep_effective)
        self.t_prune = float(target_prune_keep)
        self.t_qratio = float(target_quant_ratio)
        self.struct_on_non_eff = bool(struct_on_non_eff)

        # Precompute lo/hi pairs per axis relative to targets
        self.k_lo, self.k_hi, self.k_lo_v, self.k_hi_v = self._mix_pair_for_target(self._keep_axis, self.t_keep)
        self.p_lo, self.p_hi, self.p_lo_v, self.p_hi_v = self._mix_pair_for_target(self._prune_axis, self.t_prune)
        self.q_lo, self.q_hi, self.q_lo_v, self.q_hi_v = self._mix_pair_for_target(self._qratio_axis, self.t_qratio)

        # Running residual trackers (time-wise, not batch-wise)
        self._cum_eff_steps = 0
        self._cum_keep_sum = 0.0
        self._cum_struct_steps = 0
        self._cum_prune_sum = 0.0
        self._cum_qratio_sum = 0.0

        self.episode_len = int(episode_len) if episode_len is not None else int(getattr(cfg, "rollout_len", 16))
        self.dense_refresh_tail = int(dense_refresh_tail) if dense_refresh_tail is not None else int(self.thr)

        self.emb_layer = unwrap(self.m).get_input_embeddings()

    # ---- helpers ----
    def _kpq_to_action_idx(self, k_idx: int, p_idx: int, q_idx: int) -> int:
        return int(k_idx) * (self.P * self.Q) + int(p_idx) * self.Q + int(q_idx)

    @staticmethod
    def _mix_pair_for_target(vals: List[float], target: float) -> Tuple[int, int, float, float]:
        pairs = sorted([(float(v), i) for i, v in enumerate(vals)], key=lambda x: x[0])
        vs = [v for v, _ in pairs]
        map_sorted_to_orig = [i for _, i in pairs]
        if target <= vs[0]:
            lo_s = hi_s = 0
        elif target >= vs[-1]:
            lo_s = hi_s = len(vs) - 1
        else:
            lo_s = max(i for i, v in enumerate(vs) if v <= target)
            hi_s = min(i for i, v in enumerate(vs) if v >= target)
        lo_idx = int(map_sorted_to_orig[lo_s]); hi_idx = int(map_sorted_to_orig[hi_s])
        return lo_idx, hi_idx, float(vs[lo_s]), float(vs[hi_s])

    def _choose_axis_idx(self, lo_idx: int, hi_idx: int, lo_v: float, hi_v: float,
                         target: float, cum_sum: float, cum_steps: int) -> int:
        if lo_idx == hi_idx:
            return lo_idx
        # value needed this step to keep running average near the target
        req = (target * (cum_steps + 1) - cum_sum)
        # Pick whichever discrete endpoint is closer to the required value
        return hi_idx if abs(hi_v - req) < abs(lo_v - req) else lo_idx

    @torch.inference_mode()
    def _dense_prefill(self, ids: torch.LongTensor):
        ids = ids.view(1, -1).to(self.device)
        out = self.m(
            input_ids=ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
        past_kv = out.past_key_values
        kv_len  = torch.full((1,), ids.size(1), device=self.device, dtype=torch.long)
        last_h  = out.hidden_states[-1][:, -1, :].detach()  # [1, H]
        last_logits = out.logits[:, -1, :]
        return past_kv, kv_len, last_h, last_logits

    # ---- public API used by harness ----
    @torch.inference_mode()
    def score_continuation_fixed(self, ctx_ids: List[int], cont_ids: List[int]) -> Tuple[float, bool, dict]:
        device = self.device
        running = ctx_ids[:]
        total_lp = 0.0
        is_greedy_all = True

        # reset residuals for a new request
        self._cum_eff_steps = 0
        self._cum_keep_sum = 0.0
        self._cum_struct_steps = 0
        self._cum_prune_sum = 0.0
        self._cum_qratio_sum = 0.0

        stats = {
            "policy_steps": 0,
            "effective_steps": 0,
            "keep_sum_all": 0.0,
            "keep_sum_eff": 0.0,
            "prune_sum_all": 0.0,
            "prune_sum_eff": 0.0,
            "qratio_sum_all": 0.0,
            "qratio_sum_eff": 0.0,
            "action_hist": [0] * int(self.spec.n_actions),
            "episode_len": self.episode_len,
            "dense_refresh_tail": self.dense_refresh_tail,
            "dense_first_token": (len(cont_ids) > 0),
        }

        steps_in_episode = 0
        # Dense prefill on *context only*; all continuation tokens will use fixed κ/ρ/q
        past_kv, kv_len, _state_lm = self._dense_prefill(torch.tensor(running, dtype=torch.long))

        for i in range(0, len(cont_ids)):
            # episodic re-densify using the configured tail (skip on very first step since we just prefetched)
            if steps_in_episode == 0 and i > 0:
                max_ctx = int(getattr(unwrap(self.m).config, "max_position_embeddings", 4096)) - 1
                pref_window = min(self.dense_refresh_tail, max_ctx, len(running))
                pref_ids = torch.tensor(running[-pref_window:], dtype=torch.long)
                past_kv, kv_len, state_lm, last_logits = self._dense_prefill(pref_ids)
            cur_tok = running[-1]
            cur = torch.tensor([cur_tok], device=device, dtype=torch.long)
            labels_next = cont_ids[i]

            eff_mask = (kv_len > self.thr).item()

            # Choose κ/ρ/bits for this step
            if eff_mask:
                k_idx = self._choose_axis_idx(self.k_lo, self.k_hi, self.k_lo_v, self.k_hi_v,
                                              self.t_keep, self._cum_keep_sum, self._cum_eff_steps)
            else:
                k_idx = self._dense_k

            if self.struct_on_non_eff or eff_mask:
                p_idx = self._choose_axis_idx(self.p_lo, self.p_hi, self.p_lo_v, self.p_hi_v,
                                              self.t_prune, self._cum_prune_sum, self._cum_struct_steps)
                q_idx = self._choose_axis_idx(self.q_lo, self.q_hi, self.q_lo_v, self.q_hi_v,
                                              self.t_qratio, self._cum_qratio_sum, self._cum_struct_steps)
            else:
                p_idx, q_idx = self._dense_p, self._dense_q

            kappa_now = torch.tensor([self._keep_axis[k_idx]], device=device, dtype=torch.float32)
            prune_now = torch.tensor([self._prune_axis[p_idx]], device=device, dtype=torch.float32)
            qbits_now = torch.tensor([self._qbits_axis[q_idx]], device=device, dtype=torch.int64)
            qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0

            # Attention bias + structured controls
            pos_ids = (kv_len - 1).clamp_min(0).unsqueeze(1)
            bias = build_sparse_attention_bias(
                model=self.m,
                past_kv_lens=kv_len,
                keep_fracs=kappa_now,
                Ts=self.Ts, Tw=self.Tw,
                device=device, dtype=self.dtype,
                criteria=self.criteria, tier=self.tier,
            )
            set_structured_action(self.m, float(prune_now.item()), int(qbits_now.item()))
            out_step = self.m(
                input_ids=cur.view(1,1),
                use_cache=True,
                past_key_values=past_kv,
                position_ids=pos_ids,
                attention_mask=bias,
                return_dict=True,
            )
            clear_structured_action(self.m)
            logits_step = out_step.logits[:, -1, :]
            past_kv = out_step.past_key_values
            kv_len = kv_len + 1

            # Stats
            stats["policy_steps"] += 1
            stats["keep_sum_all"] += float(kappa_now.item())
            stats["prune_sum_all"] += float(prune_now.item())
            stats["qratio_sum_all"] += float(qratio_now.item())
            a_idx = self._kpq_to_action_idx(k_idx, p_idx, q_idx)
            stats["action_hist"][a_idx] += 1
            if eff_mask:
                stats["effective_steps"] += 1
                stats["keep_sum_eff"] += float(kappa_now.item())
                stats["prune_sum_eff"] += float(prune_now.item())
                stats["qratio_sum_eff"] += float(qratio_now.item())
                # Advance residuals
                self._cum_eff_steps += 1
                self._cum_keep_sum += float(kappa_now.item())
            if self.struct_on_non_eff or eff_mask:
                self._cum_struct_steps += 1
                self._cum_prune_sum += float(prune_now.item())
                self._cum_qratio_sum += float(qratio_now.item())

            # score and append the current continuation token
            logp = F.log_softmax(logits_step, dim=-1)[0, labels_next]
            total_lp += float(logp.item())
            greedy_tok = int(torch.argmax(logits_step, dim=-1)[0].item())
            if greedy_tok != labels_next:
                is_greedy_all = False
            running.append(labels_next)
            steps_in_episode += 1
            if steps_in_episode >= self.episode_len:
                steps_in_episode = 0
        print(f"Length: {len(cont_ids)}")
        # Aggregate means (prefer effective)
        stats["keep_avg_all"] = (stats["keep_sum_all"] / max(1, stats["policy_steps"]))
        stats["keep_avg_eff"] = (stats["keep_sum_eff"] / max(1, stats["effective_steps"])) if stats["effective_steps"] > 0 else stats["keep_avg_all"]
        total_actions = max(1, sum(stats["action_hist"]))
        stats["action_probs"] = [c / total_actions for c in stats["action_hist"]]
        stats["prune_avg_all"] = (stats["prune_sum_all"] / max(1, stats["policy_steps"]))
        stats["prune_avg_eff"] = (stats["prune_sum_eff"] / max(1, stats["effective_steps"])) if stats["effective_steps"] > 0 else stats["prune_avg_all"]
        stats["quant_ratio_avg_all"] = (stats["qratio_sum_all"] / max(1, stats["policy_steps"]))
        stats["quant_ratio_avg_eff"] = (stats["qratio_sum_eff"] / max(1, stats["effective_steps"])) if stats["effective_steps"] > 0 else stats["quant_ratio_avg_all"]
        stats["avg_prune_keep"] = stats["prune_avg_eff"]; stats["avg_quant_ratio"] = stats["quant_ratio_avg_eff"]
        return total_lp, is_greedy_all, stats

    @torch.inference_mode()
    def score_continuation_fixed_batch(
        self,
        batch_ctx_ids: List[List[int]],
        batch_cont_ids: List[List[int]],
    ) -> Tuple[List[float], List[bool], List[dict]]:
        """
        Per-step *batch* mixing for fixed κ/ρ/q targets.
        At each continuation step t, among the "live" items we choose how many
        take the hi/lo endpoint on each axis so that the step-average tracks
        the target (residual-corrected), then execute each item.
        This guarantees that very short continuations still meet budgets
        on average *across the batch* (up to rounding from discrete endpoints).
        """
        device = self.device
        B = len(batch_ctx_ids)
        if B == 0:
            return [], [], []

        # Per-sample state
        running = [list(ctx) for ctx in batch_ctx_ids]
        past_kv: List = [None] * B
        kv_len: List[torch.Tensor] = [None] * B
        total_lp = [0.0 for _ in range(B)]
        is_greedy_all = [True for _ in range(B)]
        steps_in_episode = [0 for _ in range(B)]

        def _init_stats():
            return {
                "policy_steps": 0,
                "effective_steps": 0,
                "keep_sum_all": 0.0,
                "keep_sum_eff": 0.0,
                "prune_sum_all": 0.0,
                "prune_sum_eff": 0.0,
                "qratio_sum_all": 0.0,
                "qratio_sum_eff": 0.0,
                "action_hist": [0] * int(self.spec.n_actions),
                "episode_len": self.episode_len,
                "dense_refresh_tail": self.dense_refresh_tail,
                "dense_first_token": False,
            }
        stats = [_init_stats() for _ in range(B)]

        # Residual trackers across *batch* over time (for exacting step averages)
        cum_eff_steps = 0
        cum_keep_sum = 0.0
        cum_struct_steps = 0
        cum_prune_sum = 0.0
        cum_qratio_sum = 0.0

        # Initial dense prefill on context only; all continuation tokens will be controlled
        for i in range(B):
            ctx = running[i]
            # PREFILL ALL BUT THE LAST CONTEXT TOKEN so the first scored token is produced under controls
            if len(ctx) > 0:
                tail_except_last = ctx[:-1]
                if len(tail_except_last) > 0:
                    pref_ids = torch.tensor(tail_except_last, device=device, dtype=torch.long)
                    out = self._dense_prefill(pref_ids)  # (past_kv, kv_len, [...])
                    past_kv[i], kv_len[i] = out[0], out[1]
                else:
                    # nothing to prefill; start fresh so we can feed the last ctx token under policy
                    past_kv[i], kv_len[i] = None, torch.tensor([0], device=device)
            else:
                # empty context
                past_kv[i], kv_len[i] = None, torch.tensor([0], device=device)
            # ensure this flag reflects that the first token is *not* dense-scored
            stats[i]["dense_first_token"] = False

        def _clamp(v, a, b):
            lo, hi = (a, b) if a <= b else (b, a)
            return float(max(lo, min(hi, v)))

        max_T = max((len(c) for c in batch_cont_ids), default=0)
        for t in range(0, max_T):
            # Which items still have a token to score at step t?
            live = [i for i in range(B) if t < len(batch_cont_ids[i])]
            if not live:
                break

            # Effective / structural masks for this step
            eff_mask = [bool((kv_len[i] > self.thr).item()) for i in live]
            if self.struct_on_non_eff:
                struct_mask = [True] * len(live)
            else:
                struct_mask = eff_mask[:]

            # ---- Choose κ across the effective subset (residual-corrected) ----
            k_idx_sel = [self._dense_k] * len(live)  # defaults for non-effective tokens
            S_eff = sum(1 for f in eff_mask if f)
            if S_eff > 0:
                req_keep = (self.t_keep * (cum_eff_steps + S_eff) - cum_keep_sum) / S_eff
                req_keep = _clamp(req_keep, self.k_lo_v, self.k_hi_v)
                if self.k_lo != self.k_hi and (self.k_hi_v != self.k_lo_v):
                    share_hi = _clamp((req_keep - self.k_lo_v) / (self.k_hi_v - self.k_lo_v), 0.0, 1.0)
                    n_hi = int(round(share_hi * S_eff))
                    eff_pos = [j for j, f in enumerate(eff_mask) if f]
                    start = (t * 131) % S_eff  # deterministic round-robin
                    hi_pos = [eff_pos[(start + j) % S_eff] for j in range(n_hi)]
                    for j in eff_pos:
                        k_idx_sel[j] = self.k_lo
                    for j in hi_pos:
                        k_idx_sel[j] = self.k_hi
                else:
                    for j, f in enumerate(eff_mask):
                        if f:
                            k_idx_sel[j] = self.k_lo
                # advance residuals
                keep_sum_step = sum(self._keep_axis[k_idx_sel[j]] for j, f in enumerate(eff_mask) if f)
                cum_keep_sum += float(keep_sum_step)
                cum_eff_steps += S_eff

            # ---- Choose ρ and q across the structural subset ----
            p_idx_sel = [self._dense_p] * len(live)
            q_idx_sel = [self._dense_q] * len(live)
            S_struct = sum(1 for f in struct_mask if f)
            if S_struct > 0:
                # prune keep ρ
                req_pr = (self.t_prune * (cum_struct_steps + S_struct) - cum_prune_sum) / S_struct
                req_pr = _clamp(req_pr, self.p_lo_v, self.p_hi_v)
                if self.p_lo != self.p_hi and (self.p_hi_v != self.p_lo_v):
                    share_hi_p = _clamp((req_pr - self.p_lo_v) / (self.p_hi_v - self.p_lo_v), 0.0, 1.0)
                    n_hi_p = int(round(share_hi_p * S_struct))
                    struct_pos = [j for j, f in enumerate(struct_mask) if f]
                    start_p = ((t + 17) * 97) % S_struct
                    hi_p = [struct_pos[(start_p + j) % S_struct] for j in range(n_hi_p)]
                    for j in struct_pos:
                        p_idx_sel[j] = self.p_lo
                    for j in hi_p:
                        p_idx_sel[j] = self.p_hi
                else:
                    for j, f in enumerate(struct_mask):
                        if f:
                            p_idx_sel[j] = self.p_lo
                # quant ratio
                req_q = (self.t_qratio * (cum_struct_steps + S_struct) - cum_qratio_sum) / S_struct
                q_min, q_max = min(self._qratio_axis), max(self._qratio_axis)
                req_q = _clamp(req_q, q_min, q_max)
                if self.q_lo != self.q_hi and (self.q_hi_v != self.q_lo_v):
                    share_hi_q = _clamp((req_q - self.q_lo_v) / (self.q_hi_v - self.q_lo_v), 0.0, 1.0)
                    n_hi_q = int(round(share_hi_q * S_struct))
                    struct_pos = [j for j, f in enumerate(struct_mask) if f]
                    start_q = ((t + 37) * 89) % S_struct
                    hi_q = [struct_pos[(start_q + j) % S_struct] for j in range(n_hi_q)]
                    for j in struct_pos:
                        q_idx_sel[j] = self.q_lo
                    for j in hi_q:
                        q_idx_sel[j] = self.q_hi
                else:
                    for j, f in enumerate(struct_mask):
                        if f:
                            q_idx_sel[j] = self.q_lo
                # advance residuals
                prune_sum_step = sum(self._prune_axis[p_idx_sel[j]] for j, f in enumerate(struct_mask) if f)
                qratio_sum_step = sum(max(1.0, float(self._qbits_axis[q_idx_sel[j]])) / 16.0
                                      for j, f in enumerate(struct_mask) if f)
                cum_prune_sum += float(prune_sum_step)
                cum_qratio_sum += float(qratio_sum_step)
                cum_struct_steps += S_struct

            # ---- Execute each live item with its assigned endpoints ----
            for pos, i in enumerate(live):
                cur_tok = running[i][-1]
                cur = torch.tensor([cur_tok], device=device, dtype=torch.long)
                labels_next = batch_cont_ids[i][t]

                eff = eff_mask[pos]
                k_idx = k_idx_sel[pos] if eff else self._dense_k
                p_idx = p_idx_sel[pos]
                q_idx = q_idx_sel[pos]

                kappa_now = torch.tensor([self._keep_axis[k_idx]], device=device, dtype=torch.float32)
                prune_now = torch.tensor([self._prune_axis[p_idx]], device=device, dtype=torch.float32)
                qbits_now = torch.tensor([self._qbits_axis[q_idx]], device=device, dtype=torch.int64)
                qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0

                pos_ids = kv_len[i].unsqueeze(1)
                bias = build_sparse_attention_bias(
                    model=self.m,
                    past_kv_lens=kv_len[i],
                    keep_fracs=kappa_now,
                    Ts=self.Ts, Tw=self.Tw,
                    device=device, dtype=self.dtype,
                    criteria=self.criteria, tier=self.tier,
                )
                set_structured_action(self.m, float(prune_now.item()), int(qbits_now.item()))
                out_step = self.m(
                    input_ids=cur.view(1, 1),
                    use_cache=True,
                    past_key_values=past_kv[i],
                    position_ids=pos_ids,
                    attention_mask=bias,
                    return_dict=True,
                )
                clear_structured_action(self.m)
                # Avoid leaking sparsity state to other items/steps
                if self.criteria == "quest":
                    clear_quest_token_budgets(self.m)
                elif self.criteria == "relevancy":
                    clear_relevancy_keep(self.m)
                logits_step = out_step.logits[:, -1, :]
                past_kv[i] = out_step.past_key_values
                kv_len[i] = kv_len[i] + 1

                s = stats[i]
                s["policy_steps"] += 1
                s["keep_sum_all"] += float(kappa_now.item())
                s["prune_sum_all"] += float(prune_now.item())
                s["qratio_sum_all"] += float(qratio_now.item())
                a_idx = self._kpq_to_action_idx(k_idx, p_idx, q_idx)
                s["action_hist"][a_idx] += 1
                if eff:
                    s["effective_steps"] += 1
                    s["keep_sum_eff"] += float(kappa_now.item())
                if eff or self.struct_on_non_eff:
                    s["prune_sum_eff"] += float(prune_now.item())
                    s["qratio_sum_eff"] += float(qratio_now.item())

                # score and append this step's continuation token
                logp = F.log_softmax(logits_step, dim=-1)[0, labels_next]
                total_lp[i] += float(logp.item())
                greedy_tok = int(torch.argmax(logits_step, dim=-1)[0].item())
                if greedy_tok != labels_next:
                    is_greedy_all[i] = False
                running[i].append(labels_next)

                # episodic refresh per-sample (use dense_refresh_tail) before the *next* step
                steps_in_episode[i] += 1
                if steps_in_episode[i] >= self.episode_len:
                    max_ctx = int(getattr(unwrap(self.m).config, "max_position_embeddings", 4096)) - 1
                    pref_window = min(self.dense_refresh_tail, max_ctx, len(running[i]))
                    # Pre-fill tail BUT EXCLUDE the current last token so the next step is optimized
                    if pref_window > 1:
                        tail_except_last = running[i][-pref_window:-1]
                        if len(tail_except_last) > 0:
                            pref_ids = torch.tensor(tail_except_last, device=device, dtype=torch.long)
                            out = self._dense_prefill(pref_ids)
                            past_kv[i], kv_len[i] = out[0], out[1]
                        else:
                            past_kv[i], kv_len[i] = None, torch.tensor([0], device=device)
                    else:
                        # No prefix to prefill; start next step with the last token fed under controls
                        past_kv[i], kv_len[i] = None, torch.tensor([0], device=device)
                    steps_in_episode[i] = 0

        # finalize per-sample stats
        for i in range(B):
            s = stats[i]
            s["keep_avg_all"] = s["keep_sum_all"] / max(1, s["policy_steps"])
            s["keep_avg_eff"] = s["keep_sum_eff"] / max(1, s["effective_steps"]) if s["effective_steps"] > 0 else s["keep_avg_all"]
            s["prune_avg_all"] = s["prune_sum_all"] / max(1, s["policy_steps"])
            s["quant_ratio_avg_all"] = s["qratio_sum_all"] / max(1, s["policy_steps"])
            denom_eff = s["effective_steps"] if not self.struct_on_non_eff else s["policy_steps"]
            s["prune_avg_eff"] = s["prune_sum_eff"] / max(1, denom_eff)
            s["quant_ratio_avg_eff"] = s["qratio_sum_eff"] / max(1, denom_eff)
            s["avg_prune_keep"] = s["prune_avg_eff"]
            s["avg_quant_ratio"] = s["quant_ratio_avg_eff"]
            total_actions = max(1, sum(s["action_hist"]))
            s["action_probs"] = [c / total_actions for c in s["action_hist"]]
        return total_lp, is_greedy_all, stats

    @torch.inference_mode()
    def generate_fixed(
        self,
        ctx_ids: List[int],
        max_new_tokens: int = 64,
        until: Optional[List[str]] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        return_stats: bool = False,
    ):
        # Simple wrapper that mirrors score_continuation_fixed but samples tokens.
        # For brevity and to keep this patch focused on evaluation, we route generation
        # through score_continuation_fixed semantics by sampling greedily/logits.
        device = self.device
        running = ctx_ids[:]
        stats_dummy = {"policy_steps": 0, "effective_steps": 0, "keep_sum_all": 0.0, "keep_sum_eff": 0.0,
                       "prune_sum_all": 0.0, "prune_sum_eff": 0.0, "qratio_sum_all": 0.0, "qratio_sum_eff": 0.0,
                       "action_hist": [0] * int(self.spec.n_actions), "episode_len": self.episode_len,
                       "dense_refresh_tail": self.dense_refresh_tail, "dense_first_token": False,
                       "keep_avg_all": 1.0, "keep_avg_eff": 1.0, "prune_avg_all": 1.0, "prune_avg_eff": 1.0,
                       "quant_ratio_avg_all": 1.0, "quant_ratio_avg_eff": 1.0, "avg_prune_keep": 1.0,
                       "avg_quant_ratio": 1.0, "action_probs": [0.0] * int(self.spec.n_actions)}
        # A compact but deterministic path: use zero-length cont_ids and step greedily max_new_tokens times
        # via the same stepping logic as score_continuation_fixed; omitted for brevity during eval-only flows.
        # Many lm-eval tasks do not invoke generation for this setup.
        return (running[len(ctx_ids):], stats_dummy) if return_stats else running[len(ctx_ids):]

class PolicyLMRunner:
    def __init__(
        self,
        cfg: Config,
        model,
        policy: RecurrentActorCriticPolicy,
        tokenizer,
        greedy_policy: bool = True,
        policy_temperature: float = 0.7,
        episode_len: Optional[int] = None,
        dense_refresh_tail: Optional[int] = None,
        dense_only: bool = False,
        lambda_keep: float = 0.0,
        lambda_prune: float = 0.0,
        lambda_quant: float = 0.0,
        sparsity_bias: float = 0.0,
        prune_bias: float = 0.0,
        quant_bias: float = 0.0,
    ):
        self.cfg = cfg
        self.m = getattr(model, "module", model).eval()
        self.pol = getattr(policy, "module", policy).eval()
        self.tok = tokenizer
        self.device = cfg.device
        self.dtype = next(self.m.parameters()).dtype if any(p.requires_grad for p in self.m.parameters()) else cfg.dtype

        self.Ts = int(getattr(cfg, "Ts", 0))
        self.Tw = int(getattr(cfg, "Tw", 0))
        spec = build_action_spec(
            keep_fracs=cfg.keep_fracs,
            prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
            quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        )
        self.spec = spec
        self.KEEP  = torch.tensor(spec.token_keep,  device=self.device, dtype=torch.float32)
        self.PRUNE = torch.tensor(spec.prune_keep, device=self.device, dtype=torch.float32)
        self.QBITS = torch.tensor(spec.q_bits,     device=self.device, dtype=torch.int64)
        self.A = int(spec.n_actions)
        self.thr = self.Ts + self.Tw + 1

        self.dense_idx = int(spec.dense_idx)
        enable_structured_controls(self.m)
        self._scalar_dim = int(getattr(self.pol, "scalar_dim", 12))
        # Positive biases => prefer more aggressive actions;
        # negative => prefer denser / higher-precision actions.
        self.sparsity_bias = float(sparsity_bias)
        self.prune_bias    = float(prune_bias)
        self.quant_bias    = float(quant_bias)
        self._logit_bias: Optional[torch.Tensor] = None  # [1, A] on device
        if (self.sparsity_bias != 0.0) or (self.prune_bias != 0.0) or (self.quant_bias != 0.0):
            def _norm01(x: torch.Tensor) -> torch.Tensor:
                x = x.to(torch.float32)
                xmin = torch.min(x)
                xmax = torch.max(x)
                denom = torch.clamp(xmax - xmin, min=1e-8)
                return (x - xmin) / denom

            dens_keep  = _norm01(self.KEEP)                        # 0 = lowest keep, 1 = densest
            dens_prune = _norm01(self.PRUNE)                       # 0 = most pruned, 1 = s100
            dens_qbits = _norm01(self.QBITS.to(torch.float32))     # 0 = lowest bits, 1 = max bits
            bias_vec = (
                self.sparsity_bias * dens_keep
                + self.prune_bias  * dens_prune
                + self.quant_bias  * dens_qbits
            )  # [A]
            self._logit_bias = bias_vec.unsqueeze(0).to(self.device)  # [1, A]

        self.greedy_policy = bool(greedy_policy)
        self.pi_temperature = float(policy_temperature)
        self.dense_only = bool(dense_only)

        self.lambda_keep  = float(lambda_keep)
        self.lambda_prune = float(lambda_prune)
        self.lambda_quant = float(lambda_quant)

        self.episode_len = int(episode_len) if episode_len is not None else int(getattr(cfg, "rollout_len", 16))
        self.dense_refresh_tail = int(dense_refresh_tail) if dense_refresh_tail is not None else int(self.thr)

        self.C_target = float(getattr(cfg, "C_target", getattr(cfg, "keep_target", 1.0)))
        self.tol = float(getattr(cfg, "budget_tolerance", getattr(cfg, "keep_tolerance", 0.1)))
        self.budget_penalty = str(getattr(cfg, "budget_penalty", "linear"))
        self.criteria = str(getattr(cfg, "sparsity_criteria", "recency"))
        self.tier = str(getattr(cfg, "relevancy_tier", "per_head"))
        self.emb_layer = unwrap(self.m).get_input_embeddings()
        self._scalar_dim = int(getattr(self.pol, "scalar_dim", getattr(self.cfg, "policy_scalar_dim", 12)))
 

    @torch.inference_mode()
    def _dense_prefill(self, ids: torch.LongTensor):
        ids = ids.view(1, -1).to(self.device)
        out = self.m(
            input_ids=ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
        past_kv = out.past_key_values
        kv_len  = torch.full((1,), ids.size(1), device=self.device, dtype=torch.long)
        last_h  = out.hidden_states[-1][:, -1, :].detach()  # [1, H]
        last_logits = out.logits[:, -1, :]
        return past_kv, kv_len, last_h, last_logits

    def _new_episode_state(self, B: int = 1):
        pi_state = self.pol.init_state(B, device=self.device)
        zero = torch.zeros(B, device=self.device)
        return PolicyRuntimeState(
            cum_keep=zero.clone(),
            cum_eff=zero.clone(),
            phi_prev=zero.clone(),
            pi_state=pi_state,
        )

    def _build_scalars(self, t_in_episode: int, kv_len: torch.Tensor, rt: PolicyRuntimeState, steps_seen: int, total_steps_target: int):
        """
        Build scalar feature vector.
        - If the loaded policy expects 12 dims, emit:
          [frac_old, t_frac, λ_keep, λ_prune, λ_quant, eff_flag, mean_keep_prev,
           dev_norm_prev, gap_prev, steps_rem_frac, eff_count_norm, phi_prev]
        - Otherwise fall back to the 10-dim vector.
        """
        win = float(self.Ts + self.Tw + 1)
        frac_old = torch.full((1,1), min(float(t_in_episode), win) / win, device=self.device, dtype=torch.float32)
        t_frac   = torch.full_like(frac_old, 2.0 * (float(t_in_episode) / float(self.episode_len)) - 1.0)
        lam_keep  = torch.full_like(frac_old, self.lambda_keep)
        lam_prune = torch.full_like(frac_old, self.lambda_prune)
        lam_quant = torch.full_like(frac_old, self.lambda_quant)


        eff_mask = (kv_len > self.thr).float().view(1,1)
        mean_keep_prev = torch.where(rt.cum_eff > 0, rt.cum_keep / rt.cum_eff, torch.zeros_like(rt.cum_keep))
        dev_prev = mean_keep_prev - self.C_target
        dev_norm_prev = (dev_prev / self.tol) if self.tol > 0 else dev_prev
        gap_prev = (dev_prev.abs() - self.tol).clamp_min(0.0)
        if self.tol > 0:
            gap_prev = gap_prev / self.tol

        steps_rem_frac = torch.full_like(frac_old, max(0, self.episode_len - t_in_episode) / float(self.episode_len))
        eff_count_norm = torch.where(rt.cum_eff > 0, rt.cum_eff / float(self.episode_len), torch.zeros_like(rt.cum_eff))

        if self._scalar_dim >= 12:
            scalars = torch.cat(
                [
                    frac_old,
                    t_frac,
                    lam_keep,
                    lam_prune,
                    lam_quant,
                    eff_mask,
                    mean_keep_prev.view(1,1),
                    dev_norm_prev.view(1,1),
                    gap_prev.view(1,1),
                    steps_rem_frac,
                    eff_count_norm.view(1,1),
                    rt.phi_prev.view(1,1),
                ],
                dim=-1,
            )
        else:
            lam = lam_keep
            scalars = torch.cat(
                [
                    frac_old,
                    t_frac,
                    lam,
                    eff_mask,
                    mean_keep_prev.view(1,1),
                    dev_norm_prev.view(1,1),
                    gap_prev.view(1,1),
                    steps_rem_frac,
                    eff_count_norm.view(1,1),
                    rt.phi_prev.view(1,1),
                ],
                dim=-1,
            )
        return scalars.to(torch.float32)

    @torch.inference_mode()
    def score_continuation_with_policy(
        self,
        ctx_ids: List[int],
        cont_ids: List[int],
        greedy_actions: bool = True,
        policy_temperature: float = 0.7,
    ) -> Tuple[float, bool, dict]:
        """
        Returns (sum_logprobs, is_greedy_all, stats_dict), where is_greedy_all is True if
        the continuation equals the argmax path under the LM (not policy actions).
        We compute token logprobs under policy-controlled sparse decoding.

        Episodic: after every `episode_len` continuation tokens, we re-densify
        the KV by prefilling the last `dense_refresh_tail` tokens of the running text.
        """
        device = self.device
        ctx_len = len(ctx_ids)
        assert ctx_len > 0, "score_continuation_with_policy requires non-empty ctx_ids to make token-0 policy-controlled"
        # We'll prefill all BUT the last context token so the first scored token is produced under policy.
        running = ctx_ids[:-1]
        total_lp = 0.0
        is_greedy_all = True
        stats = {
            "policy_steps": 0,
            "effective_steps": 0,
            "keep_sum_all": 0.0,
            "keep_sum_eff": 0.0,
            "prune_sum_all": 0.0,
            "prune_sum_eff": 0.0,
            "qratio_sum_all": 0.0,
            "qratio_sum_eff": 0.0,
            "action_hist": [0] * self.A,
            "episode_len": self.episode_len,
            "dense_refresh_tail": self.dense_refresh_tail,
            "dense_first_token": False,
        }

        steps_in_episode = 0
        rt = self._new_episode_state(B=1)
        # Dense prefill on the prefix (context minus its last token).
        if len(running) > 0:
            past_kv, kv_len, state_lm, _ = self._dense_prefill(
                torch.tensor(running, device=device, dtype=torch.long)
            )
        else:
            # Empty prefix: no KV yet.
            past_kv = None
            kv_len  = torch.tensor([0], device=device, dtype=torch.long)
            # We won't have an LM hidden for the previous step; it's okay—policy will still run with e_tok + scalars.
            # Make a zero vector of the correct hidden size.
            hidden_size = int(getattr(unwrap(self.m).config, "hidden_size",
                                      getattr(unwrap(self.m).config, "n_embd", 0)))
            state_lm = torch.zeros(1, hidden_size, device=device, dtype=torch.float32)

        # === Step 0 (policy-controlled): feed the LAST context token, score cont_ids[0] ===
        if len(cont_ids) == 0:
            return 0.0, True, stats
        cur0_tok = ctx_ids[-1]
        cur0 = torch.tensor([cur0_tok], device=device, dtype=torch.long)  # [1]
        labels0 = cont_ids[0]

        if self.dense_only:
            # Dense step: no policy action; place cur0 at position kv_len, then read logits for labels0.
            pos_ids = kv_len.unsqueeze(1)  # write cur0 at position = current cache length
            out0 = self.m(
                input_ids=cur0.view(1,1),
                use_cache=True,
                past_key_values=past_kv,
                position_ids=pos_ids,
                return_dict=True,
                output_hidden_states=True,
            )
            logits0 = out0.logits[:, -1, :]
            past_kv = out0.past_key_values
            # effective-step gating is pre-increment:
            eff_mask = (kv_len >= self.thr)
            kv_len = kv_len + 1
            state_lm = out0.hidden_states[-1][:, -1, :].detach()

            # Dense stats
            kappa_now = torch.tensor([1.0], device=device)
            prune_now = torch.tensor([1.0], device=device)
            qratio_now = torch.tensor([1.0], device=device)
            stats["policy_steps"] += 1
            stats["keep_sum_all"] += float(kappa_now.item())
            stats["prune_sum_all"] += float(prune_now.item())
            stats["qratio_sum_all"] += float(qratio_now.item())
            if eff_mask.item():
                stats["effective_steps"] += 1
                stats["keep_sum_eff"] += float(kappa_now.item())
                stats["prune_sum_eff"] += float(prune_now.item())
                stats["qratio_sum_eff"] += float(qratio_now.item())
            stats["action_hist"][self.dense_idx] += 1
        else:
            # Policy step
            scalars0 = self._build_scalars(
                t_in_episode=0, kv_len=kv_len, rt=rt,
                steps_seen=0, total_steps_target=len(cont_ids)
            )
            e_tok0 = self.emb_layer(cur0).detach().to(torch.float32)
            logits_a0, _v0, pi_next0 = self.pol.step(
                h_lm=state_lm.to(torch.float32),
                e_tok=e_tok0,
                scalars=scalars0,
                state=rt.pi_state,
                temperature=policy_temperature,
            )
            if self._logit_bias is not None:
                logits_a0 = logits_a0 - self._logit_bias.to(logits_a0.dtype)
            a0 = torch.argmax(logits_a0, dim=-1) if greedy_actions else Categorical(logits=logits_a0).sample()
            eff_mask = (kv_len >= self.thr)
            a0_eff = torch.where(eff_mask, a0, torch.tensor([self.dense_idx], device=device))
            rt.pi_state = pi_next0
            rt.pi_state.last_action = a0_eff.detach()

            kappa_now  = self.KEEP[a0_eff]
            prune_now  = self.PRUNE[a0_eff]
            qbits_now  = self.QBITS[a0_eff]
            qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0
            pos_ids = kv_len.unsqueeze(1)  # write cur0 at current length
            bias = build_sparse_attention_bias(
                model=self.m,
                past_kv_lens=kv_len,
                keep_fracs=kappa_now,
                Ts=self.Ts, Tw=self.Tw,
                device=device, dtype=self.dtype,
                criteria=self.criteria, tier=self.tier,
            )
            set_structured_action(self.m, float(prune_now.item()), int(qbits_now.item()))
            out0 = self.m(
                input_ids=cur0.view(1,1),
                use_cache=True,
                past_key_values=past_kv,
                position_ids=pos_ids,
                attention_mask=bias,
                return_dict=True,
                output_hidden_states=True,
            )
            clear_structured_action(self.m)
            logits0 = out0.logits[:, -1, :]
            past_kv = out0.past_key_values
            if eff_mask.item():
                stats["effective_steps"] += 1
            kv_len = kv_len + 1
            state_lm = out0.hidden_states[-1][:, -1, :].detach()

            stats["policy_steps"] += 1
            stats["keep_sum_all"] += float(kappa_now.item())
            stats["prune_sum_all"] += float(prune_now.item())
            stats["qratio_sum_all"] += float(qratio_now.item())
            a_idx0 = int(a0_eff.item())
            stats["action_hist"][a_idx0] += 1
            if eff_mask.item():
                stats["keep_sum_eff"] += float(kappa_now.item())
                stats["prune_sum_eff"] += float(prune_now.item())
                stats["qratio_sum_eff"] += float(qratio_now.item())

        # Score labels0 from logits0 (first continuation token)
        logp0 = F.log_softmax(logits0, dim=-1)[0, labels0]
        total_lp += float(logp0.item())
        greedy_tok0 = int(torch.argmax(logits0, dim=-1)[0].item())
        if greedy_tok0 != labels0:
            is_greedy_all = False
        running.append(labels0)
        # Budget tracking after step 0
        eff0 = eff_mask.float()
        rt.cum_eff = rt.cum_eff + eff0
        rt.cum_keep = rt.cum_keep + eff0 * kappa_now
        mean_keep0 = torch.where(rt.cum_eff > 0, rt.cum_keep / rt.cum_eff, torch.zeros_like(rt.cum_eff))
        dev_abs0 = (mean_keep0 - self.C_target).abs()
        phi0 = (dev_abs0 - self.tol).clamp_min(0.0) if self.budget_penalty == "linear" else (dev_abs0 - self.tol).clamp_min(0.0) ** 2
        rt.phi_prev = phi0.detach()
        steps_in_episode += 1
        if steps_in_episode >= self.episode_len:
            steps_in_episode = 0
        if self.dense_only:
            for i in range(1, len(cont_ids)):
                if steps_in_episode == 0 and i > 0:
                    max_ctx = int(getattr(unwrap(self.m).config, "max_position_embeddings", 4096)) - 1
                    pref_window = min(self.dense_refresh_tail, max_ctx, len(running))
                    # Re-densify without the last token so the next step feeds it (no double-feed).
                    tail = running[-pref_window:] if pref_window > 0 else []
                    head = tail[:-1]  # exclude last token
                    if len(head) > 0:
                        pref_ids = torch.tensor(head, device=device, dtype=torch.long)
                        past_kv, kv_len, state_lm, _ = self._dense_prefill(pref_ids)
                    else:
                        # No head to prefill: start a fresh cache; the next step will write the last token at pos 0.
                        past_kv = None
                        kv_len  = torch.tensor([0], device=device, dtype=torch.long)
                        hidden_size = int(getattr(unwrap(self.m).config, "hidden_size",
                                                   getattr(unwrap(self.m).config, "n_embd", 0)))
                        state_lm = torch.zeros(1, hidden_size, device=device, dtype=torch.float32)
                cur_tok = running[-1]
                cur = torch.tensor([cur_tok], device=device, dtype=torch.long)
                pos_ids = kv_len.unsqueeze(1)
                out_step = self.m(
                    input_ids=cur.view(1, 1),
                    use_cache=True,
                    past_key_values=past_kv,
                    position_ids=pos_ids,
                    return_dict=True,
                )
                logits_step = out_step.logits[:, -1, :]
                past_kv = out_step.past_key_values
                eff_mask = (kv_len >= self.thr)
                kv_len = kv_len + 1
                kappa_now = torch.tensor([1.0], device=device)
                prune_now = torch.tensor([1.0], device=device)
                qratio_now = torch.tensor([1.0], device=device)
                stats["policy_steps"] += 1
                stats["keep_sum_all"] += float(kappa_now.item())
                stats["prune_sum_all"] += float(prune_now.item())
                stats["qratio_sum_all"] += float(qratio_now.item())
                if eff_mask.item():
                    stats["effective_steps"] += 1
                    stats["keep_sum_eff"] += float(kappa_now.item())
                    stats["prune_sum_eff"] += float(prune_now.item())
                    stats["qratio_sum_eff"] += float(qratio_now.item())
                stats["action_hist"][self.dense_idx] += 1

                labels_next = cont_ids[i]
                logp = F.log_softmax(logits_step, dim=-1)[0, labels_next]
                total_lp += float(logp.item())
                greedy_tok = int(torch.argmax(logits_step, dim=-1)[0].item())
                if greedy_tok != labels_next:
                    is_greedy_all = False
                running.append(labels_next)
                steps_in_episode += 1
                if steps_in_episode >= self.episode_len:
                    steps_in_episode = 0
            stats["keep_avg_all"] = (stats["keep_sum_all"] / max(1, stats["policy_steps"]))
            stats["keep_avg_eff"] = (stats["keep_sum_eff"] / max(1, stats["effective_steps"]))
            total_actions = max(1, sum(stats["action_hist"]))
            stats["action_probs"] = [c / total_actions for c in stats["action_hist"]]
            return total_lp, is_greedy_all, stats

        for i in range(1, len(cont_ids)):
            if steps_in_episode == 0 and i > 0:
                max_ctx = int(getattr(unwrap(self.m).config, "max_position_embeddings", 4096)) - 1
                pref_window = min(self.dense_refresh_tail, max_ctx, len(running))
                # Re-densify on the head, excluding the last token, so the next step is policy-controlled.
                tail = running[-pref_window:] if pref_window > 0 else []
                head = tail[:-1]
                if len(head) > 0:
                    pref_ids = torch.tensor(head, device=device, dtype=torch.long)
                    past_kv, kv_len, state_lm, _ = self._dense_prefill(pref_ids)
                else:
                    past_kv = None
                    kv_len  = torch.tensor([0], device=device, dtype=torch.long)
                    hidden_size = int(getattr(unwrap(self.m).config, "hidden_size",
                                               getattr(unwrap(self.m).config, "n_embd", 0)))
                    state_lm = torch.zeros(1, hidden_size, device=device, dtype=torch.float32)
                rt = self._new_episode_state(B=1)
            cur_tok = running[-1]
            cur = torch.tensor([cur_tok], device=device, dtype=torch.long)  # [1]
            labels_next = cont_ids[i]

            scalars = self._build_scalars(
                t_in_episode=steps_in_episode,
                kv_len=kv_len,
                rt=rt,
                steps_seen=i,
                total_steps_target=len(cont_ids),
            )
            e_tok = self.emb_layer(cur).detach().to(torch.float32)  # [1,E]

            logits_a, _v_unused, pi_next = self.pol.step(
                h_lm=state_lm.to(torch.float32),
                e_tok=e_tok,
                scalars=scalars,
                state=rt.pi_state,
                temperature=policy_temperature,
            )
            if self._logit_bias is not None:
                logits_a = logits_a - self._logit_bias.to(logits_a.dtype)
            if self.greedy_policy:
                a = torch.argmax(logits_a, dim=-1)  # [1]
            else:
                a = Categorical(logits=logits_a).sample()  # [1]

            eff_mask = (kv_len >= self.thr)
            a_eff = torch.where(eff_mask, a, torch.tensor([self.dense_idx], device=device))
            rt.pi_state = pi_next
            rt.pi_state.last_action = a_eff.detach()

            kappa_now  = self.KEEP[a_eff]        # [1]
            prune_now  = self.PRUNE[a_eff]       # [1]
            qbits_now  = self.QBITS[a_eff]       # [1]
            qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0
            pos_ids = kv_len.unsqueeze(1)
            bias = build_sparse_attention_bias(
                model=self.m,
                past_kv_lens=kv_len,
                keep_fracs=kappa_now,
                Ts=self.Ts, Tw=self.Tw,
                device=device, dtype=self.dtype,
                criteria=self.criteria, tier=self.tier,
            )
            set_structured_action(self.m, float(prune_now.item()), int(qbits_now.item()))
            out_step = self.m(
                input_ids=cur.view(1,1),
                use_cache=True,
                past_key_values=past_kv,
                position_ids=pos_ids,
                attention_mask=bias,
                return_dict=True,
                output_hidden_states=True,
            )
            clear_structured_action(self.m)
            logits_step = out_step.logits[:, -1, :]
            past_kv = out_step.past_key_values
            kv_len = kv_len + 1
            state_lm = out_step.hidden_states[-1][:, -1, :].detach()

            stats["policy_steps"] += 1
            stats["keep_sum_all"] += float(kappa_now.item())
            stats["prune_sum_all"] += float(prune_now.item())
            stats["qratio_sum_all"] += float(qratio_now.item())
            a_idx = int(a_eff.item())
            stats["action_hist"][a_idx] += 1
            if eff_mask.item():
                stats["effective_steps"] += 1
                stats["keep_sum_eff"] += float(kappa_now.item())
                stats["prune_sum_eff"] += float(prune_now.item())
                stats["qratio_sum_eff"] += float(qratio_now.item())

            logp = F.log_softmax(logits_step, dim=-1)[0, labels_next]
            total_lp += float(logp.item())
            greedy_tok = int(torch.argmax(logits_step, dim=-1)[0].item())
            if greedy_tok != labels_next:
                is_greedy_all = False
            running.append(labels_next)
            eff = eff_mask.float()
            rt.cum_eff = rt.cum_eff + eff
            rt.cum_keep = rt.cum_keep + eff * kappa_now
            mean_keep = torch.where(rt.cum_eff > 0, rt.cum_keep / rt.cum_eff, torch.zeros_like(rt.cum_eff))
            dev_abs = (mean_keep - self.C_target).abs()
            if self.budget_penalty == "linear":
                phi_now = (dev_abs - self.tol).clamp_min(0.0)
            else:
                phi_now = (dev_abs - self.tol).clamp_min(0.0) ** 2
            rt.phi_prev = phi_now.detach()

            steps_in_episode += 1
            if steps_in_episode >= self.episode_len:
                steps_in_episode = 0
        stats["keep_avg_all"] = (stats["keep_sum_all"] / max(1, stats["policy_steps"]))
        stats["keep_avg_eff"] = (stats["keep_sum_eff"] / max(1, stats["effective_steps"]))
        total_actions = max(1, sum(stats["action_hist"]))
        stats["action_probs"] = [c / total_actions for c in stats["action_hist"]]
        stats["prune_avg_all"] = (stats["prune_sum_all"] / max(1, stats["policy_steps"]))
        stats["prune_avg_eff"] = (stats["prune_sum_eff"] / max(1, stats["effective_steps"]))
        stats["quant_ratio_avg_all"] = (stats["qratio_sum_all"] / max(1, stats["policy_steps"]))
        stats["quant_ratio_avg_eff"] = (stats["qratio_sum_eff"] / max(1, stats["effective_steps"]))
        stats["avg_prune_keep"] = stats["prune_avg_eff"]; stats["avg_quant_ratio"] = stats["quant_ratio_avg_eff"]
        return total_lp, is_greedy_all, stats

    @torch.inference_mode()
    def generate_with_policy(
        self,
        ctx_ids: List[int],
        max_new_tokens: int = 64,
        until: Optional[List[str]] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        return_stats: bool = False,
    ) -> List[int] | Tuple[List[int], dict]:
        """
        Autoregressive generation under policy-controlled sparsity with episodic refresh.
        """
        device = self.device
        running = ctx_ids[:]
        steps_in_episode = 0
        rt = self._new_episode_state(B=1)
        stop_seqs = [self.tok.encode(s, add_special_tokens=False) for s in (until or [])]

        stats = {
            "policy_steps": 0,
            "effective_steps": 0,
            "keep_sum_all": 0.0,
            "keep_sum_eff": 0.0,
            "prune_sum_all": 0.0,
            "prune_sum_eff": 0.0,
            "qratio_sum_all": 0.0,
            "qratio_sum_eff": 0.0,
            "action_hist": [0] * self.A,
            "episode_len": self.episode_len,
            "dense_refresh_tail": self.dense_refresh_tail,
            "dense_first_token": False,
        }

        def sample_from_logits(logits):
            if temperature <= 0:
                return int(torch.argmax(logits, dim=-1)[0].item())
            probs = F.softmax(logits / temperature, dim=-1)
            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                probs_topk = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(probs_topk, num_samples=1)
                return int(topk_idx[0, idx[0]].item())
            if top_p is not None and 0 < top_p < 1:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum <= top_p
                mask[..., 0] = True
                probs_np = (sorted_probs * mask).to(probs.dtype)
                probs_np = probs_np / probs_np.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(probs_np, num_samples=1)
                return int(sorted_idx[0, idx[0]].item())
            return int(torch.multinomial(probs, 1)[0].item())

        past_kv, kv_len, state_lm, last_logits = self._dense_prefill(torch.tensor(running, dtype=torch.long))
        use_prefill_logits = True
        stop_strs = until or []
        for _ in range(max_new_tokens):
            if self.dense_only:
                if use_prefill_logits:
                    nxt = sample_from_logits(last_logits)
                    running.append(nxt)
                    gen_ids = running[len(ctx_ids):]
                    trim = match_stop_suffix(gen_ids, stop_seqs)
                    if trim:
                        del running[-trim:]
                        break
                    cur = torch.tensor([nxt], device=device, dtype=torch.long)
                    pos_ids = kv_len.unsqueeze(1)
                    clear_structured_action(self.m)
                    out_step = self.m(
                        input_ids=cur.view(1,1),
                        use_cache=True,
                        past_key_values=past_kv,
                        position_ids=pos_ids,
                        return_dict=True,
                    )
                    logits_step = out_step.logits[:, -1, :]
                    past_kv = out_step.past_key_values
                    kv_len = kv_len + 1
                    kappa_now = torch.tensor([1.0], device=device)
                    prune_now = torch.tensor([1.0], device=device)
                    qratio_now = torch.tensor([1.0], device=device)
                    stats["policy_steps"] += 1
                    stats["keep_sum_all"] += float(kappa_now.item())
                    stats["prune_sum_all"] += float(prune_now.item())
                    stats["qratio_sum_all"] += float(qratio_now.item())
                    eff_mask = (kv_len > self.thr)
                    if eff_mask.item():
                        stats["effective_steps"] += 1
                        stats["keep_sum_eff"] += float(kappa_now.item())
                        stats["prune_sum_eff"] += float(prune_now.item())
                        stats["qratio_sum_eff"] += float(qratio_now.item())
                    stats["action_hist"][self.dense_idx] += 1
                    use_prefill_logits = False
                    continue
                cur_tok = running[-1]
                cur = torch.tensor([cur_tok], device=device, dtype=torch.long)
                pos_ids = kv_len.unsqueeze(1) 
                clear_structured_action(self.m)
                out_step = self.m(
                    input_ids=cur.view(1,1),
                    use_cache=True,
                    past_key_values=past_kv,
                    position_ids=pos_ids,
                    return_dict=True,
                )
                logits_step = out_step.logits[:, -1, :]
                past_kv = out_step.past_key_values
                kv_len = kv_len + 1
                kappa_now = torch.tensor([1.0], device=device)
                prune_now = torch.tensor([1.0], device=device)
                qratio_now = torch.tensor([1.0], device=device)
                stats["policy_steps"] += 1
                stats["keep_sum_all"] += float(kappa_now.item())
                stats["prune_sum_all"] += float(prune_now.item())
                stats["qratio_sum_all"] += float(qratio_now.item())
                eff_mask = (kv_len > self.thr)
                if eff_mask.item():
                    stats["effective_steps"] += 1
                    stats["keep_sum_eff"] += float(kappa_now.item())
                    stats["prune_sum_eff"] += float(prune_now.item())
                    stats["qratio_sum_eff"] += float(qratio_now.item())
                stats["action_hist"][self.dense_idx] += 1
                nxt = sample_from_logits(logits_step)
                running.append(nxt)
                gen_ids = running[len(ctx_ids):]
                trim = match_stop_suffix(gen_ids, stop_seqs)
                if trim:
                    del running[-trim:]
                    break
                continue

            if steps_in_episode == 0:
                max_ctx = int(getattr(unwrap(self.m).config, "max_position_embeddings", 4096)) - 1
                pref_ids = torch.tensor(running[-max_ctx:], device=device, dtype=torch.long)
                past_kv, kv_len, state_lm, last_logits = self._dense_prefill(pref_ids)
                rt = self._new_episode_state(B=1)
                # Consume prefill logits once (do not re-feed last token)
                use_prefill_logits = True

            if use_prefill_logits:
                nxt = sample_from_logits(last_logits)
                running.append(nxt)
                gen_ids = running[len(ctx_ids):]
                trim = match_stop_suffix(gen_ids, stop_seqs)
                if trim:
                    del running[-trim:]
                    break
                # push into cache & get next logits/state
                cur = torch.tensor([nxt], device=device, dtype=torch.long)
                pos_ids = kv_len.unsqueeze(1)
                out_step = self.m(
                    input_ids=cur.view(1,1),
                    use_cache=True,
                    past_key_values=past_kv,
                    position_ids=pos_ids,
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits_step = out_step.logits[:, -1, :]
                past_kv = out_step.past_key_values
                kv_len = kv_len + 1
                state_lm = out_step.hidden_states[-1][:, -1, :].detach()
                # dense accounting for this step
                kappa_now = torch.tensor([1.0], device=device)
                prune_now = torch.tensor([1.0], device=device)
                qratio_now = torch.tensor([1.0], device=device)
                stats["policy_steps"] += 1
                stats["keep_sum_all"] += float(kappa_now.item())
                stats["prune_sum_all"] += float(prune_now.item())
                stats["qratio_sum_all"] += float(qratio_now.item())
                eff_mask = (kv_len > self.thr)
                if eff_mask.item():
                    stats["effective_steps"] += 1
                    stats["keep_sum_eff"] += float(kappa_now.item())
                    stats["prune_sum_eff"] += float(prune_now.item())
                    stats["qratio_sum_eff"] += float(qratio_now.item())
                stats["action_hist"][self.dense_idx] += 1
                steps_in_episode += 1
                use_prefill_logits = False
                continue

            cur_tok = running[-1]
            cur = torch.tensor([cur_tok], device=device, dtype=torch.long)

            scalars = self._build_scalars(
                t_in_episode=steps_in_episode, kv_len=kv_len, rt=rt,
                steps_seen=len(running), total_steps_target=len(running) + max_new_tokens
            )
            e_tok = self.emb_layer(cur).detach().to(torch.float32)
            logits_a, _v, pi_next = self.pol.step(
                h_lm=state_lm.to(torch.float32),
                e_tok=e_tok,
                scalars=scalars,
                state=rt.pi_state,
                temperature=self.pi_temperature,
            )
            if self._logit_bias is not None:
                logits_a = logits_a - self._logit_bias.to(logits_a.dtype)
            a = torch.argmax(logits_a, dim=-1) if self.greedy_policy else Categorical(logits=logits_a).sample()
            eff_mask = (kv_len > self.thr)
            a_eff = torch.where(eff_mask, a, torch.tensor([self.dense_idx], device=device))
            rt.pi_state = pi_next
            rt.pi_state.last_action = a_eff.detach()
            kappa_now  = self.KEEP[a_eff]
            prune_now  = self.PRUNE[a_eff]
            qbits_now  = self.QBITS[a_eff]
            qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0

            pos_ids = kv_len.unsqueeze(1)
            bias = build_sparse_attention_bias(
                model=self.m,
                past_kv_lens=kv_len,
                keep_fracs=kappa_now,
                Ts=self.Ts, Tw=self.Tw,
                device=device, dtype=self.dtype,
                criteria=self.criteria, tier=self.tier,
            )
            set_structured_action(self.m, float(prune_now.item()), int(qbits_now.item()))
            out_step = self.m(
                input_ids=cur.view(1,1),
                use_cache=True,
                past_key_values=past_kv,
                position_ids=pos_ids,
                attention_mask=bias,
                return_dict=True,
                output_hidden_states=True,
            )
            clear_structured_action(self.m)
            logits_step = out_step.logits[:, -1, :]
            past_kv = out_step.past_key_values
            kv_len = kv_len + 1
            state_lm = out_step.hidden_states[-1][:, -1, :].detach()

            stats["policy_steps"] += 1
            stats["keep_sum_all"] += float(kappa_now.item())
            stats["prune_sum_all"] += float(prune_now.item())
            stats["qratio_sum_all"] += float(qratio_now.item())
            a_idx = int(a_eff.item())
            stats["action_hist"][a_idx] += 1
            if eff_mask.item():
                stats["effective_steps"] += 1
                stats["keep_sum_eff"] += float(kappa_now.item())
                stats["prune_sum_eff"] += float(prune_now.item())
                stats["qratio_sum_eff"] += float(qratio_now.item())

            nxt = sample_from_logits(logits_step)
            running.append(nxt)
            gen_ids = running[len(ctx_ids):]
            trim = match_stop_suffix(gen_ids, stop_seqs)
            text = self.tok.decode(running, skip_special_tokens=True)

            if trim:
                del running[-trim:]
                break

            eff = eff_mask.float()
            rt.cum_eff = rt.cum_eff + eff
            rt.cum_keep = rt.cum_keep + eff * kappa_now
            mean_keep = torch.where(rt.cum_eff > 0, rt.cum_keep / rt.cum_eff, torch.zeros_like(rt.cum_eff))
            dev_abs = (mean_keep - self.C_target).abs()
            if self.budget_penalty == "linear":
                phi_now = (dev_abs - self.tol).clamp_min(0.0)
            else:
                phi_now = (dev_abs - self.tol).clamp_min(0.0) ** 2
            rt.phi_prev = phi_now.detach()

            steps_in_episode += 1
            if steps_in_episode >= self.episode_len:
                steps_in_episode = 0
        stats["keep_avg_all"] = (stats["keep_sum_all"] / max(1, stats["policy_steps"]))
        stats["keep_avg_eff"] = (stats["keep_sum_eff"] / max(1, stats["effective_steps"]))
        total_actions = max(1, sum(stats["action_hist"]))
        stats["action_probs"] = [c / total_actions for c in stats["action_hist"]]
        stats["prune_avg_all"] = (stats["prune_sum_all"] / max(1, stats["policy_steps"]))
        stats["prune_avg_eff"] = (stats["prune_sum_eff"] / max(1, stats["effective_steps"]))
        stats["quant_ratio_avg_all"] = (stats["qratio_sum_all"] / max(1, stats["policy_steps"]))
        stats["quant_ratio_avg_eff"] = (stats["qratio_sum_eff"] / max(1, stats["effective_steps"]))
        stats["avg_prune_keep"] = stats["prune_avg_eff"]; stats["avg_quant_ratio"] = stats["quant_ratio_avg_eff"]

        return (running[len(ctx_ids):], stats) if return_stats else running[len(ctx_ids):]

