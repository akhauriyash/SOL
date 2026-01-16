# policy_runtime.py
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Any
import math
import warnings
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.config import Config
from utils.model import load_lm_and_tokenizer, unwrap
from utils.masks import (
    build_sparse_attention_bias,
    enable_structured_controls,
    enable_quest_attention,
    enable_relevancy_attention,
    set_structured_action,
    clear_structured_action,
    clear_quest_token_budgets,
    clear_relevancy_keep,
)
import types
from predictor import RecurrentActorCriticPolicy
from utils.actions import build_action_spec
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval import evaluator
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import numpy as np
from transformers.cache_utils import DynamicCache

def _stream_decoded_token(tok, step_idx: int, gen_ids: list[int], new_token_id: int, prev_text: str = ""):
    """
    Prints (step_idx, token_text) and full decoded text so far.
    Returns updated prev_text (so caller can diff cheaply/robustly).
    """
    # Full decoded text so far (generated portion only)
    cur_text = tok.decode(
        gen_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    # "Detokenized text for *this* token": robustly take suffix difference.
    if cur_text.startswith(prev_text):
        delta = cur_text[len(prev_text):]
    else:
        # Fallback: decode single token (usually fine for LLaMA tokenizers)
        delta = tok.decode(
            [new_token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    # Token “piece” view can also be useful for debugging
    try:
        piece = tok.convert_ids_to_tokens([new_token_id])[0]
    except Exception:
        piece = "<unk>"
    print(delta, end="", flush=True)
    # print(f"{step_idx}\t{repr(delta)}\t(id={new_token_id}, piece={repr(piece)})")
    # print(cur_text)
    # print("-" * 80)

    return cur_text


@dataclass
class PolicyRuntimeState:
    cum_keep: torch.Tensor    # [B]
    cum_eff: torch.Tensor     # [B]
    cum_prune: torch.Tensor   # [B]  (normalized prune keep)
    cum_qratio: torch.Tensor  # [B]  (bits/16)
    pi_state: any             # predictor.PolicyState


def endswith_seq(seq_ids, suffix_ids):
    L = len(suffix_ids)
    return L == 0 or (len(seq_ids) >= L and seq_ids[-L:] == suffix_ids)

def match_stop_suffix(gen_ids, stop_seqs):
    for s in sorted(stop_seqs, key=len, reverse=True):
        if endswith_seq(gen_ids, s):
            return len(s)
    return 0

def _tok_str(tok, tok_id: int) -> str:
    return tok.convert_ids_to_tokens([tok_id])[0]

def _decode_tail(tok, ids: list[int], n: int = 30) -> str:
    return tok.decode(ids[-n:], skip_special_tokens=False, clean_up_tokenization_spaces=False)


class FixedLMRunner:
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

        self.criteria = str(getattr(cfg, "sparsity_criteria", "recency")).lower()
        self.tier = str(getattr(cfg, "relevancy_tier", "per_head"))

        if self.criteria == "quest":
            page = int(getattr(cfg, "quest_page_size", 16))
            enable_quest_attention(self.m, page_size=page)
        elif self.criteria == "relevancy":
            enable_relevancy_attention(self.m, tier=self.tier, cfg=cfg)
        elif self.criteria != "recency":
            raise ValueError(f"Unknown sparsity_criteria: {self.criteria}")
        self.Ts = int(getattr(cfg, "Ts", 0))
        self.Tw = int(getattr(cfg, "Tw", 0))
        self.thr = self.Ts + self.Tw + 1

        spec = build_action_spec(
            keep_fracs=cfg.keep_fracs,
            prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
            quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        )
        self.spec = spec
        self.P_MAX = float(max(spec.prune_keep)) if len(spec.prune_keep) > 0 else 1.0
        enable_structured_controls(self.m)

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
        self._qratio_axis: List[float] = [max(1.0, float(b)) / 16.0 for b in self._qbits_axis]

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
        self.dense_refresh_tail = int(dense_refresh_tail) if dense_refresh_tail is not None else int(self.episode_len)

        self.emb_layer = unwrap(self.m).get_input_embeddings()
        self._struct_mlps = [mod for mod in self.m.modules() if hasattr(mod, "_struct_quant_bits")]


        # ---------------------------------------------------------------------
        # Fast-path: avoid per-token torch.tensor([...]) allocations and avoid
        # scanning model.modules() every step to set structured controls.
        # ---------------------------------------------------------------------
        self._keep_axis_t  = torch.tensor(self._keep_axis,  device=self.device, dtype=torch.float32)
        self._prune_axis_t = torch.tensor(self._prune_axis, device=self.device, dtype=torch.float32)
        self._qbits_axis_t = torch.tensor(self._qbits_axis, device=self.device, dtype=torch.int64)
        self._qratio_axis_t = (self._qbits_axis_t.to(torch.float32).clamp_min(1.0) / 16.0)

        # Cache module refs once (enable_structured_controls already attached attrs)
        self._struct_prune_mods: List[torch.nn.Module] = []
        self._struct_quant_mods: List[torch.nn.Module] = []
        for mod in self.m.modules():
            if hasattr(mod, "_struct_prune_keep"):
                self._struct_prune_mods.append(mod)
            if hasattr(mod, "_struct_quant_bits"):
                self._struct_quant_mods.append(mod)

    def _set_structured_fast(self, prune_keep: torch.Tensor, qbits: torch.Tensor) -> None:
        pk = prune_keep
        qb = qbits

        # Treat dense settings as "disabled"
        if pk is not None and pk.numel() == 1 and float(pk.item()) >= 1.0 - 1e-6:
            pk = None
        if qb is not None and qb.numel() == 1 and int(qb.item()) >= 16:
            qb = None

        for mod in self._struct_prune_mods:
            mod._struct_prune_keep = pk
        for mod in self._struct_quant_mods:
            mod._struct_quant_bits = qb

    def _clear_structured_fast(self) -> None:
        for mod in self._struct_prune_mods:
            mod._struct_prune_keep = None
        for mod in self._struct_quant_mods:
            mod._struct_quant_bits = None

    def _clear_sparse_fast(self) -> None:
        # only matters for quest/relevancy (they stash per-call state on modules)
        if self.criteria == "quest":
            clear_quest_token_budgets(self.m)
        elif self.criteria == "relevancy":
            clear_relevancy_keep(self.m)

    @torch.inference_mode()
    def _dense_prefill_kv_only(self, ids: torch.LongTensor):
        """
        Fixed runner doesn’t need hidden states/logits for prefills.
        This is noticeably cheaper than _dense_prefill(..., output_hidden_states=True).
        """
        ids = ids.view(1, -1).to(self.device)
        out = self.m(input_ids=ids, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        kv_len  = torch.full((1,), ids.size(1) + 1, device=self.device, dtype=torch.long)
        return past_kv, kv_len
 

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
        req = (target * (cum_steps + 1) - cum_sum)
        return hi_idx if abs(hi_v - req) < abs(lo_v - req) else lo_idx

    @torch.inference_mode()
    def _dense_prefill(self, ids: torch.LongTensor):
        ids = ids.view(1, -1).to(self.device)
        out = self.m(input_ids=ids, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        kv_len  = torch.full((1,), ids.size(1) + 1, device=self.device, dtype=torch.long)
        last_h  = None  # [1, H]
        last_logits = out.logits[:, -1, :]
        return past_kv, kv_len, last_h, last_logits

    def _clone_past_kv(self, past_kv):
        """
        Deep-clone a HF KV cache so we can safely restore it later.
        Works for:
          - legacy tuple-of-layer-tuples (k,v)
          - Cache objects with .to_legacy_cache()
        Returns:
          - DynamicCache if input is a Cache-like object
          - legacy tuple otherwise
        """
        if past_kv is None:
            return None
        if hasattr(past_kv, "to_legacy_cache"):
            legacy = past_kv.to_legacy_cache()
            cloned_legacy = tuple(tuple(t.clone() for t in layer) for layer in legacy)
            return DynamicCache.from_legacy_cache(cloned_legacy)
        # Assume legacy tuple structure
        return tuple(tuple(t.clone() for t in layer) for layer in past_kv)

    @torch.inference_mode()
    def _dense_replay_episode(
        self,
        past_kv_base,
        kv_len_base: torch.Tensor,
        episode_cur_tokens: List[int],
    ):
        """
        Paper-style KV refresh:
          - Restore the *dense* pre-episode cache (past_kv_base, kv_len_base)
          - Replay the episode's processed "cur" tokens densely to rebuild their KV entries
        Returns: (past_kv_new, kv_len_new, dense_state_lm_last)
        """
        # Ensure no sparsity/struct state bleeds into the dense replay.
        clear_structured_action(self.m)
        if self.criteria == "quest":
            clear_quest_token_budgets(self.m)
        elif self.criteria == "relevancy":
            clear_relevancy_keep(self.m)

        if len(episode_cur_tokens) == 0:
            return past_kv_base, kv_len_base, None

        input_ids = torch.tensor(
            episode_cur_tokens, device=self.device, dtype=torch.long
        ).view(1, -1)

        past_len = int(kv_len_base.item()) - 1
        pos_ids = torch.arange(
            past_len, past_len + input_ids.size(1),
            device=self.device, dtype=torch.long
        ).view(1, -1)

        out = self.m(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=past_kv_base,
            position_ids=pos_ids,
            attention_mask=None,  # dense replay
            return_dict=True,
            output_hidden_states=True,
        )
        past_kv_new = out.past_key_values
        kv_len_new = kv_len_base + input_ids.size(1)
        state_lm_dense = out.hidden_states[-1][:, -1, :].detach()
        return past_kv_new, kv_len_new, state_lm_dense


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
        self._clear_structured_fast()
        self._clear_sparse_fast()
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
        bos = self.tok.bos_token_id
        if bos is None:
            bos = self.tok.eos_token_id
        bos = int(bos) if bos is not None else 0

        for i in range(B):
            # Ensure non-empty context (needed because we always read running[i][-1])
            if len(running[i]) == 0:
                running[i] = [bos]
            ctx = running[i]
            # PREFILL ALL BUT THE LAST CONTEXT TOKEN so the first scored token is produced under controls
            if len(ctx) > 0:
                tail_except_last = ctx[:-1]
                if len(tail_except_last) > 0:
                    pref_ids = torch.tensor(tail_except_last, device=device, dtype=torch.long)
                    out = self._dense_prefill(pref_ids)  # (past_kv, kv_len, [...])
                    past_kv[i], kv_len[i] = self._dense_prefill_kv_only(pref_ids)
                else:
                    # nothing to prefill; start fresh so we can feed the last ctx token under policy
                    past_kv[i], kv_len[i] = None, torch.tensor([1], device=device)
            else:
                # empty context
                past_kv[i], kv_len[i] = None, torch.tensor([1], device=device)
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

            for pos, i in enumerate(live):
                cur_tok = running[i][-1]
                cur = torch.tensor([cur_tok], device=device, dtype=torch.long)
                labels_next = batch_cont_ids[i][t]

                eff = eff_mask[pos]
                k_idx = k_idx_sel[pos] if eff else self._dense_k
                p_idx = p_idx_sel[pos]
                q_idx = q_idx_sel[pos]

                # fast: slice prebuilt axis tensors (no new allocations)
                kappa_now = self._keep_axis_t[k_idx:k_idx+1]
                prune_now = self._prune_axis_t[p_idx:p_idx+1]
                qbits_now = self._qbits_axis_t[q_idx:q_idx+1]
                qratio_now = self._qratio_axis_t[q_idx:q_idx+1]
 
                # qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0

                # pos_ids = kv_len[i].unsqueeze(1)
                pos_ids = (kv_len[i] - 1).clamp_min(0).unsqueeze(1)
                bias = build_sparse_attention_bias(
                    model=self.m,
                    past_kv_lens=kv_len[i],
                    keep_fracs=kappa_now,
                    Ts=self.Ts, Tw=self.Tw,
                    device=device, dtype=self.dtype,
                    criteria=self.criteria, tier=self.tier,
                )
                self._set_structured_fast(prune_now, qbits_now)
                out_step = self.m(
                    input_ids=cur.view(1, 1),
                    use_cache=True,
                    past_key_values=past_kv[i],
                    position_ids=pos_ids,
                    attention_mask=bias,
                    return_dict=True,
                )
                # keep correctness: clear per-call sparse state (quest/relevancy)
                self._clear_sparse_fast()
 
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

                steps_in_episode[i] += 1
                if steps_in_episode[i] >= self.episode_len:
                    self._clear_structured_fast()
                    self._clear_sparse_fast()
                    max_ctx = int(getattr(unwrap(self.m).config, "max_position_embeddings", 4096)) - 1
                    pref_window = min(self.dense_refresh_tail, max_ctx, len(running[i]))
                    if pref_window > 1:
                        tail_except_last = running[i][-pref_window:-1]
                        if len(tail_except_last) > 0:
                            pref_ids = torch.tensor(tail_except_last, device=device, dtype=torch.long)
                            out = self._dense_prefill(pref_ids)
                            past_kv[i], kv_len[i] = self._dense_prefill_kv_only(pref_ids)
                        else:
                            past_kv[i], kv_len[i] = None, torch.tensor([1], device=device)
                    else:
                        past_kv[i], kv_len[i] = None, torch.tensor([1], device=device)
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
        # ensure the model is left in a clean state for the next request
        self._clear_structured_fast()
        self._clear_sparse_fast()
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
        device = self.device
        running = list(ctx_ids)
        orig_ctx_len = len(running)

        # reset any state from prior request
        self._clear_structured_fast()
        self._clear_sparse_fast()

        # Ensure non-empty context (so we can process "current token" each step).
        if len(running) == 0:
            bos = self.tok.bos_token_id
            if bos is None:
                bos = self.tok.eos_token_id
            running = [int(bos) if bos is not None else 0]
            orig_ctx_len = len(running)

        stop_seqs = [self.tok.encode(s, add_special_tokens=False) for s in (until or [])]

        # Clear any residual structured/sparsity state between requests
        clear_structured_action(self.m)
        if self.criteria == "quest":
            clear_quest_token_budgets(self.m)
        elif self.criteria == "relevancy":
            clear_relevancy_keep(self.m)

        # Fixed strategy for generation: choose the closest discrete action once and stick with it.
        # (This avoids cross-request coupling that batched mixing would introduce for generation.)
        def _nearest_idx(vals: List[float], target: float) -> int:
            return int(min(range(len(vals)), key=lambda i: abs(float(vals[i]) - float(target))))

        k_idx_const = _nearest_idx(self._keep_axis, self.t_keep)
        p_idx_const = _nearest_idx(self._prune_axis, self.t_prune)
        q_idx_const = _nearest_idx(self._qratio_axis, self.t_qratio)

        stats = {
            "policy_steps": 0,
            "effective_steps": 0,          # token-sparsity effective steps
            "keep_sum_all": 0.0,
            "keep_sum_eff": 0.0,
            "prune_sum_all": 0.0,
            "prune_sum_eff": 0.0,          # structural effective steps (eff || struct_on_non_eff)
            "qratio_sum_all": 0.0,
            "qratio_sum_eff": 0.0,         # structural effective steps (eff || struct_on_non_eff)
            "action_hist": [0] * int(self.spec.n_actions),
            "episode_len": self.episode_len,
            "dense_refresh_tail": self.dense_refresh_tail,
            "dense_first_token": False,
        }

        def sample_from_logits(logits_1xV: torch.Tensor) -> int:
            # logits_1xV: [1, V]
            if temperature <= 0:
                return int(torch.argmax(logits_1xV, dim=-1)[0].item())
            probs = F.softmax(logits_1xV / float(temperature), dim=-1)
            if top_k is not None and int(top_k) > 0:
                k = min(int(top_k), probs.size(-1))
                topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
                topk_probs = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(topk_probs, num_samples=1)
                return int(topk_idx[0, idx[0]].item())
            if top_p is not None and 0.0 < float(top_p) < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum <= float(top_p)
                mask[..., 0] = True
                probs_f = (sorted_probs * mask).to(probs.dtype)
                probs_f = probs_f / probs_f.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(probs_f, num_samples=1)
                return int(sorted_idx[0, idx[0]].item())
            return int(torch.multinomial(probs, num_samples=1)[0].item())

        # Prefill all but the last context token so the first generated token is produced under controls.
        head = running[:-1]
        if len(head) > 0:
            past_kv, kv_len = self._dense_prefill_kv_only(torch.tensor(head, device=device, dtype=torch.long))
        else:
            past_kv = None
            kv_len = torch.tensor([1], device=device, dtype=torch.long)

        # === Episode bookkeeping for periodic KV refresh (policy-style) ===
        past_kv_base = self._clone_past_kv(past_kv)
        kv_len_base = kv_len.clone()
        episode_cur_tokens: List[int] = []

        boxed_triggered = False
        boxed_remaining = 0

        decoded_prev = ""

        steps_in_episode = 0
        for step in range(int(max_new_tokens)):
            cur_tok = int(running[-1])
            # Record the token processed this step for later dense replay refresh.
            episode_cur_tokens.append(cur_tok)

            cur = torch.tensor([cur_tok], device=device, dtype=torch.long)

            eff = bool((kv_len > self.thr).item())

            # Token sparsity axis κ: only meaningful if effective
            if eff:
                k_idx = k_idx_const
            else:
                k_idx = self._dense_k

            # Structured + quant axes: respect struct_on_non_eff
            if self.struct_on_non_eff or eff:
                p_idx = p_idx_const
                q_idx = q_idx_const
            else:
                p_idx, q_idx = self._dense_p, self._dense_q

            kappa_now = self._keep_axis_t[k_idx:k_idx+1]
            prune_now = self._prune_axis_t[p_idx:p_idx+1]
            qbits_now = self._qbits_axis_t[q_idx:q_idx+1]
            qratio_now = self._qratio_axis_t[q_idx:q_idx+1]
            pos_ids = (kv_len - 1).clamp_min(0).unsqueeze(1)
            bias = build_sparse_attention_bias(
                model=self.m,
                past_kv_lens=kv_len,
                keep_fracs=kappa_now,
                Ts=self.Ts, Tw=self.Tw,
                device=device, dtype=self.dtype,
                criteria=self.criteria, tier=self.tier,
            )
            self._set_structured_fast(prune_now, qbits_now)
            out_step = self.m(
                input_ids=cur.view(1, 1),
                use_cache=True,
                past_key_values=past_kv,
                position_ids=pos_ids,
                attention_mask=bias,
                return_dict=True,
            )
            self._clear_sparse_fast()

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
            if eff:
                stats["effective_steps"] += 1
                stats["keep_sum_eff"] += float(kappa_now.item())
            if eff or self.struct_on_non_eff:
                stats["prune_sum_eff"] += float(prune_now.item())
                stats["qratio_sum_eff"] += float(qratio_now.item())

            nxt = sample_from_logits(logits_step)
            running.append(int(nxt))

            gen_ids_now = running[orig_ctx_len:]
            # if os.environ.get("STREAM_DECODE", "0") == "1":
            decoded_prev = _stream_decoded_token(
                self.tok,
                step_idx=step,
                gen_ids=gen_ids_now,
                new_token_id=int(nxt),
                prev_text=decoded_prev,
            )

            gen_ids = running[orig_ctx_len:]
            trim = match_stop_suffix(gen_ids, stop_seqs)
            if trim:
                del running[-trim:]
                break

            # Stop after we see '\boxed' + 8 more tokens
            if not boxed_triggered:
                tail_txt = self.tok.decode(
                    running[orig_ctx_len:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if r"\boxed" in tail_txt or "end▁of▁sentence" in tail_txt or ("final answer is" in tail_txt.lower()):
                    boxed_triggered = True
                    boxed_remaining = 8
            else:
                boxed_remaining -= 1
                if boxed_remaining <= 0:
                    break
            steps_in_episode += 1
            if steps_in_episode >= self.episode_len:
                # Policy-style KV refresh at episode boundary:
                # restore dense base cache, replay episode cur tokens densely.
                if step < (int(max_new_tokens) - 1):
                    past_kv, kv_len, _ = self._dense_replay_episode(
                        past_kv_base=past_kv_base,
                        kv_len_base=kv_len_base,
                        episode_cur_tokens=episode_cur_tokens,
                    )
                    past_kv_base = self._clone_past_kv(past_kv)
                    kv_len_base = kv_len.clone()
                    episode_cur_tokens = []
                steps_in_episode = 0
        print("\n", flush=True)
        # Finalize stats
        stats["keep_avg_all"] = stats["keep_sum_all"] / max(1, stats["policy_steps"])
        stats["keep_avg_eff"] = (stats["keep_sum_eff"] / max(1, stats["effective_steps"])) if stats["effective_steps"] > 0 else stats["keep_avg_all"]
        stats["prune_avg_all"] = stats["prune_sum_all"] / max(1, stats["policy_steps"])
        stats["quant_ratio_avg_all"] = stats["qratio_sum_all"] / max(1, stats["policy_steps"])
        denom_eff = stats["effective_steps"] if not self.struct_on_non_eff else stats["policy_steps"]
        stats["prune_avg_eff"] = stats["prune_sum_eff"] / max(1, denom_eff)
        stats["quant_ratio_avg_eff"] = stats["qratio_sum_eff"] / max(1, denom_eff)
        stats["avg_prune_keep"] = stats["prune_avg_eff"]
        stats["avg_quant_ratio"] = stats["quant_ratio_avg_eff"]
        total_actions = max(1, sum(stats["action_hist"]))
        stats["action_probs"] = [c / total_actions for c in stats["action_hist"]]

        self._clear_structured_fast()
        self._clear_sparse_fast()

        gen = running[orig_ctx_len:]
        return (gen, stats) if return_stats else gen



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
        target_C_tok: Optional[float] = None,
        target_C_pru: Optional[float] = None,
        target_C_qbits: Optional[float] = None,
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
        self.P_MAX = float(max(spec.prune_keep)) if len(spec.prune_keep) > 0 else 1.0

        enable_structured_controls(self.m)

        # --- Enable quest/relevancy attention if requested ---
        self.criteria = str(getattr(cfg, "sparsity_criteria", "recency")).lower()
        self.tier = str(getattr(cfg, "relevancy_tier", "per_head"))
        if self.criteria == "quest":
            page = int(getattr(cfg, "quest_page_size", 16))
            enable_quest_attention(self.m, page_size=page)
        elif self.criteria == "relevancy":
            enable_relevancy_attention(self.m, tier=self.tier, cfg=cfg)
        elif self.criteria != "recency":
            raise ValueError(f"Unknown sparsity_criteria: {self.criteria}")
        self._scalar_dim = int(getattr(self.pol, "scalar_dim", 8))
        self.sparsity_bias = float(sparsity_bias)
        self.prune_bias    = float(prune_bias)
        self.quant_bias    = float(quant_bias)
        self._logit_bias: Optional[torch.Tensor] = None
        if (self.sparsity_bias != 0.0) or (self.prune_bias != 0.0) or (self.quant_bias != 0.0):
            def _norm01(x: torch.Tensor) -> torch.Tensor:
                x = x.to(torch.float32)
                xmin = torch.min(x)
                xmax = torch.max(x)
                denom = torch.clamp(xmax - xmin, min=1e-8)
                return (x - xmin) / denom

            dens_keep  = _norm01(self.KEEP)
            dens_prune = _norm01(self.PRUNE)
            dens_qbits = _norm01(self.QBITS.to(torch.float32))
            bias_vec = (
                self.sparsity_bias * dens_keep
                + self.prune_bias  * dens_prune
                + self.quant_bias  * dens_qbits
            )  # [A]
            self._logit_bias = bias_vec.unsqueeze(0).to(self.device)

        self.greedy_policy = bool(greedy_policy)
        self.pi_temperature = float(policy_temperature)
        self.dense_only = bool(dense_only)

        self.episode_len = int(episode_len) if episode_len is not None else int(getattr(cfg, "rollout_len", 16))
        self.dense_refresh_tail = int(dense_refresh_tail) if dense_refresh_tail is not None else int(self.episode_len)

        # === Target budgets (policy conditions on these) ===
        C_tok_default = float(getattr(cfg, "C_target_token", getattr(cfg, "C_target", getattr(cfg, "keep_target", 1.0))))
        C_pru_default = float(getattr(cfg, "C_target_prune", 0.70))
        C_qbits_default = float(getattr(cfg, "C_target_quant_bits", 8.0))
        if target_C_tok is None:
            target_C_tok = getattr(cfg, "eval_C_tok", None)
        if target_C_pru is None:
            target_C_pru = getattr(cfg, "eval_C_pru", None)
        if target_C_qbits is None:
            target_C_qbits = getattr(cfg, "eval_C_qbits", None)
        self.C_tok = float(C_tok_default if target_C_tok is None else target_C_tok)
        self.C_pru = float(C_pru_default if target_C_pru is None else target_C_pru)
        self.C_qbits = float(C_qbits_default if target_C_qbits is None else target_C_qbits)
        self.C_qratio = float(self.C_qbits) / 16.0
        self.emb_layer = unwrap(self.m).get_input_embeddings()
        self._scalar_dim = int(getattr(self.pol, "scalar_dim", getattr(self.cfg, "policy_scalar_dim", 8)))
 
        base = unwrap(self.m)
        # For HF LLaMA CausalLM, base.model is the transformer; lm_head maps hidden->logits.
        self._core_lm = getattr(base, "model", None) or getattr(base, "base_model", None) or base
        self._lm_head = getattr(base, "lm_head", None) or base.get_output_embeddings()

        # --- PERF: avoid scanning model.modules() every token in set/clear_structured_action ---
        # Only LlamaMLP modules actually consume _struct_prune_keep/_struct_quant_bits.
        self._struct_mlps = [mod for mod in self.m.modules() if hasattr(mod, "_struct_quant_bits")]

        # tiny constant tensor to avoid realloc in torch.where(...)
        self._dense_idx_tensor = torch.tensor([self.dense_idx], device=self.device, dtype=torch.long)

    def _clear_structured_fast(self) -> None:
        for mlp in self._struct_mlps:
            mlp._struct_prune_keep = None
            mlp._struct_quant_bits = None

    def _set_structured_fast(self, prune_keep: torch.Tensor, qbits: torch.Tensor) -> None:
        """
        Set structured controls without walking the whole module tree.
        Use None for no-ops (prune_keep==1, qbits==16) so patched MLP wrapper early-exits.
        """
        pk = prune_keep
        qb = qbits
        if pk is not None and pk.numel() == 1 and float(pk.item()) >= 1.0 - 1e-6:
            pk = None
        if qb is not None and qb.numel() == 1 and int(qb.item()) >= 16:
            qb = None
        for mlp in self._struct_mlps:
            mlp._struct_prune_keep = pk
            mlp._struct_quant_bits = qb

    @torch.inference_mode()
    def _lm_step(self, input_ids, past_key_values, position_ids, attention_mask):
        """
        Run ONE forward step through the *base* model (no hidden_states list),
        return (logits_last [B,V], past_kv, h_last [B,H]).
        """
        out = self._core_lm(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        h_last = out.last_hidden_state[:, -1, :]         # [B,H]
        # HF LLaMA CausalLM typically casts logits to float() before returning.
        logits_last = self._lm_head(h_last).to(torch.float32)   # [B,V]
        return logits_last, out.past_key_values, h_last


    def _clone_past_kv(self, past_kv):
        """
        Deep-clone a HF KV cache so we can safely restore it later.
        Works for:
          - legacy tuple-of-layer-tuples (k,v)
          - Cache objects with .to_legacy_cache()
        Returns:
          - DynamicCache if input is a Cache-like object
          - legacy tuple otherwise
        """
        if past_kv is None:
            return None
        if hasattr(past_kv, "to_legacy_cache"):
            legacy = past_kv.to_legacy_cache()
            cloned_legacy = tuple(tuple(t.clone() for t in layer) for layer in legacy)
            return DynamicCache.from_legacy_cache(cloned_legacy)
        # Assume legacy tuple structure
        return tuple(tuple(t.clone() for t in layer) for layer in past_kv)

    @torch.inference_mode()
    def _dense_replay_episode(
        self,
        past_kv_base,
        kv_len_base: torch.Tensor,
        episode_cur_tokens: List[int],
    ):
        """
        Paper-style KV refresh:
          - Restore the *dense* pre-episode cache (past_kv_base, kv_len_base)
          - Replay the episode's processed "cur" tokens densely to rebuild their KV entries
        Returns: (past_kv_new, kv_len_new, dense_state_lm_last)
        """
        # Ensure no sparsity/struct state bleeds into the dense replay.
        self._clear_structured_fast()
        if self.criteria == "quest":
            clear_quest_token_budgets(self.m)
        elif self.criteria == "relevancy":
            clear_relevancy_keep(self.m)

        if len(episode_cur_tokens) == 0:
            return past_kv_base, kv_len_base, None

        input_ids = torch.tensor(
            episode_cur_tokens, device=self.device, dtype=torch.long
        ).view(1, -1)

        past_len = int(kv_len_base.item()) - 1
        pos_ids = torch.arange(
            past_len, past_len + input_ids.size(1),
            device=self.device, dtype=torch.long
        ).view(1, -1)

        logits_last, past_kv_new, h_last = self._lm_step(
            input_ids=input_ids,
            past_key_values=past_kv_base,
            position_ids=pos_ids,
            attention_mask=None,
        )
        kv_len_new = kv_len_base + input_ids.size(1)
        state_lm_dense = h_last.detach().to(torch.float32)
        return past_kv_new, kv_len_new, state_lm_dense


    @torch.inference_mode()
    def _dense_prefill(self, ids: torch.LongTensor):
        ids = ids.view(1, -1).to(self.device)
        logits_last, past_kv, h_last = self._lm_step(
            input_ids=ids,
            past_key_values=None,
            position_ids=None,
            attention_mask=None,
        )
        kv_len  = torch.full((1,), ids.size(1) + 1, device=self.device, dtype=torch.long)
        return past_kv, kv_len, h_last.detach().to(torch.float32), logits_last


    def _new_episode_state(self, B: int = 1):
        pi_state = self.pol.init_state(B, device=self.device)
        zero = torch.zeros(B, device=self.device)
        return PolicyRuntimeState(
            cum_keep=zero.clone(),
            cum_eff=zero.clone(),
            cum_prune=zero.clone(),
            cum_qratio=zero.clone(),
            pi_state=pi_state,
        )

    def _build_scalars(self, t_in_episode: int, kv_len: torch.Tensor, rt: PolicyRuntimeState, steps_seen: int, total_steps_target: int):
        B = kv_len.shape[0]

        # 8-D scalar layout (matches training/eval):
        #   [0] t_frac            in [0,1]
        #   [1] eff_flag          in {0,1}
        #   [2] C_tok_target      in [0,1]
        #   [3] C_pru_target      in [0,1]
        #   [4] C_qratio_target   in [0,1] (bits/16)
        #   [5] dev_keep          = mean_keep_prev   - C_tok_target
        #   [6] dev_prune         = mean_prune_prev  - C_pru_target
        #   [7] dev_qratio        = mean_qratio_prev - C_qratio_target
        t_frac = torch.full(
            (B, 1),
            (t_in_episode + 1) / float(self.episode_len),
            device=self.device,
            dtype=torch.float32,
        )
        eff_flag = (kv_len > self.thr).float().view(B, 1)

        C_tok = torch.full_like(t_frac, self.C_tok)
        C_pru = torch.full_like(t_frac, self.C_pru)
        C_qratio = torch.full_like(t_frac, self.C_qratio)


        mean_keep_prev = torch.where(
            rt.cum_eff > 0,
            rt.cum_keep / rt.cum_eff,
            torch.full_like(rt.cum_keep, self.C_tok),
        )
        mean_prune_prev = torch.where(
            rt.cum_eff > 0,
            rt.cum_prune / rt.cum_eff,
            torch.full_like(rt.cum_prune, self.C_pru),
        )
        mean_qratio_prev = torch.where(
            rt.cum_eff > 0,
            rt.cum_qratio / rt.cum_eff,
            torch.full_like(rt.cum_qratio, self.C_qratio),
        )

        dev_keep = mean_keep_prev - self.C_tok
        dev_prune = mean_prune_prev - self.C_pru
        dev_qratio = mean_qratio_prev - self.C_qratio

        scalars = torch.cat(
            [
                t_frac,
                eff_flag,
                C_tok,
                C_pru,
                C_qratio,
                dev_keep.view(B, 1),
                dev_prune.view(B, 1),
                dev_qratio.view(B, 1),
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
        device = self.device
        running = list(ctx_ids)
        if len(running) == 0:
            bos = self.tok.bos_token_id
            if bos is None:
                bos = self.tok.eos_token_id
            running = [int(bos)]

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

        # Clear any residual structured state between requests
        self._clear_structured_fast()
        if self.criteria == "quest":
            clear_quest_token_budgets(self.m)
        elif self.criteria == "relevancy":
            clear_relevancy_keep(self.m)

        steps_in_episode = 0
        rt = self._new_episode_state(B=1)

        # Dense prefill of the "head" (exclude last token so it is processed under controls)
        head = running[:-1]
        if len(head) > 0:
            past_kv, kv_len, state_lm, _ = self._dense_prefill(
                torch.tensor(head, device=device, dtype=torch.long)
            )
        else:
            past_kv = None
            kv_len = torch.tensor([1], device=device, dtype=torch.long)
            hidden_size = int(getattr(unwrap(self.m).config, "hidden_size",
                                      getattr(unwrap(self.m).config, "n_embd", 0)))
            state_lm = torch.zeros(1, hidden_size, device=device, dtype=torch.float32)
        # === Episode bookkeeping for periodic KV refresh ===
        past_kv_base = self._clone_past_kv(past_kv)
        kv_len_base = kv_len.clone()
        episode_cur_tokens: List[int] = []


        for i, labels_next in enumerate(cont_ids):
            cur_tok = int(running[-1])
            cur = torch.tensor([cur_tok], device=device, dtype=torch.long)
            labels_next = int(labels_next)
            episode_cur_tokens.append(cur_tok)
            eff_mask = (kv_len > self.thr)  # [1] bool

            if self.dense_only:
                a_eff = self._dense_idx_tensor
                kappa_now = self.KEEP[a_eff]
                prune_now = self.PRUNE[a_eff]
                qbits_now = self.QBITS[a_eff]
                qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0
            else:
                scalars = self._build_scalars(
                    t_in_episode=steps_in_episode,
                    kv_len=kv_len,
                    rt=rt,
                    steps_seen=i,
                    total_steps_target=len(cont_ids),
                )
                e_tok = self.emb_layer(cur).detach().to(torch.float32)
                logits_a, _v, pi_next = self.pol.step(
                    h_lm=state_lm.to(torch.float32),
                    e_tok=e_tok,
                    scalars=scalars,
                    state=rt.pi_state,
                    temperature=policy_temperature,
                )
                if self._logit_bias is not None:
                    # logits_a is float32; keep bias float32 to avoid per-step .to()
                    logits_a = logits_a - self._logit_bias
                if greedy_actions or self.greedy_policy:
                    a = torch.argmax(logits_a, dim=-1)
                else:
                    a = Categorical(logits=logits_a).sample()
                a_eff = torch.where(eff_mask, a, self._dense_idx_tensor)
                rt.pi_state = pi_next
                rt.pi_state.last_action = a_eff.detach()

                kappa_now = self.KEEP[a_eff]
                prune_now = self.PRUNE[a_eff]
                qbits_now = self.QBITS[a_eff]
                qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0

            pos_ids = (kv_len - 1).clamp_min(0).unsqueeze(1)
            bias = build_sparse_attention_bias(
                model=self.m,
                past_kv_lens=kv_len,
                keep_fracs=kappa_now,
                Ts=self.Ts, Tw=self.Tw,
                device=device, dtype=self.dtype,
                criteria=self.criteria, tier=self.tier,
            )
            self._set_structured_fast(prune_now, qbits_now)
            logits_step, past_kv, h_last = self._lm_step(
                input_ids=cur.view(1, 1),
                past_key_values=past_kv,
                position_ids=pos_ids,
                attention_mask=bias,
            )
            kv_len = kv_len + 1
            # keep policy state in float32 so we don't cast every step
            state_lm = h_last.detach().to(torch.float32)
            # Stats
            stats["policy_steps"] += 1
            stats["keep_sum_all"] += float(kappa_now.item())
            stats["prune_sum_all"] += float(prune_now.item())
            stats["qratio_sum_all"] += float(qratio_now.item())
            stats["action_hist"][int(a_eff.item())] += 1
            if eff_mask.item():
                stats["effective_steps"] += 1
                stats["keep_sum_eff"] += float(kappa_now.item())
                stats["prune_sum_eff"] += float(prune_now.item())
                stats["qratio_sum_eff"] += float(qratio_now.item())

            # Score continuation token
            logp = F.log_softmax(logits_step, dim=-1)[0, labels_next]
            total_lp += float(logp.item())
            greedy_tok = int(torch.argmax(logits_step, dim=-1)[0].item())
            if greedy_tok != labels_next:
                is_greedy_all = False

            # Budget tracking for next step scalars (effective steps only)
            eff = eff_mask.to(torch.float32)
            rt.cum_eff = rt.cum_eff + eff
            rt.cum_keep = rt.cum_keep + eff * kappa_now
            rt.cum_prune = rt.cum_prune + eff * (prune_now.to(torch.float32) / self.P_MAX)
            rt.cum_qratio = rt.cum_qratio + eff * qratio_now

            running.append(labels_next)
            steps_in_episode += 1
            if steps_in_episode >= self.episode_len:
                # Periodic KV refresh at episode boundary (paper behavior):
                # restore dense base cache, replay episode cur tokens densely, reset policy state.
                if i < (len(cont_ids) - 1):
                    past_kv, kv_len, state_lm_dense = self._dense_replay_episode(
                        past_kv_base=past_kv_base,
                        kv_len_base=kv_len_base,
                        episode_cur_tokens=episode_cur_tokens,
                    )
                    if state_lm_dense is not None:
                        state_lm = state_lm_dense
                    past_kv_base = self._clone_past_kv(past_kv)
                    kv_len_base = kv_len.clone()
                    episode_cur_tokens = []
                    rt = self._new_episode_state(B=1)
                steps_in_episode = 0

        # finalize
        stats["keep_avg_all"] = (stats["keep_sum_all"] / max(1, stats["policy_steps"]))
        if stats["effective_steps"] > 0:
            stats["keep_avg_eff"] = (stats["keep_sum_eff"] / max(1, stats["effective_steps"]))
        else:
            stats["keep_avg_eff"] = stats["keep_avg_all"]
        stats["prune_avg_all"] = (stats["prune_sum_all"] / max(1, stats["policy_steps"]))
        stats["prune_avg_eff"] = (stats["prune_sum_eff"] / max(1, stats["effective_steps"])) if stats["effective_steps"] > 0 else stats["prune_avg_all"]
        stats["quant_ratio_avg_all"] = (stats["qratio_sum_all"] / max(1, stats["policy_steps"]))
        stats["quant_ratio_avg_eff"] = (stats["qratio_sum_eff"] / max(1, stats["effective_steps"])) if stats["effective_steps"] > 0 else stats["quant_ratio_avg_all"]
        stats["avg_prune_keep"] = stats["prune_avg_eff"]
        stats["avg_quant_ratio"] = stats["quant_ratio_avg_eff"]
        total_actions = max(1, sum(stats["action_hist"]))
        stats["action_probs"] = [c / total_actions for c in stats["action_hist"]]
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
        device = self.device
        orig_ctx_len = len(ctx_ids)
        running = list(ctx_ids)
        if len(running) == 0:
            bos = self.tok.bos_token_id
            if bos is None:
                bos = self.tok.eos_token_id
            running = [int(bos)]
            orig_ctx_len = len(running)  # treat inserted BOS as context

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

        # Clear any residual structured state between requests
        self._clear_structured_fast()
        if self.criteria == "quest":
            clear_quest_token_budgets(self.m)
        elif self.criteria == "relevancy":
            clear_relevancy_keep(self.m)

        def sample_from_logits(logits):
            if temperature <= 0:
                return int(torch.argmax(logits, dim=-1)[0].item())
            probs = F.softmax(logits / float(temperature), dim=-1)
            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, k=min(int(top_k), probs.size(-1)), dim=-1)
                probs_topk = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(probs_topk, num_samples=1)
                return int(topk_idx[0, idx[0]].item())
            if top_p is not None and 0 < float(top_p) < 1:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum <= float(top_p)
                mask[..., 0] = True
                probs_np = (sorted_probs * mask).to(probs.dtype)
                probs_np = probs_np / probs_np.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(probs_np, num_samples=1)
                return int(sorted_idx[0, idx[0]].item())
            return int(torch.multinomial(probs, 1)[0].item())

        steps_in_episode = 0
        rt = self._new_episode_state(B=1)

        # Dense prefill head (exclude last token so it is processed under controls)
        head = running[:-1]
        if len(head) > 0:
            past_kv, kv_len, state_lm, _ = self._dense_prefill(
                torch.tensor(head, device=device, dtype=torch.long)
            )
        else:
            past_kv = None
            kv_len = torch.tensor([1], device=device, dtype=torch.long)
            hidden_size = int(getattr(unwrap(self.m).config, "hidden_size",
                                      getattr(unwrap(self.m).config, "n_embd", 0)))
            state_lm = torch.zeros(1, hidden_size, device=device, dtype=torch.float32)
        
        decoded_prev = ""
        print("=== INPUT (ctx_ids decoded) ===", flush=True)
        print(self.tok.decode(
            running[:orig_ctx_len],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ), flush=True)
        print("===============================", flush=True)

        # === Episode bookkeeping for periodic KV refresh ===
        past_kv_base = self._clone_past_kv(past_kv)
        kv_len_base = kv_len.clone()

        boxed_triggered = False
        boxed_remaining = 0

        episode_cur_tokens: List[int] = []
        for step in range(max_new_tokens):
            cur_tok = int(running[-1])
            cur = torch.tensor([cur_tok], device=device, dtype=torch.long)
            eff_mask = (kv_len > self.thr)
            # Record the token processed this step for later dense replay refresh.
            episode_cur_tokens.append(cur_tok)


            if self.dense_only:
                a_eff = self._dense_idx_tensor
                kappa_now = self.KEEP[a_eff]
                prune_now = self.PRUNE[a_eff]
                qbits_now = self.QBITS[a_eff]
                qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0
            else:
                scalars = self._build_scalars(
                    t_in_episode=steps_in_episode,
                    kv_len=kv_len,
                    rt=rt,
                    steps_seen=len(running),
                    total_steps_target=len(running) + max_new_tokens,
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
                    logits_a = logits_a - self._logit_bias

                a = torch.argmax(logits_a, dim=-1) if self.greedy_policy else Categorical(logits=logits_a).sample()
                a_eff = torch.where(eff_mask, a, self._dense_idx_tensor)
                rt.pi_state = pi_next
                rt.pi_state.last_action = a_eff.detach()

                kappa_now = self.KEEP[a_eff]
                prune_now = self.PRUNE[a_eff]
                qbits_now = self.QBITS[a_eff]
                qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0

            pos_ids = (kv_len - 1).clamp_min(0).unsqueeze(1)
            bias = build_sparse_attention_bias(
                model=self.m,
                past_kv_lens=kv_len,
                keep_fracs=kappa_now,
                Ts=self.Ts, Tw=self.Tw,
                device=device, dtype=self.dtype,
                criteria=self.criteria, tier=self.tier,
            )
            self._set_structured_fast(prune_now, qbits_now)
            logits_step, past_kv, h_last = self._lm_step(
                input_ids=cur.view(1, 1),
                past_key_values=past_kv,
                position_ids=pos_ids,
                attention_mask=bias,
            )
            kv_len = kv_len + 1
            state_lm = h_last.detach().to(torch.float32)

            # Stats
            stats["policy_steps"] += 1
            stats["keep_sum_all"] += float(kappa_now.item())
            stats["prune_sum_all"] += float(prune_now.item())
            stats["qratio_sum_all"] += float(qratio_now.item())
            stats["action_hist"][int(a_eff.item())] += 1
            if eff_mask.item():
                stats["effective_steps"] += 1
                stats["keep_sum_eff"] += float(kappa_now.item())
                stats["prune_sum_eff"] += float(prune_now.item())
                stats["qratio_sum_eff"] += float(qratio_now.item())

            nxt = sample_from_logits(logits_step)
            running.append(int(nxt))
            gen_ids_now = running[orig_ctx_len:]
            # if os.environ.get("STREAM_DECODE", "0") == "1":
            decoded_prev = _stream_decoded_token(
                self.tok,
                step_idx=step,
                gen_ids=gen_ids_now,
                new_token_id=int(nxt),
                prev_text=decoded_prev,
            )

            gen_ids = running[orig_ctx_len:]
            trim = match_stop_suffix(gen_ids, stop_seqs)
            if trim:
                del running[-trim:]
                break
            # Stop after we see '\boxed' + 8 more tokens
            if not boxed_triggered:
                tail_txt = self.tok.decode(
                    running[orig_ctx_len:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if r"\boxed" in tail_txt or "end▁of▁sentence" in tail_txt or ("final answer is" in tail_txt.lower()):
                    boxed_triggered = True
                    boxed_remaining = 8
            else:
                boxed_remaining -= 1
                if boxed_remaining <= 0:
                    print("Trigger detected end", end="", flush=True)
                    break

            # Budget tracking for next-step features (effective steps only)
            eff = eff_mask.to(torch.float32)
            rt.cum_eff = rt.cum_eff + eff
            rt.cum_keep = rt.cum_keep + eff * kappa_now
            rt.cum_prune = rt.cum_prune + eff * (prune_now.to(torch.float32) / self.P_MAX)
            rt.cum_qratio = rt.cum_qratio + eff * qratio_now

            steps_in_episode += 1
            if steps_in_episode >= self.episode_len:
                if step < (max_new_tokens - 1):
                    past_kv, kv_len, state_lm_dense = self._dense_replay_episode(
                        past_kv_base=past_kv_base,
                        kv_len_base=kv_len_base,
                        episode_cur_tokens=episode_cur_tokens,
                    )
                    if state_lm_dense is not None:
                        state_lm = state_lm_dense
                    past_kv_base = self._clone_past_kv(past_kv)
                    kv_len_base = kv_len.clone()
                    episode_cur_tokens = []
                    rt = self._new_episode_state(B=1)
                steps_in_episode = 0

        stats["keep_avg_all"] = (stats["keep_sum_all"] / max(1, stats["policy_steps"]))
        stats["keep_avg_eff"] = (stats["keep_sum_eff"] / max(1, stats["effective_steps"])) if stats["effective_steps"] > 0 else stats["keep_avg_all"]
        stats["prune_avg_all"] = (stats["prune_sum_all"] / max(1, stats["policy_steps"]))
        stats["prune_avg_eff"] = (stats["prune_sum_eff"] / max(1, stats["effective_steps"])) if stats["effective_steps"] > 0 else stats["prune_avg_all"]
        stats["quant_ratio_avg_all"] = (stats["qratio_sum_all"] / max(1, stats["policy_steps"]))
        stats["quant_ratio_avg_eff"] = (stats["qratio_sum_eff"] / max(1, stats["effective_steps"])) if stats["effective_steps"] > 0 else stats["quant_ratio_avg_all"]
        stats["avg_prune_keep"] = stats["prune_avg_eff"]
        stats["avg_quant_ratio"] = stats["quant_ratio_avg_eff"]
        total_actions = max(1, sum(stats["action_hist"]))
        stats["action_probs"] = [c / total_actions for c in stats["action_hist"]]

        gen = running[orig_ctx_len:]
        return (gen, stats) if return_stats else gen