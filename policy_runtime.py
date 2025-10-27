# policy_runtime.py
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
        """
        Dense prefill over `ids` (shape [T]) to build KV cache and get last hidden state.
        Returns: past_kv, kv_len (scalar tensor), state_lm (last hidden, [1,H])
        """
        ids = ids.view(1, -1).to(self.device)
        out = self.m(
            input_ids=ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
        past_kv = out.past_key_values
        kv_len = torch.full((1,), ids.size(1) + 1, device=self.device, dtype=torch.long)
        last_h = out.hidden_states[-1][:, -1, :].detach()  # [1,H]
        return past_kv, kv_len, last_h

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
        running = ctx_ids[:]
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
            "dense_first_token": (len(cont_ids) > 0),
        }

        steps_in_episode = 0
        rt = self._new_episode_state(B=1)

        with torch.inference_mode():
            past_kv, kv_len, state_lm = self._dense_prefill(torch.tensor(running, dtype=torch.long))

            if len(cont_ids) > 0:
                ids = torch.tensor(running, dtype=torch.long, device=device).unsqueeze(0)
                out = self.m(input_ids=ids, use_cache=False, return_dict=True)
                logprobs_next = F.log_softmax(out.logits[:, -1, :], dim=-1)
                lp0 = float(logprobs_next[0, cont_ids[0]].item())
                total_lp += lp0
                running.append(cont_ids[0])

        if self.dense_only:
            for i in range(1, len(cont_ids) + 1):
                cur_tok = running[-1]
                cur = torch.tensor([cur_tok], device=device, dtype=torch.long)
                pos_ids = (kv_len - 1).clamp_min(0).unsqueeze(1)
                out_step = self.m(
                    input_ids=cur.view(1, 1),
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
                stats["action_hist"][self.dense_idx] += 1



                labels_next = cont_ids[i] if i < len(cont_ids) else None
                if labels_next is not None:
                    logp = F.log_softmax(logits_step, dim=-1)[0, labels_next]
                    total_lp += float(logp.item())
                    greedy_tok = int(torch.argmax(logits_step, dim=-1)[0].item())
                    if greedy_tok != labels_next:
                        is_greedy_all = False
                    running.append(labels_next)
            stats["keep_avg_all"] = (stats["keep_sum_all"] / max(1, stats["policy_steps"]))
            stats["keep_avg_eff"] = (stats["keep_sum_eff"] / max(1, stats["effective_steps"]))
            total_actions = max(1, sum(stats["action_hist"]))
            stats["action_probs"] = [c / total_actions for c in stats["action_hist"]]
            return total_lp, is_greedy_all, stats
        for i in range(1, len(cont_ids) + 1):
            if steps_in_episode == 0:
                max_ctx = int(getattr(unwrap(self.m).config, "max_position_embeddings", 4096)) - 1
                pref_ids = torch.tensor(running[-max_ctx:], dtype=torch.long)

                past_kv, kv_len, state_lm = self._dense_prefill(pref_ids)
                rt = self._new_episode_state(B=1)
            cur_tok = running[-1]
            cur = torch.tensor([cur_tok], device=device, dtype=torch.long)  # [1]
            labels_next = cont_ids[i] if i < len(cont_ids) else None

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

            eff_mask = (kv_len > self.thr)
            a_eff = torch.where(eff_mask, a, torch.tensor([self.dense_idx], device=device))
            rt.pi_state = pi_next
            rt.pi_state.last_action = a_eff.detach()

            kappa_now  = self.KEEP[a_eff]        # [1]
            prune_now  = self.PRUNE[a_eff]       # [1]
            qbits_now  = self.QBITS[a_eff]       # [1]
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

            if labels_next is not None:
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
        past_kv, kv_len, state_lm = self._dense_prefill(torch.tensor(running, dtype=torch.long))

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

        stop_strs = until or []
        for _ in range(max_new_tokens):
            if self.dense_only:
                cur_tok = running[-1]
                cur = torch.tensor([cur_tok], device=device, dtype=torch.long)
                pos_ids = (kv_len - 1).clamp_min(0).unsqueeze(1)
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

                def sample_from_logits(logits):
                    if temperature <= 0:
                        return int(torch.argmax(logits, dim=-1)[0].item())
                    probs = F.softmax(logits / temperature, dim=-1)
                    return int(torch.multinomial(probs, 1)[0].item())

                nxt = sample_from_logits(logits_step)
                running.append(nxt)
                text = self.tok.decode(running, skip_special_tokens=True)
                if any(s in text for s in stop_strs):
                    break
                continue

            if steps_in_episode == 0:
                max_ctx = int(getattr(unwrap(self.m).config, "max_position_embeddings", 4096)) - 1
                pref_ids = torch.tensor(running[-max_ctx:], dtype=torch.long)
                past_kv, kv_len, state_lm = self._dense_prefill(pref_ids)
                rt = self._new_episode_state(B=1)
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

            text = self.tok.decode(running, skip_special_tokens=True)
            if any(s in text for s in stop_strs):
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

