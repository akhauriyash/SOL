# eval_policy_lmeval.py
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

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        if o.ndim == 0:
            return o.item()
        return o.detach().cpu().tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)

def print_compact_summary(res):
    results = res.get("results", {})
    accs = []
    print("\n=== Per-task accuracy ===")
    for task, metrics in results.items():
        for k in ("acc,none", "acc", "exact_match,none", "exact_match"):
            if k in metrics:
                v = metrics[k]
                print(f"{task}: {v:.4f}")
                accs.append(v)
                break
        else:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"{task} ({k}): {v:.4f}")
                    break

    agg = res.get("aggregated") or res.get("groups") or {}
    macro = None
    for key in ("macro_avg", "overall", "total"):
        if isinstance(agg, dict) and key in agg:
            block = agg[key]
            for k in ("acc,none", "acc", "exact_match,none", "exact_match"):
                if k in block:
                    macro = block[k]
                    break
            if macro is not None:
                break

    print("\n=== Overall ===")
    if macro is not None:
        print(f"Macro average accuracy: {macro:.4f}")
    elif accs:
        print(f"Macro average accuracy (computed): {sum(accs)/len(accs):.4f}")
    else:
        print("No accuracy-like metric found in results.")

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
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
) -> Config:
    cfg = Config()
    sd_cpu = torch.load(ckpt_path, map_location="cpu")
    sd_cfg = sd_cpu.get("cfg")
    if sd_cfg:
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
        base_rel = None
        if meta is not None:
            base_rel = meta.get("config_paths", {}).get("base")
        if base_rel:
            yaml_path = os.path.join(ckpt_dir, "code", base_rel)
            if os.path.exists(yaml_path):
                from utils.config import apply_cfg_overrides_from_file
                apply_cfg_overrides_from_file(cfg, yaml_path, is_main=True)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        cfg.dtype = torch.bfloat16
    else:
        cfg.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if dataset_name is not None:
        cfg.dataset_name = dataset_name
    if dataset_config is not None:
        cfg.dataset_config = dataset_config
    return cfg

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
        full_refresh: bool = True,
        lambda_keep: float = 0.0,
        lambda_prune: float = 0.0,
        lambda_quant: float = 0.0,
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
        self.full_refresh = bool(full_refresh)
        assert self.full_refresh, "Only full_refresh=True is supported"

        self.dense_idx = int(spec.dense_idx)
        enable_structured_controls(self.m)
        self._scalar_dim = int(getattr(self.pol, "scalar_dim", 12))

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
                stats["policy_steps"] += 1
                stats["keep_sum_all"] += float(kappa_now.item())
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
                if self.full_refresh:
                    pref_ids = torch.tensor(running[-max_ctx:], dtype=torch.long)
                else:
                    tail_len = min(self.dense_refresh_tail, len(running), max_ctx)
                    pref_ids = torch.tensor(running[-tail_len:], dtype=torch.long)

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
            a_idx = int(a_eff.item())
            stats["action_hist"][a_idx] += 1
            if eff_mask.item():
                stats["effective_steps"] += 1
                stats["keep_sum_eff"] += float(kappa_now.item())

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
                stats["policy_steps"] += 1
                stats["keep_sum_all"] += float(kappa_now.item())
                eff_mask = (kv_len > self.thr)
                if eff_mask.item():
                    stats["effective_steps"] += 1
                    stats["keep_sum_eff"] += float(kappa_now.item())
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
                if self.full_refresh:
                    pref_ids = torch.tensor(running[-max_ctx:], dtype=torch.long)
                else:
                    tail_len = min(self.dense_refresh_tail, len(running), max_ctx)
                    pref_ids = torch.tensor(running[-tail_len:], dtype=torch.long)

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
            a = torch.argmax(logits_a, dim=-1) if self.greedy_policy else Categorical(logits=logits_a).sample()
            eff_mask = (kv_len > self.thr)
            a_eff = torch.where(eff_mask, a, torch.tensor([self.dense_idx], device=device))
            rt.pi_state = pi_next
            rt.pi_state.last_action = a_eff.detach()
            kappa_now  = self.KEEP[a_eff]
            prune_now  = self.PRUNE[a_eff]
            qbits_now  = self.QBITS[a_eff]

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
            a_idx = int(a_eff.item())
            stats["action_hist"][a_idx] += 1
            if eff_mask.item():
                stats["effective_steps"] += 1
                stats["keep_sum_eff"] += float(kappa_now.item())

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

        return (running[len(ctx_ids):], stats) if return_stats else running[len(ctx_ids):]


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
    ):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = find_latest_ckpt(ckpt_dir, mode)
        if self.ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

        self.cfg = load_cfg_from_checkpoint_or_yaml(ckpt_dir, self.ckpt_path)
        self.tok, self.model = load_lm_and_tokenizer(self.cfg)

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
            "action_hist": [0] * int(self.spec.n_actions),
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
            "keep_fracs": list(self.spec.token_keep),
            "action_tags": list(self.spec.tags),
            **stats,
        }
        self.per_request_stats.append(entry)
        self._agg["policy_steps"] += stats.get("policy_steps", 0)
        self._agg["effective_steps"] += stats.get("effective_steps", 0)
        self._agg["keep_sum_all"] += stats.get("keep_sum_all", 0.0)
        self._agg["keep_sum_eff"] += stats.get("keep_sum_eff", 0.0)
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="Directory with policy_*.pt")
    p.add_argument("--mode", type=str, default="latest", choices=["latest", "best"])
    p.add_argument("--tasks", type=str, default="piqa,arc_easy")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Limit samples per task (LM-Eval)")
    p.add_argument("--episode_len", type=int, default=None, help="Override episode length (default cfg.rollout_len)")
    p.add_argument("--dense_refresh_tail", type=int, default=None, help="Tail tokens to dense-prefill between episodes (default Ts+Tw+1)")
    p.add_argument("--policy_temperature", type=float, default=0.6)
    p.add_argument("--greedy_policy", action="store_true", help="Use argmax over κ actions (default True)")
    p.add_argument("--dense_baseline", action="store_true", help="Also run dense baseline (no policy, no sparse masks)")
    p.add_argument("--export_sparsity_json", type=str, default=None,
                   help="If set, write per-request sparsity stats to this JSON file")
    args = p.parse_args()

    model = PolicyHarnessLM(
        ckpt_dir=args.ckpt_dir,
        mode=args.mode,
        greedy_policy = args.greedy_policy,
        policy_temperature=args.policy_temperature,
        episode_len=args.episode_len,
        dense_refresh_tail=args.dense_refresh_tail,
        dense_only=False,
        max_batch=args.batch_size,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    res_policy = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=0,
    )
    stats_all = model.export_sparsity_stats()
    print("\n=== Observed sparsity (policy run) ===")
    print(json.dumps(stats_all["global"], indent=2))
    if args.export_sparsity_json:
        with open(args.export_sparsity_json, "w") as f:
            json.dump(stats_all, f, indent=2, default=_json_default)
        print(f"[saved] per-request sparsity → {args.export_sparsity_json}")

    if not args.dense_baseline:
        print(json.dumps(res_policy, indent=2, default=_json_default))
        return

    print_compact_summary(res_policy)
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
    res_dense = evaluator.simple_evaluate(
        model=dense_model, tasks=tasks, batch_size=args.batch_size, limit=args.limit, num_fewshot=0
    )

    print(json.dumps({"policy_sparse": res_policy, "dense_baseline": res_dense},
                    indent=2, default=_json_default))
    dense_stats = dense_model.export_sparsity_stats()
    print("\n=== Observed sparsity (dense baseline) ===")
    print(json.dumps(dense_stats["global"], indent=2))
    if args.export_sparsity_json:
        root, ext = os.path.splitext(args.export_sparsity_json)
        dense_path = root + "_dense" + ext
        with open(dense_path, "w") as f:
            json.dump(dense_stats, f, indent=2)
        print(f"[saved] per-request sparsity (dense) → {dense_path}")

    print("\n\n## Policy Result")
    print_compact_summary(res_policy)
    print("\n\n## Dense baseline")
    print_compact_summary(res_dense)

if __name__ == "__main__":
    main()
