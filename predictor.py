import os
import math
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from typing import Tuple, Optional, List
from typing import Any
from dataclasses import dataclass

@dataclass
class PolicyState:
    """Holds policy KV cache and the current policy step index."""
    past_k: List[torch.Tensor]
    past_v: List[torch.Tensor]
    step_idx: torch.LongTensor
    last_action: torch.LongTensor

class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, H, T, Dh]
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, Dh] -> [B, T, D]
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        x: torch.Tensor,                         # [B, T, D]
        past_k: Optional[torch.Tensor] = None,   # [B, H, Lp, Dh]
        past_v: Optional[torch.Tensor] = None,   # [B, H, Lp, Dh]
        use_cache: bool = False,
    ):
        B, T, D = x.shape
        q = self._split_heads(self.q_proj(x))    # [B, H, T, Dh]
        k = self._split_heads(self.k_proj(x))    # [B, H, T, Dh]
        v = self._split_heads(self.v_proj(x))    # [B, H, T, Dh]

        if past_k is not None and past_v is not None:
            k_all = torch.cat([past_k, k], dim=2)   # [B,H,Lp+T,Dh]
            v_all = torch.cat([past_v, v], dim=2)   # [B,H,Lp+T,Dh]
            past_len = past_k.size(2)
        else:
            k_all, v_all = k, v
            past_len = 0

        L = k_all.size(2)
        # mask shape: [T, L] True => masked
        i = torch.arange(T, device=x.device).unsqueeze(1)
        j = torch.arange(L, device=x.device).unsqueeze(0)
        causal_mask = j > (past_len + i)

        attn_scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.d_head)   # [B,H,T,L]
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        y = torch.matmul(attn_probs, v_all)      # [B,H,T,Dh]
        y = self._merge_heads(y)                 # [B,T,D]
        y = self.o_proj(y)                       # [B,T,D]

        present_k = k_all if use_cache else None
        present_v = v_all if use_cache else None
        if present_k is not None:
            present_k = present_k.detach()
            present_v = present_v.detach()
        return y, present_k, present_v

class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                         # [B,T,D]
        past_k: Optional[torch.Tensor] = None,   # [B,H,Lp,Dh]
        past_v: Optional[torch.Tensor] = None,   # [B,H,Lp,Dh]
        use_cache: bool = False,
    ):
        h, pk, pv = self.attn(self.ln1(x), past_k=past_k, past_v=past_v, use_cache=use_cache)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x, pk, pv

class RecurrentActorCriticPolicy(nn.Module):
    """
    Recurrent actor-critic built as a tiny Transformer with its own KV cache and learned positions.

    Inputs per step t (all LM-derived tensors are .detach()'d by caller):
      - h_{t-1}  : LM last hidden from previous token         [B, H_lm]
      - e(x_t)   : embedding of current input token           [B, E_lm]
      - scalars  : 8D vector with:
            [0] t_frac            \in [0,1]
            [1] eff_flag          \in {0,1}
            [2] lambda_keep
            [3] lambda_prune
            [4] lambda_quant
            [5] dev_keep          = mean_keep_prev   - C_tok
            [6] dev_prune         = mean_prune_prev  - C_pru
            [7] dev_qratio        = mean_qratio_prev - C_q
      - a_{t-1}  : previous action id                          [B]
    """
    def __init__(
        self,
        h_dim: int,
        e_dim: int,
        n_actions: int,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 2,
        mlp_ratio: float = 4.0,
        action_dim: int = 32,
        max_len: int = 4096,
        dropout: float = 0.0,
        scalar_dim: int = 8,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.scalar_dim = scalar_dim
        # We only *use* a subset of the scalar features internally:
        #   [0] t_frac, [1] eff_flag, [5] dev_keep, [6] dev_prune, [7] dev_qratio
        # Lambdas [2:5] are ignored here, so callers can keep passing the original 8D vector.
        kept_scalar_idx = [i for i in (0, 1, 5, 6, 7) if i < scalar_dim]
        self.register_buffer(
            "scalar_keep_idx",
            torch.tensor(kept_scalar_idx, dtype=torch.long),
            persistent=False,
        )
        self.start_action_id = n_actions  # reserve BOS action id
        self.action_emb = nn.Embedding(n_actions + 1, action_dim)  # +1 for BOS
        in_dim = h_dim + e_dim + scalar_dim + action_dim  # will be overwritten below if scalar_proj=True
        self.obs_proj = nn.Linear(in_dim, d_model)
        scalar_proj = True
        if scalar_proj:
            self.action_emb = nn.Embedding(n_actions + 1, action_dim)  # +1 for BOS
            eff_scalar_dim = len(kept_scalar_idx)
            self.scalar_proj = nn.Sequential(
                nn.Linear(eff_scalar_dim, 64),
                nn.GELU(),
                nn.LayerNorm(64),
            )
            scalar_proj_dim = 64
            in_dim = h_dim + e_dim + scalar_proj_dim + action_dim
            self.obs_proj = nn.Linear(in_dim, d_model)
        else:
            self.scalar_proj = nn.Identity()
        self.pos_emb = nn.Embedding(max_len, d_model)              # learned absolute positions
        self.blocks = nn.ModuleList([_TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.pi = nn.Linear(d_model, n_actions)
        self.v  = nn.Linear(d_model, 1)
        nn.init.zeros_(self.pi.bias); nn.init.zeros_(self.v.bias)

    def init_state(self, B: int, device: torch.device) -> PolicyState:
        past_k = [None] * len(self.blocks)
        past_v = [None] * len(self.blocks)
        step_idx = torch.zeros(B, dtype=torch.long, device=device)
        last_action = torch.full((B,), self.start_action_id, dtype=torch.long, device=device)
        return PolicyState(past_k=past_k, past_v=past_v, step_idx=step_idx, last_action=last_action)

    def _form_step_inputs(
        self,
        h_lm: torch.Tensor,         # [B,H_lm]   (detached)
        e_tok: torch.Tensor,        # [B,E_lm]   (detached)
        scalars: torch.Tensor,      # [B, scalar_proj_dim] after projection
        prev_action: torch.LongTensor,  # [B]
        pos_idx: torch.LongTensor,      # [B]
    ) -> torch.Tensor:
        a_emb = self.action_emb(prev_action)               # [B, A_dim]
        x = torch.cat([h_lm, e_tok, scalars, a_emb], -1)   # [B, H+E+S_used+A_dim]
        x = self.obs_proj(x).unsqueeze(1)                  # [B,1,D]
        x = x + self.pos_emb(pos_idx).unsqueeze(1)         # add learned position
        return x

    def step(
        self,
        h_lm: torch.Tensor,
        e_tok: torch.Tensor,
        scalars: torch.Tensor,
        state: PolicyState,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, PolicyState]:
        B = h_lm.size(0)
        # Drop lambda_* dimensions inside the policy; keep only
        # t_frac, eff_flag, dev_keep, dev_prune, dev_qratio.
        if self.scalar_keep_idx is not None:
            scalars_used = torch.index_select(
                scalars, dim=-1, index=self.scalar_keep_idx.to(scalars.device)
            )
        else:
            scalars_used = scalars
        x = self._form_step_inputs(
            h_lm,
            e_tok,
            self.scalar_proj(scalars_used),
            state.last_action,
            state.step_idx,
        )
        # x = self._form_step_inputs(h_lm, e_tok, self.scalar_proj(scalars), state.last_action, state.step_idx)
        past_k = state.past_k
        past_v = state.past_v
        new_k, new_v = [], []
        h = x
        for li, block in enumerate(self.blocks):
            h, pk, pv = block(h, past_k=past_k[li], past_v=past_v[li], use_cache=True)
            new_k.append(pk)
            new_v.append(pv)
        h = self.ln_f(h)                        # [B,1,D]
        h_last = h.squeeze(1)                   # [B,D]
        logits = self.pi(h_last) / max(1e-6, temperature)
        value  = self.v(h_last).squeeze(-1)     # [B]
        next_state = PolicyState(past_k=new_k, past_v=new_v, step_idx=state.step_idx + 1, last_action=state.last_action)
        return logits, value, next_state

    @torch.no_grad()
    def _positions(self, T: int, B: int, device) -> torch.LongTensor:
        return torch.arange(T, device=device).view(T, 1).expand(T, B).contiguous()  # [T,B]

    def forward_sequence(
        self,
        h_seq: torch.Tensor,            # [T,B,H_lm]  detached
        e_seq: torch.Tensor,            # [T,B,E_lm]  detached
        scalars_seq: torch.Tensor,      # [T,B,scalar_dim]
        prev_actions_seq: torch.Tensor, # [T,B]  (a_{t-1} for each t)
        tbptt_k: int = 0,               # kept for API compatibility; ignored
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full-episode causal Transformer over the rollout.

        We ignore tbptt_k here and simply run a standard decoder-style stack:
        all T steps in one shot with a causal mask. This is easier to reason about
        than chunked TBPTT and matches the mental model "one transformer per rollout".
        """
        T, B, _ = h_seq.shape
        device = h_seq.device
        pos = self._positions(T, B, device)  # [T,B]
        # Same scalar subset as in step(): drop the lambda_* entries.
        if self.scalar_keep_idx is not None:
            scalars_used = torch.index_select(
                scalars_seq, dim=-1, index=self.scalar_keep_idx.to(scalars_seq.device)
            )
        else:
            scalars_used = scalars_seq

        x = torch.cat(
            [
                h_seq,                               # [T,B,H_lm]
                e_seq,                               # [T,B,E_lm]
                self.scalar_proj(scalars_used),      # [T,B,scalar_proj_dim]
                self.action_emb(prev_actions_seq),   # [T,B,A_dim]
            ],
            dim=-1,
        )        
        x = self.obs_proj(x) + self.pos_emb(pos)    # [T,B,D_model]
        x = x.transpose(0, 1).contiguous()          # [B,T,D_model]

        h = x
        for block in self.blocks:
            h, _, _ = block(h, past_k=None, past_v=None, use_cache=False)
        h = self.ln_f(h)                             # [B,T,D_model]

        logits = self.pi(h) / max(1e-6, temperature) # [B,T,A]
        values = self.v(h).squeeze(-1)               # [B,T]

        return logits.transpose(0, 1), values.transpose(0, 1)