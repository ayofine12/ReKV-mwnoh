import copy
import torch
from typing import Optional

from .kv_cache_manager import ContextManager
from .dot_production_attention import get_multi_stage_dot_production_attention
from model.profiling import profile_section


def rekv_attention_forward(
    n_local, n_init, topk, chunk_size,
    block_size, max_cached_block,
    exc_block_size, fattn,
    kv_repr="mean",
    q_repr="mean",
    q_token_agg="topk",
    q_topk_ratio=0.3,
    k_token_agg="max",
    k_topk_ratio=0.3,
    head_specific_retrieval=False,
    retrieval_fusion="none",
    fusion_mean_topk=None,
    fusion_token_topk=None,
    async_global_stream=True,
    pin_memory=False,
    *args, **kwargs
):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)
    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv,
    ):

        with profile_section("attention.rekv_forward"):
            """ 1. Project QKV """
            batch_size = query.size(0)
            len_q = query.size(1)
            len_k = key_value.size(1)

            assert use_cache

            h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
            h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
            h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

            h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
            h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
            h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

            if position_bias._cos_cached is not None and position_bias._cos_cached.device != h_q.device:
                position_bias = copy.deepcopy(position_bias)
                if position_bias.inv_freq.device != h_q.device:
                    position_bias.inv_freq = position_bias.inv_freq.to(h_q.device)
                if position_bias._cos_cached is not None:
                    position_bias._cos_cached = position_bias._cos_cached.to(h_q.device)
                if position_bias._sin_cached is not None:
                    position_bias._sin_cached = position_bias._sin_cached.to(h_q.device)

            if past_key_value is None:
                past_key_value = ContextManager(
                    position_bias,
                    n_init, n_local,
                    block_size, max_cached_block, topk, chunk_size, exc_block_size,
                    kv_repr=kv_repr,
                    q_repr=q_repr,
                    q_token_agg=q_token_agg,
                    q_topk_ratio=q_topk_ratio,
                    k_token_agg=k_token_agg,
                    k_topk_ratio=k_topk_ratio,
                    head_specific_retrieval=head_specific_retrieval,
                    retrieval_fusion=retrieval_fusion,
                    fusion_mean_topk=fusion_mean_topk,
                    fusion_token_topk=fusion_token_topk,
                    fattn=fattn,
                    async_global_stream=async_global_stream,
                    pin_memory=pin_memory,
                )

            local_q, local_k, local_v = h_q, h_k, h_v
            global_q, global_k, global_v = h_q, h_k, h_v

            # NOTE: Question-answering, fall back to sliding-window attention (infinite_lm)
            if type(past_key_value) is not ContextManager or past_key_value.to_retrieve:
                if type(past_key_value) is ContextManager:  # retrieval
                    if past_key_value.retrieved_block_indices is None:  # retrieve based on global_q (question's query)
                        past_k, past_v = past_key_value.get_retrieved_kv(global_q)
                    else:  # retrieve based on pre-computed retrieved_block_indices
                        past_k, past_v = past_key_value.get_retrieved_kv()
                    updata_kv_cache = False
                else:  # sliding-window attention
                    past_k = past_key_value[0]
                    past_v = past_key_value[1]
                    updata_kv_cache = True

                h_k = torch.cat([past_k, h_k], dim=-2)
                h_v = torch.cat([past_v, h_v], dim=-2)
                len_k += past_k.shape[2]

                if updata_kv_cache:
                    if len_k <= n_local + n_init:
                        h_k_cache = h_k
                        h_v_cache = h_v
                    else:
                        h_k_cache = torch.cat([h_k[:, :, :n_init, :], h_k[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                        h_v_cache = torch.cat([h_v[:, :, :n_init, :], h_v[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                    current_key_value = (h_k_cache, h_v_cache)
                else:
                    current_key_value = (past_k, past_v)

                h_q_, h_k_, h_v_ = h_q, h_k, h_v
                if len_q + n_local < h_k_.size(-2):
                    h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - n_local:, :]
                    h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - n_local:, :]

                local_h_q, local_h_k = position_bias(h_q_, h_k_)
                local_h_v = h_v_

                if len_k > n_local:
                    init_h_q = position_bias.apply_rotary_pos_emb_one_angle(h_q, n_local)
                    init_h_k = h_k[:, :, :n_init, :].contiguous()
                    init_h_v = h_v[:, :, :n_init, :].contiguous()
                else:
                    init_h_q = h_q
                    init_h_k = torch.empty((batch_size, num_heads_kv, 0, dim_head), device=h_k.device, dtype=h_k.dtype)
                    init_h_v = torch.empty((batch_size, num_heads_kv, 0, dim_head), device=h_v.device, dtype=h_v.dtype)

                with profile_section("attention.kernel"):
                    attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
                    attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local)
                    attn.append(init_h_q, init_h_k, init_h_v, end=True, sliding_window=(len_k - len_q, n_local), complement_sliding_window=True)
                    score, _ = attn.get_result()

                score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
                score = score.reshape(batch_size, len_q, num_heads * dim_head)
                score = attention_out(score)
                return score, current_key_value

            with profile_section("attention.kernel"):
                o = past_key_value.append(
                    local_q, local_k, local_v,
                    global_q, global_k, global_v,
                )
            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            o = o.reshape(batch_size, len_q, dim_head * num_heads)
            o = attention_out(o)
            return o, past_key_value

    return forward
