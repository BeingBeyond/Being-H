# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch.distributed as dist
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple
from safetensors import safe_open
import re
import glob
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    # flash-attn is CUDA-only; inference uses flex_attention and does not need it
    flash_attn_varlen_func = None
from BeingH.model.llm.qwen2.modeling_qwen2 import (
    Qwen2Attention, 
    Qwen2MLP, 
    Qwen2PreTrainedModel, 
    Qwen2RMSNorm, 
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from BeingH.model.llm.qwen2.configuration_qwen2 import Qwen2Config as _Qwen2Config
from BeingH.model.cache_utils.taylorseer import (
    cal_type, taylor_cache_init, derivative_approximation, taylor_formula,
)


torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
# flex_attention = torch.compile(flex_attention) # , dynamic=True, mode='max-autotune'
flex_attention = torch.compile(flex_attention)


class Qwen2Config(_Qwen2Config):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        is_causal=True,
        _attn_implementation="flash_attention_2",
        qk_norm=True,
        layer_module="Qwen2DecoderLayer",
        freeze_und=False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            is_causal=is_causal,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


class PackedAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
    ):
        total_seq_len = packed_sequence.shape[0]
        dtype, device = packed_sequence.dtype, packed_sequence.device
        packed_query_states = torch.zeros((total_seq_len, self.num_heads * self.head_dim), dtype=dtype, device=device)
        packed_key_states = torch.zeros((total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device)
        packed_value_states = torch.zeros((total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device)

        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence)
        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence)
        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        
        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, List):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0), # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output[packed_und_token_indexes])

        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ):
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class PackedAttentionMoT(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()
    
        moe_gen_hidden_size = config.llm_config.expert_config.hidden_size 
        self.q_proj_moe_gen = nn.Linear(moe_gen_hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(moe_gen_hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(moe_gen_hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, moe_gen_hidden_size, bias=False)
       
    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)
        # if self.training:
        #     return self.forward_train(*args, **kwargs)
        # else:
        #     return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        total_seq_len = packed_sequence_und.shape[0] + packed_sequence_gen.shape[0]
        dtype, device = packed_sequence_und.dtype, packed_sequence_und.device
        packed_query_states = torch.zeros((total_seq_len, self.num_heads * self.head_dim), dtype=dtype, device=device)
        packed_key_states = torch.zeros((total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device)
        packed_value_states = torch.zeros((total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device)
        
        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(packed_sequence_gen)

        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(packed_sequence_gen)

        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(packed_sequence_gen)
  
        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
        if self.config.freeze_und:
            packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()
        
        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_query_states_[packed_und_token_indexes] = packed_query_states_[packed_und_token_indexes].detach()
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes])
        
        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_key_states_[packed_und_token_indexes] = packed_key_states_[packed_und_token_indexes].detach()
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )
        
        if isinstance(attention_mask, List):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0), # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)

        #packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        #packed_attn_output_[packed_und_token_indexes] = self.o_proj(packed_attn_output[packed_und_token_indexes])
        #packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes])

        packed_attn_output_und = self.o_proj(packed_attn_output[packed_und_token_indexes])
        packed_attn_output_gen = self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes])

        return packed_attn_output_und, packed_attn_output_gen

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        if mode == 'und':
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        elif mode == 'gen':
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_heads * self.head_dim))
            packed_key_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
            packed_value_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)

            packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)

            packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_vae_token_indexes])

            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if mode == 'und':
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == 'gen':
            packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
            packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        
        residual = packed_sequence_und
        packed_sequence = self.input_layernorm(packed_sequence_und)
        
        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence_und,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes
        )

        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence, packed_sequence_gen

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(
        self, 
        config, 
        layer_idx: Optional[int] = None, 
        attn_module: Optional[Qwen2Attention] = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_und = config.freeze_und
        self.use_mode = config.use_moe
        self.self_attn = attn_module(config, layer_idx)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config.expert_config)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.expert_config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.expert_config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)
        # if self.training:
        #     return self.forward_train(*args, **kwargs)
        # else:
        #     return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
       
        residual_und = packed_sequence_und
        packed_sequence_und_ = self.input_layernorm(packed_sequence_und)
  
        residual_gen = packed_sequence_gen
        packed_sequence_gen_ = self.input_layernorm_moe_gen(packed_sequence_gen)

        # Self Attention
        packed_sequence_und_, packed_sequence_gen_ = self.self_attn(
            packed_sequence_und=packed_sequence_und_,
            packed_sequence_gen=packed_sequence_gen_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )

        if self.freeze_und:
            packed_sequence_und_ = packed_sequence_und_.detach()
        
        packed_sequence_und = residual_und + packed_sequence_und_
        packed_sequence_gen = residual_gen + packed_sequence_gen_
    
        residual_und = packed_sequence_und
        residual_gen = packed_sequence_gen

        packed_sequence_und_ = packed_sequence_und.new_zeros(packed_sequence_und.shape)
        packed_sequence_gen_ = packed_sequence_gen.new_zeros(packed_sequence_gen.shape)

        packed_sequence_und_ = self.mlp(self.post_attention_layernorm(packed_sequence_und))
        if self.freeze_und:
            packed_sequence_und_ = packed_sequence_und_.detach()
    
        packed_sequence_gen_ = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(packed_sequence_gen)
        )
     
        packed_sequence_und = residual_und + packed_sequence_und_
        packed_sequence_gen = residual_gen + packed_sequence_gen_
      
        return packed_sequence_und, packed_sequence_gen

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        
        enable_taylorseer = getattr(self, 'enable_taylorseer', False)

        if enable_taylorseer and self.current['type'] == 'full':
            self.current['module'] = 'total'
            taylor_cache_init(cache_dic=self.cache_dic, current=self.current)

        if not enable_taylorseer or (enable_taylorseer and self.current['type'] == 'full'):
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = self.input_layernorm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_

            # Self Attention
            packed_query_sequence, past_key_values = self.self_attn(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )
            packed_query_sequence = residual + packed_query_sequence

            # Fully Connected
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
                packed_query_sequence = self.mlp(packed_query_sequence)
            elif mode == "gen":
                packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
                packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
                packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
                packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(torch.bfloat16)

                packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
                packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
                packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
                packed_query_sequence = packed_query_sequence_

            packed_query_sequence = residual + packed_query_sequence
        
        if enable_taylorseer:
            if self.current['type'] == 'full':
                derivative_approximation(cache_dic=self.cache_dic, current=self.current, feature=packed_query_sequence)
            elif self.current['type'] == 'Taylor':
                self.current['module'] = 'total'
                packed_query_sequence = taylor_formula(cache_dic=self.cache_dic, current=self.current)

        return packed_query_sequence, past_key_values


class Qwen2MoEDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)

        packed_sequence_new = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_und = self.mlp(packed_sequence[packed_und_token_indexes])
        packed_sequence_gen = self.mlp_moe_gen(packed_sequence[packed_gen_token_indexes])
        packed_sequence_new[packed_und_token_indexes] = packed_sequence_und
        packed_sequence_new[packed_gen_token_indexes] = packed_sequence_gen

        packed_sequence = residual + packed_sequence_new

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        if mode == "und":
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_query_sequence[packed_vae_token_indexes])
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


Decoder_layer_dict = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoEDecoderLayer": Qwen2MoEDecoderLayer,
    "Qwen2MoTDecoderLayer": partial(Qwen2MoTDecoderLayer, attn_module=PackedAttentionMoT),
}


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.use_moe = config.use_moe
        
        layer_module = Decoder_layer_dict[config.layer_module]

        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = Qwen2RMSNorm(config.expert_config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)
        # if self.training:
        #     return self.forward_train(*args, **kwargs)
        # else:
        #     return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        if self.config.freeze_und:
            #packed_sequence[packed_und_token_indexes] = packed_sequence[packed_und_token_indexes].detach()
            packed_sequence_und = packed_sequence_und.detach()

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_sequence_und, packed_position_ids.unsqueeze(0))   # normal order
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)
        
        # del cos, sin
        
        extra_inputs = {}
        if self.use_moe:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )
        else:
            extra_inputs.update(packed_und_token_indexes=packed_und_token_indexes)

        for decoder_layer in self.layers:
            packed_sequence_und, packed_sequence_gen = decoder_layer(
                packed_sequence_und=packed_sequence_und,
                packed_sequence_gen=packed_sequence_gen,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs
            )

        if self.use_moe:
            dtype, device = packed_sequence_und.dtype, packed_sequence_und.device
            # packed_sequence_ = torch.zeros((packed_sequence_und.shape[0]+packed_sequence_gen.shape[0], packed_sequence_und.shape[1]), dtype=dtype, device=device)
            packed_sequence_und_ = torch.zeros_like(packed_sequence_und)
            packed_sequence_gen_ = torch.zeros_like(packed_sequence_gen)

            # 0 1 2 3                             3 4 5
            # packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence_und)
            packed_sequence_und_ = self.norm(packed_sequence_und)

            if self.config.freeze_und:
                packed_sequence_und_ = packed_sequence_und_.detach()
            packed_sequence_gen_ = self.norm_moe_gen(packed_sequence_gen)
            return packed_sequence_und_, packed_sequence_gen_
        else:
            assert packed_sequence_gen.shape[0]==0
            return self.norm(packed_sequence_und), packed_sequence_gen

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        
        enable_taylorseer = getattr(self, 'enable_taylorseer', False)
        if enable_taylorseer:
            cal_type(self.cache_dic, self.current)
            self.current['stream'] = 'layers_stream'

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == 'gen':
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        for layer_idx, decoder_layer in enumerate(self.layers):
            if enable_taylorseer:
                decoder_layer.current = self.current
                decoder_layer.cache_dic = self.cache_dic
                decoder_layer.enable_taylorseer = True
                self.current['layer'] = layer_idx
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)
        
        if enable_taylorseer:
            self.current['step'] += 1

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


def get_layer_mapping_strategy(L, K):
    """Create mapping strategy from L layers to K layers. L: current model layers, K: expert model layers"""
    assert L >= K
    mapping = []

    if L <= K * 1.5: # Light expansion: primarily interpolation strategy
        for i in range(L):
            # Calculate the corresponding floating-point position in expert
            expert_pos = i * (K - 1) / (L - 1) if L > 1 else 0
            mapping.append(expert_pos)
    else: # Heavy expansion: segmented copy strategy
        # First 1/3 uniformly distributed
        section1_size = L // 3
        for i in range(section1_size):
            expert_pos = i * (K - 1) / (section1_size - 1) if section1_size > 1 else 0
            mapping.append(expert_pos)
        # Middle 1/3 densely distributed
        section2_size = L // 3
        for i in range(section2_size):
            expert_pos = (K - 1) * 0.3 + i * (K - 1) * 0.4 / (section2_size - 1) if section2_size > 1 else (K - 1) * 0.5
            mapping.append(expert_pos)
        # Last 1/3 to the end
        section3_size = L - section1_size - section2_size
        for i in range(section3_size):
            expert_pos = (K - 1) * 0.7 + i * (K - 1) * 0.3 / (section3_size - 1) if section3_size > 1 else (K - 1) * 0.85
            mapping.append(expert_pos)

    return mapping


def interpolate_layer_params(expert_state_dict, expert_pos, param_name_template, num_expert_layers):
    """Interpolate parameters between two expert layers"""
    floor_idx = int(expert_pos)
    floor_param_name = param_name_template.format(floor_idx)
    if int(expert_pos)-expert_pos==0:
        return expert_state_dict[floor_param_name]

    ceil_idx = min(floor_idx + 1, num_expert_layers-1)  
    alpha = expert_pos - floor_idx
    ceil_param_name = param_name_template.format(ceil_idx)
    
    if floor_param_name in expert_state_dict and ceil_param_name in expert_state_dict:
        floor_param = expert_state_dict[floor_param_name]
        ceil_param = expert_state_dict[ceil_param_name]
        
        interpolated = (1 - alpha) * floor_param + alpha * ceil_param
        if alpha > 0:  
            noise_scale = 0.01 * alpha 
            noise = torch.randn_like(interpolated) * noise_scale
            interpolated += noise
            
        return interpolated
    elif floor_param_name in expert_state_dict:
        return expert_state_dict[floor_param_name]
    else:
        raise NotImplementedError
    

class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.llm_layers = config.num_hidden_layers
        self.use_moe = config.use_moe
        if self.use_moe:
            self.expert_layers = config.expert_config.num_hidden_layers
            self.llm_identical = config.hidden_size==config.expert_config.hidden_size
        self.post_init()

    def custom_init_pretrained(self, mllm_state_dict, logger):
        print("Initializing LLM decoder from MLLM weights") 
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                continue
            
            original_name = "language_model."+name.replace("_und", "") if "norm_und" in name \
                            else "language_model."+name
            try:
                param.data.copy_(mllm_state_dict.pop(original_name).data)
            except:
                logger.info(f"'{name}' not initialized")
        return mllm_state_dict

    def init_pretrained(self, mllm_state_dict):
        print("Initializing LLM decoder from MLLM weights") 
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                continue
            
            original_name = "language_model."+name.replace("_und", "") if "norm_und" in name \
                            else "language_model."+name

            # breakpoint()
            # param.data.copy_(mllm_state_dict.pop(original_name).data)
            try:
                param.data.copy_(mllm_state_dict.pop(original_name).data)
            except:
                print(f"'{name}' not initialized")
                # breakpoint()
        return mllm_state_dict

    def init_expert(self, expert_path):
        """initialize parameters of action expert using the pretrained llm ckpt"""
        safetensor_files = glob.glob(f"{expert_path}/*.safetensors")
        expert_state_dict = dict()
        for file_path in safetensor_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    expert_state_dict[key] = f.get_tensor(key)

        layer_mapping = get_layer_mapping_strategy(self.llm_layers, self.expert_layers)
        
        for name, param in self.named_parameters():
            if "moe_gen" not in name:
                continue
            original_name = name.replace("_moe_gen", "")  
            
            if any(n in name for n in ["q_norm_moe_gen", "k_norm_moe_gen"]):      
                param.data.copy_(self.state_dict()[original_name].data)    
            elif any(proj in name for proj in ["q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"]):
                if self.llm_identical and not expert_path:
                    param.data.copy_(expert_state_dict[original_name].data)
            else:
                layer_match = re.search(r'layers\.(\d+)\.', name)
                if layer_match:
                    curr_layer_idx = int(layer_match.group(1))
                    if curr_layer_idx < len(layer_mapping):
                        expert_pos = layer_mapping[curr_layer_idx]
                        original_name_template = name.replace("_moe_gen", "").replace(f"layers.{curr_layer_idx}.", "layers.{}.") # Build parameter name template
                 
                        interpolated_param = interpolate_layer_params(
                            expert_state_dict, expert_pos, original_name_template,
                            num_expert_layers=self.expert_layers
                        )

                        if interpolated_param is not None:
                            param.data.copy_(interpolated_param)
                        else:
                            fallback_idx = min(int(expert_pos), self.expert_layers-1)
                            fallback_name = original_name_template.format(fallback_idx)
                            if fallback_name in expert_state_dict:
                                param.data.copy_(expert_state_dict[fallback_name])
                                print(f"Layer {curr_layer_idx}: fallback to expert layer {fallback_idx}")
                else:
                    if original_name in expert_state_dict:
                        param.data.copy_(expert_state_dict[original_name].data)

    def init_moe(self):
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)
        # if self.training:
        #     return self.forward_train(*args, **kwargs)
        # else:
        #     return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        outputs = self.model(
            packed_sequence_und=packed_sequence_und,
            packed_sequence_gen=packed_sequence_gen,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        outputs = self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )

        return outputs
