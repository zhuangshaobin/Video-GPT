import math
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    torch_npu = None
import time
import deepspeed
import numpy as np
from torch import Tensor
from types import MethodType
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union, Any
from LVM.acceleration.parallel_states import  hccl_info
from deepspeed.sequence.layer import DistributedAttention, _SeqAllToAll
from transformers.models.phi3.modeling_phi3 import Phi3SdpaAttention, apply_rotary_pos_emb, repeat_kv


COUNT = 0


def new_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Phi3Model is using Phi3SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.size()
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # B N S D
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # B N S D
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # B N S D

        kv_seq_len = key_states.shape[-2] * hccl_info.world_size
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=5000)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()
        attn_output = self.dist_attn(
            query_states,
            key_states,
            value_states,
            0,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
            
        return attn_output, None, past_key_value


def new_attn_forward(self, query: Tensor, key: Tensor, value: Tensor, batch_dim_idx: int, *args: Any, **kwargs) -> Tensor:
    """ forward

    Arguments:
        query (Tensor): query input to the layer
        key (Tensor): key input to the layer
        value (Tensor): value input to the layer
        batch_dim_idx (int): indicating which dim is batch
        args: other args

    Returns:
        * output (Tensor): context output
    """

    # TODO Merge three alltoall calls into one
    # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
    #in shape : e.g.,  [s/p:h:]

    def bwd_hook(layer_type):

        def pre_hook_fun(grad):
            type = 'd' + layer_type
            self.overlap_handles[type + '_work'].wait()
            self.sp_stream.wait_stream(self.dafult_stream)
            all2all_output = self.overlap_handles[type + '_grad']
            grad = list(grad)
            grad[0] = self.overlap_handles[type + '_post_all2all_func'](all2all_output)
            grad = tuple(grad)

        return pre_hook_fun

    self.layer_sync(query)
    query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx, batch_dim_idx, None,
                                        self.overlap_handles, 'q')
    self.layer_sync(key)
    key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx, batch_dim_idx, None,
                                    self.overlap_handles, 'k')
    if self.sp_overlap_comm:
        self.dafult_stream.wait_stream(self.sp_stream)

    value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx, batch_dim_idx, None,
                                        self.overlap_handles, 'v')

    if self.sp_overlap_comm:
        # Register a hook to synchronize dq and dk after the all-to-all
        # operation when the gradient data is used.
        # Place this logic after the q, k, v all-to-all operation to
        # improve interpreter speed to
        # call and launch of the forward all-to-all communication.
        grad_fn_q = query.grad_fn.next_functions[0][0]
        grad_fn_q.register_prehook(bwd_hook(layer_type='q'))
        grad_fn_k = key.grad_fn.next_functions[0][0]
        grad_fn_k.register_prehook(bwd_hook(layer_type='k'))

    #out shape : e.g., [s:h/p:]
    head_num = query_layer.shape[2]
    scale_factor = 1 / math.sqrt(query_layer.size(-1))
    context_layer = self.local_attn(
        query_layer, 
        key_layer, 
        value_layer, 
        head_num, 
        "BSND", 
        keep_prob=1.0, 
        atten_mask=kwargs['attn_mask'],
        scale=scale_factor,
        )[0]

    output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx, batch_dim_idx,
                                self.sp_stream, self.overlap_handles, 'o')

    #out e.g., [s/p::h]
    return output


def new_simple_attn_forward(self, query: Tensor, key: Tensor, value: Tensor, batch_dim_idx: int, *args: Any, **kwargs) -> Tensor:
    """ forward

    Arguments:
        query (Tensor): query input to the layer
        key (Tensor): key input to the layer
        value (Tensor): value input to the layer
        batch_dim_idx (int): indicating which dim is batch
        args: other args

    Returns:
        * output (Tensor): context output
    """

    query_layer = query
    key_layer = key
    value_layer = value

    #out shape : e.g., [s:h/p:]
    head_num = query_layer.shape[2]
    scale_factor = 1 / math.sqrt(query_layer.size(-1))
    context_layer = self.local_attn(
        query_layer, 
        key_layer, 
        value_layer, 
        head_num, 
        "BSND", 
        keep_prob=1.0, 
        atten_mask=kwargs['attn_mask'],
        scale=scale_factor,
        sparse_mode=1,
        # inner_precise=2,
        )[0]

    #out e.g., [s/p::h]
    return context_layer


def replace_attention(model):
    for module in model.modules():
        if isinstance(module, Phi3SdpaAttention): 
            module.forward = MethodType(new_forward, module)
            module.local_attn = torch_npu.npu_fusion_attention
            module.dist_attn = DistributedAttention(module.local_attn, hccl_info.group)
            module.dist_attn.forward = MethodType(new_attn_forward, module.dist_attn)
    
    return model


def replace_simple_attention(model):
    for module in model.modules():
        if isinstance(module, Phi3SdpaAttention): 
            module.forward = MethodType(new_forward, module)
            module.local_attn = torch_npu.npu_fusion_attention
            module.dist_attn = DistributedAttention(module.local_attn, hccl_info.group)
            module.dist_attn.forward = MethodType(new_simple_attn_forward, module.dist_attn)
    
    return model