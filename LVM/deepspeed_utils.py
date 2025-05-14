import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from deepspeed.sequence.layer import DistributedAttention
from transformers.models.phi3.modeling_phi3 import Phi3SdpaAttention


_SEQUENCE_PARALLEL_GROUP = None


def initialize_model_parallel(
    sequence_parallel_size,
):
    world_size = dist.get_world_size()
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    global _SEQUENCE_PARALLEL_GROUP
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        rank = dist.get_rank()
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP


def replace_attention_with_sequence_parallel(sequence_parallel_size, model):
    if not dist.is_initialized():
        deepspeed.init_distributed()
    initialize_model_parallel(sequence_parallel_size)

    # 获取序列并行通信组
    sequence_parallel_group = get_sequence_parallel_group()
    
    # 遍历模型的所有模块
    for module_name, module in model.named_modules():
        if isinstance(module, Phi3SdpaAttention):
            # 获取父模块
            parent_module_name = '.'.join(module_name.split('.')[:-1])
            parent_module = model
            if parent_module_name != '':
                for name in parent_module_name.split('.'):
                    parent_module = getattr(parent_module, name)
            
            # 原始的 self_attn 模块
            original_self_attn = module
            
            # 用 DistributedAttention 包装原始的 self_attn 模块
            wrapped_self_attn = DistributedAttention(original_self_attn, sequence_parallel_group)
            
            # 将包装后的模块替换到模型中
            setattr(parent_module, module_name.split('.')[-1], wrapped_self_attn)