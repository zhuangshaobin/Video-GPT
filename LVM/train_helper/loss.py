import torch
import random
import numpy as np
from PIL import Image
from copy import deepcopy
import torch.distributed as dist
from LVM.utils import broadcast_data, broadcast_pickle
from LVM.acceleration.parallel_states import hccl_info


count = 0


def is_all_equal(data):
    """
    判断 `data` 中所有元素是否相等（列表或 PyTorch 张量）。
    
    对于列表，会检查所有元素是否与第一个元素相同；
    对于 PyTorch Tensor，会检查在第 0 维（batch 维度）上的所有样本是否相同。
    """
    if isinstance(data, list):
        # 如果是列表，逐元素与第一个元素比较
        if not data:  # 空列表，返回 True 或者根据需求自行处理
            return True
        if all(isinstance(x, torch.Tensor) for x in data):
            first = data[0]
            all_same = all(torch.equal(x, first) for x in data)
            if all_same:
                return True
            else: 
                max_diff = 0.0
                n = len(data)
                for i in range(n):
                    for j in range(i + 1, n):
                        # 绝对值差的最大值
                        diff = torch.max(torch.abs(data[i] - data[j]))
                        # 更新全局最大值
                        if diff.item() > max_diff:
                            max_diff = diff.item()
                print(f"[is_all_equal] Tensors are not all equal. Max abs diff = {max_diff}")
                return False
        else:
            return all(x == data[0] for x in data)
    
    elif isinstance(data, torch.Tensor):
        # 如果是 PyTorch 张量，在 batch 维度（第 0 维）上进行比较
        # 先确保 batch 大小大于 1
        if data.shape[0] < 2:
            return True
        # 比较张量在第 0 维上的所有样本是否与第一个样本相同
        return torch.all(data == data[0])
    
    else:
        raise TypeError("只支持列表或 PyTorch Tensor 类型")


def recursive_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: recursive_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [recursive_to_device(element, device) for element in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to_device(element, device) for element in data)
    elif isinstance(data, set):
        return {recursive_to_device(element, device) for element in data}
    else:
        # 非张量，直接返回
        return data


def sample_x0(x1):
    """Sampling x0 & t based on shape of x1 (if needed)
    Args:
      x1 - data point; [batch, *dim]
    """
    # SEED = 0
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)


    if isinstance(x1, (list, tuple)):
        x0 = [torch.randn_like(img_start) for img_start in x1]
    else:
        x0 = torch.randn_like(x1)

    return x0


def sample_timestep(x1):
    t = torch.rand(len(x1))
    t = t.to(x1[0])
    return t


def sample_exp_timestep(x1):
    u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
    t = 1 / (1 + torch.exp(-u))
    t = t.to(x1[0])
    return t


def sample_frame_block_timestep(x1, frame_blocks):
    t = []
    for b_inx in frame_blocks.keys():
        for k, frame_block in enumerate(frame_blocks[b_inx]):
            t.extend([random.random()] * frame_block)
    t = torch.tensor(t)
    assert len(t) == len(x1)
    t = t.to(x1[0])
    return t


def sample_timestep_max_noise(x1, max_noise_level=0):
    t = max_noise_level + (1 - max_noise_level) * torch.rand(len(x1))
    t = t.to(x1[0])
    return t


def sample_timestep_fix_max_noise(x1, max_noise_level=0):
    t = max_noise_level + (1 - max_noise_level) * torch.zeros(len(x1))
    t = t.to(x1[0])
    return t


def training_losses_x1_noise_input(
    model, 
    x1, 
    model_kwargs=None, 
    snr_type='uniform', 
    patch_weight=None, 
    input_noise=0.9, 
    cls_weight=None, 
    order=None, 
    frame_blocks=None,
    exp_time=False,
    device=None,
    ):
    """Loss for training torche score model
    Args:
    - model: backbone model; could be score, noise, or velocity
    - x1: datapoint
    - model_kwargs: additional arguments for torch model
    """
    with torch.no_grad():
        if model_kwargs == None:
            model_kwargs = {}
        x1 = broadcast_data(x1, src=dist.get_rank()-hccl_info.rank,  group=hccl_info.group, device=device)

        B = len(x1)
        B_input = len(model_kwargs["input_img_latents"])

        x0 = sample_x0(x1)
        if frame_blocks is None:
            if exp_time:
                t = sample_exp_timestep(x1)
            else:
                t = sample_timestep(x1)
        else:
            t = sample_frame_block_timestep(x1, frame_blocks)

        x0_input = sample_x0(model_kwargs["input_img_latents"])
        if len(model_kwargs["input_img_latents"]) > 0:
            t_input = sample_timestep_max_noise(model_kwargs["input_img_latents"], max_noise_level=input_noise)

        x0 = broadcast_data(x0, src=dist.get_rank()-hccl_info.rank,  group=hccl_info.group, device=device)
        t = broadcast_data(t, src=dist.get_rank()-hccl_info.rank,  group=hccl_info.group, device=device)
        x0_input = broadcast_data(x0_input, src=dist.get_rank()-hccl_info.rank,  group=hccl_info.group, device=device)
        if len(model_kwargs["input_img_latents"]) > 0:
            t_input = broadcast_data(t_input, src=dist.get_rank()-hccl_info.rank,  group=hccl_info.group, device=device)

        if isinstance(x1, (list, tuple)):
            xt = [t[i] * x1[i] + (1 - t[i]) * x0[i] for i in range(B)]
            ut = [x1[i] for i in range(B)]
        else:
            dims = [1] * (len(x1.size()) - 1)
            t_ = t.view(t.size(0), *dims)
            xt = t_ * x1 + (1 - t_) * x0
            ut = x1

        if len(model_kwargs["input_img_latents"]) > 0:
            input_latents_gt = model_kwargs["input_img_latents"].copy()
            if isinstance(model_kwargs["input_img_latents"], (list, tuple)):
                model_kwargs["input_img_latents"] = [t_input[i] * model_kwargs["input_img_latents"][i] + (1 - t_input[i]) * x0_input[i] for i in range(B_input)]
                print("add noise to input", flush=True)
            else:  
                dims = [1] * (len(model_kwargs["input_img_latents"].size()) - 1)
                t_input_ = t_input.view(t_input.size(0), *dims)
                model_kwargs["input_img_latents"] = t_input_ * model_kwargs["input_img_latents"] + (1 - t_input_) * x0_input
                print("add noise to input", flush=True)

    if not model_kwargs.get("input_output_return", False):
        model_output = model(xt, t, **model_kwargs)
    else:
        model_output, input_latents_pred = model(xt, t, **model_kwargs)

    terms = {}
    if isinstance(x1, (list, tuple)):
        assert len(model_output) == len(ut) == len(x1)
        if patch_weight is not None:
            terms["loss"] = th.stack(
            [((ut[i] - model_output[i]) ** 2 * patch_weight[i]).mean() for i in range(B)],
            dim=0,
            )
        else:
            loss_terms = []
            for i in range(B):
                scale = 1
                if order is not None:
                    with torch.no_grad():
                        scale = ((ut[i] - model_output[i]) ** 2).mean() / ((ut[i] - model_output[i]) ** order).mean()
                else:
                    order = 2
                loss_term = scale * ((ut[i] - model_output[i]) ** order).mean()
                loss_terms.append(loss_term)
            terms["loss"] = torch.stack(loss_terms, dim=0)

            if model_kwargs.get("input_output_return", False):
                terms["input_loss"] = torch.stack(
                    [((input_latents_gt[i] - input_latents_pred[i]) ** 2).mean() for i in range(B_input)],
                    dim=0,
                    )
                terms["loss"] = torch.cat([terms["loss"], terms["input_loss"]], dim=0)

            if cls_weight is not None:
                B_cls = B - B_input
                seq_len = B / B_cls
                for cls_idx in range(B_cls):
                    idx = int(cls_idx * seq_len)
                    terms["loss"][idx] = terms["loss"][idx] * cls_weight
                    
            print(terms["loss"], flush=True)
    else:
        if patch_weight is not None:
            loss = (model_output - ut) ** 2
            loss = loss * patch_weight
            terms["loss"] = mean_flat(loss)
        else:
            terms["loss"] = mean_flat(((model_output - ut) ** 2))

    return terms


def mean_flat(x):
    """
    Take torche mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))
