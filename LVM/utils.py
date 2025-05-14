import logging

from PIL import Image
import torch
import torch.distributed as dist
import numpy as np
import pickle
import random
from LVM.acceleration.parallel_states import hccl_info


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)




def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def crop_arr(pil_image, max_image_size):
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    if min(*pil_image.size) < 16:
        scale = 16 / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % 16) // 2
    crop_y2 = arr.shape[0] % 16 - crop_y1

    crop_x1 = (arr.shape[1] % 16) // 2
    crop_x2 = arr.shape[1] % 16 - crop_x1

    arr = arr[crop_y1:arr.shape[0]-crop_y2, crop_x1:arr.shape[1]-crop_x2]    
    return Image.fromarray(arr)



def vae_encode(vae, x, weight_dtype, seed=None, batch_encode=False):
    if batch_encode:
        images = torch.cat(x, dim=0)
        if vae.config.shift_factor is not None:
            latents = vae.encode(images).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            latents = vae.encode(images).latent_dist.sample().mul_(vae.config.scaling_factor)
        latents = latents.to(weight_dtype)
        latents = list(torch.split(latents, 1, dim=0))
    else:
        latents = []
        for image in x:
            if seed is not None:
                np_state = np.random.get_state()
                torch_state = torch.get_rng_state()
                torch_cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                random_state = random.getstate()

                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                random.seed(seed)

            if vae.config.shift_factor is not None:
                latent = vae.encode(image).latent_dist.sample()
                latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
            else:
                latent = vae.encode(image).latent_dist.sample().mul_(vae.config.scaling_factor)
            latent = latent.to(weight_dtype)
            latents.append(latent)

            if seed is not None:
                random.setstate(random_state)
                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                if torch.cuda.is_available() and torch_cuda_state is not None:
                    torch.cuda.set_rng_state(torch_cuda_state)
    return latents


def vae_encode_list(vae, x, weight_dtype):
    latents = []
    for img in x:
        img = vae_encode(vae, img, weight_dtype)
        latents.append(img)
    return latents


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.rank = hccl_info.rank
        ctx.world_size = hccl_info.world_size
        ctx.input_shape = input_tensor.shape
        ctx.device = input_tensor.device  # 保存输入所在设备

        tensor_list = [torch.empty_like(input_tensor) for _ in range(ctx.world_size)]
        dist.all_gather(tensor_list, input_tensor, group=hccl_info.group)
        output = torch.cat(tensor_list, dim=1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rank = ctx.rank
        world_size = ctx.world_size
        grad_output = grad_output.to(ctx.device)
        grad_input = grad_output.chunk(world_size, dim=1)[rank]
        return grad_input


def gather_output_to_rank0(input_tensor):
    """
    封装调用自定义的 autograd Function
    """
    return Gather.apply(input_tensor)


def _broadcast_tensor(tensor, src, group, device):
    """
    辅助函数：先广播 tensor 的尺寸，再在非 src 进程上构造相同尺寸的空 tensor，
    最后广播实际数据。
    
    参数:
      tensor: 参与广播的 tensor
      src: 源进程编号
      group: 进程组
      device: 目标设备（例如 "cuda:0"）
      
    返回:
      广播后的 tensor
    """
    rank = dist.get_rank()

    # 第一步：广播 tensor 的维度数
    if rank == src:
        num_dims = torch.tensor([tensor.dim()], dtype=torch.int64, device=device)
    else:
        num_dims = torch.empty(1, dtype=torch.int64, device=device)
    dist.broadcast(num_dims, src=src, group=group)
    n_dims = int(num_dims.item())

    # 第二步：广播各维度的尺寸
    if rank == src:
        shape_tensor = torch.tensor(tensor.size(), dtype=torch.int64, device=device)
    else:
        shape_tensor = torch.empty(n_dims, dtype=torch.int64, device=device)
    dist.broadcast(shape_tensor, src=src, group=group)
    target_shape = tuple(shape_tensor.tolist())

    # 第三步：非 src 进程构造空 tensor（dtype 与 device 必须一致）
    if rank == src:
        tensor_to_broadcast = tensor.to(device)
    else:
        tensor_to_broadcast = torch.empty(target_shape, dtype=tensor.dtype, device=device)

    # 第四步：广播具体数据（各进程必须以完全相同的顺序调用此函数）
    dist.broadcast(tensor_to_broadcast, src=src, group=group)
    return tensor_to_broadcast

def is_numeric_sequence(seq):
    """
    判断一个列表或元组是否是纯数值序列（只包含 int、float 或 bool）。
    注意：空序列也认为是纯数字序列。
    """
    if not isinstance(seq, (list, tuple)):
        return False
    return all(isinstance(x, (int, float, bool)) for x in seq)

def broadcast_data(data, src, group, device, dtype=None):
    """
    递归地广播 data 中所有的元素，确保各进程调用相同的广播顺序。
    
    处理策略：
      - 如果 data 是 dict，则对每个键对应的 value 递归调用（按键排序保证各进程顺序一致）。
      - 如果 data 是由纯数值构成的 list 或 tuple（例如 frame_blocks 中的数值列表），
        则将其视为整体，转换为 tensor 后调用 _broadcast_tensor，再转换回原数据形态；
      - 如果 data 是列表/元组但内部不是纯数值（例如包含 tensor 或嵌套列表），
        则先广播该容器的长度（保证各阶调用顺序相同），不匹配时报错，再递归处理各个元素；
      - 如果 data 已经是 tensor，则先确保在 device 上，再调用 _broadcast_tensor；
      - 其它类型（如 int、float、bool 等），转换为 tensor 后处理。
      
    参数:
      data: 需要广播的数据（可能嵌套多层容器）
      src: 源进程编号
      group: 进程组
      device: 目标设备（例如 "cuda:0"）
      
    返回:
      广播后的数据，其结构与原数据一致
    """
    rank = dist.get_rank()

    # 字典：按键排序后递归广播每个 value
    if isinstance(data, dict):
        new_data = {}
        for key in sorted(data.keys()):
            if key == "input_pixel_values" or key == "output_images":
                new_data[key] = broadcast_data(data[key], src, group, device, dtype=torch.float32)
            else:
                new_data[key] = broadcast_data(data[key], src, group, device)
        return new_data

    # 如果是纯数字序列（包括空序列），直接作为整体进行广播
    elif isinstance(data, (list, tuple)) and is_numeric_sequence(data):
        t = torch.tensor(data, device=device)
        t = _broadcast_tensor(t, src, group, device)
        # 如果是标量（dim==0）则转换为 Python 标量，否则转换为 list
        if t.dim() == 0:
            return t.item()
        else:
            return t.tolist()
    
    # 对于列表（非纯数字序列），先广播列表长度，再递归处理每个元素
    elif isinstance(data, list):
        local_len = torch.tensor([len(data)], dtype=torch.int64, device=device)
        if rank == src:
            global_len = local_len.clone()
        else:
            global_len = torch.empty(1, dtype=torch.int64, device=device)
        dist.broadcast(global_len, src=src, group=group)
        expected_length = int(global_len.item())
        if len(data) != expected_length:
            data = [torch.empty(1, dtype=dtype, device=device)]* expected_length if dtype is not None else [0] * expected_length 
        
        return [broadcast_data(item, src, group, device) for item in data]

    # 对于元组（非纯数字序列），先广播元组长度，再递归处理每个元素
    elif isinstance(data, tuple):
        local_len = torch.tensor([len(data)], dtype=torch.int64, device=device)
        if rank == src:
            global_len = local_len.clone()
        else:
            global_len = torch.empty(1, dtype=torch.int64, device=device)
        dist.broadcast(global_len, src=src, group=group)
        expected_length = int(global_len.item())
        if len(data) != expected_length:
            data = [torch.empty(1, dtype=dtype, device=device)]* expected_length if dtype is not None else [0] * expected_length 
        return tuple(broadcast_data(item, src, group, device) for item in data)

    # 如果 data 已经是 tensor，确保其在 device 上，并调用 _broadcast_tensor
    elif torch.is_tensor(data):
        data = data.to(device)
        return _broadcast_tensor(data, src, group, device)

    # 其它基本类型：转换为 tensor 后广播，再还原原类型
    else:
        t = torch.tensor(data, device=device)
        t = _broadcast_tensor(t, src, group, device)
        if t.dim() == 0:
            return t.item()
        else:
            return t.tolist()


def broadcast_pickle(obj, src, group, device):
    """
    使用 pickle 序列化对象后广播，适用于任意 pickleable 对象。
    
    参数:
      obj: 需要广播的数据（任意 pickleable 对象）
      src: 源进程编号
      group: 进程组
      device: 目标设备（例如 "cuda:0"）
      
    返回:
      广播后的数据，其结构和原数据一致
    """
    rank = dist.get_rank()
    
    if rank == src:
        # 1. 序列化对象
        buffer = pickle.dumps(obj)
        # 转换为 ByteTensor（先将 bytes 转成 list，再转换为 ByteTensor）
        byte_tensor = torch.ByteTensor(list(buffer)).to(device)
        # 得到 tensor 长度
        length_tensor = torch.tensor([byte_tensor.numel()], device=device, dtype=torch.int64)
    else:
        # 非源进程：先申请一个用来存放长度的 tensor
        length_tensor = torch.empty(1, device=device, dtype=torch.int64)
    
    # 2. 广播序列化结果的长度（确保所有进程知道字节数）
    dist.broadcast(length_tensor, src=src, group=group)
    length = int(length_tensor.item())
    
    if rank != src:
        # 非源进程根据广播得到的长度申请 ByteTensor
        byte_tensor = torch.empty(length, device=device, dtype=torch.uint8)
    
    # 3. 广播存放序列化数据的 ByteTensor（调用次数严格一致）
    dist.broadcast(byte_tensor, src=src, group=group)
    
    # 4. 反序列化还原对象
    buffer = bytes(byte_tensor.tolist())
    obj = pickle.loads(buffer)
    return obj