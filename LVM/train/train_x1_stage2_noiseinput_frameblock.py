import torch
try:
    import torch_npu
    import torch.multiprocessing as mp
    from torch_npu.contrib import transfer_to_npu
    print('use Ascend NPU')
except:
    torch_npu = None
    print('use NVIDIA GPU')
import gc
import glob
import json
from time import time
import argparse
import logging
import os
from pathlib import Path
import math
import deepspeed
import numpy as np
from PIL import Image
from copy import deepcopy
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from torchvision import transforms

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import ProjectConfiguration, set_seed, DeepSpeedPlugin
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedType
from peft import LoraConfig, set_peft_model_state_dict, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from diffusers.models import AutoencoderKL

from LVM.train_helper.loss import is_all_equal
from LVM import LVMTraining_CP, LVMProcessor
if torch_npu is not None:
    from LVM.transform.fa_transform import replace_attention, replace_simple_attention
else:
    from LVM.transform.sdpa_transform import replace_attention
from LVM.acceleration.parallel_states import initialize_sequence_parallel_state, hccl_info
from LVM.train_helper import DatasetFromVideoBlockFrame, TrainDataCollator_FrameBlock
from LVM.train_helper import training_losses_x1_noise_input
from LVM.utils import (
    vae_encode, 
    # crop_arr,
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    broadcast_data,
    broadcast_pickle,
)

MAX_IMAGE_SIZE = 480

torch._dynamo.config.suppress_errors = True


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


def crop_arr(pil_image):
    global MAX_IMAGE_SIZE
    while min(*pil_image.size) >= 2 * MAX_IMAGE_SIZE:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if max(*pil_image.size) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(*pil_image.size)
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


def sync_gradients(model):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    for param in model.parameters():
        if param.grad is not None:
            # 执行 all-reduce，将各个 GPU 上的梯度进行求和
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            # 对梯度求平均
            param.grad.data /= world_size


def main(args):
    # Setup accelerator:
    from accelerate import DistributedDataParallelKwargs as DDPK
    kwargs = DDPK(find_unused_parameters=True)
    if args.deepspeed_plugin is not None:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_plugin)
    else:
        deepspeed_plugin = None
        deepspeed.init_distributed()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.results_dir,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[kwargs],
        )
    device = accelerator.device
    accelerator.init_trackers("tensorboard_log", config=args.__dict__)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    logger = create_logger(args.results_dir)
    checkpoint_dir = f"{args.results_dir}/checkpoints"  # Stores saved model checkpoints
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created at {args.results_dir}")
        json.dump(args.__dict__, open(os.path.join(args.results_dir, 'train_args.json'), 'w'))

    # Create model:    
    if not os.path.exists(args.model_name_or_path):
        cache_folder = os.getenv('HF_HUB_CACHE')
        args.model_name_or_path = snapshot_download(repo_id=args.model_name_or_path,
                                        cache_dir=cache_folder,
                                        ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])
        logger.info(f"Downloaded model to {args.model_name_or_path}")
    model = LVMTraining_CP.from_pretrained(args.model_name_or_path, load_llm_ckpt=False)
    model.llm.config.use_cache = False

    if args.model_ckpt is not None:
        state_dicts = {}
        dir_path = os.path.join(args.model_ckpt, "pytorch_model.bin")
        if os.path.exists(os.path.join(args.model_ckpt, 'model.pt')):
            state_dicts = torch.load(os.path.join(args.model_ckpt, 'model.pt'), map_location="cpu")
        elif os.path.exists(os.path.join(args.model_ckpt, 'model.safetensors')):
            state_dicts = load_file(os.path.join(args.model_ckpt, 'model.safetensors'))
        elif os.path.isfile(dir_path):
            state_dict = torch.load(dir_path, map_location="cpu")
            state_dicts.update(state_dict)
        else:
            file_list = os.listdir(dir_path)
            for ckpt_file in file_list:
                if ckpt_file.endswith(".bin"):
                    ckpt_file_path = os.path.join(dir_path, ckpt_file)
                    state_dict = torch.load(ckpt_file_path, map_location="cpu")  # 加载保存的参数
                    state_dicts.update(state_dict)
        model.load_state_dict(state_dicts)
        print(f"load model from {args.model_ckpt} OK")

    if args.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable()
    model = model.to(device)

    if args.vae_path is None:
        vae_path = os.path.join(args.model_name_or_path, "vae")
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        else:
            logger.info("No VAE found in model, downloading stabilityai/sdxl-vae from HF")
            logger.info("If you have VAE in local folder, please specify the path with --vae_path")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(dtype=torch.float32)
    vae.eval()
    model.to(weight_dtype)

    # Replace attention with distributed flash attention:
    initialize_sequence_parallel_state(args.sequence_parallel_size)
    model.llm = replace_attention(model.llm)
    print(hccl_info.rank, hccl_info.world_size, device, flush=True)

    processor = LVMProcessor.from_pretrained(args.model_name_or_path, 
                                             sequence_parallel_size=args.sequence_parallel_size
                                             )

    requires_grad(vae, False)
    if args.use_lora:
        if accelerator.distributed_type == DistributedType.FSDP:
            raise NotImplementedError("FSDP does not support LoRA")
        requires_grad(model, False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["qkv_proj", "o_proj"],
        )
        model.llm.enable_input_require_grads()
        model = get_peft_model(model, transformer_lora_config)
        model.to(weight_dtype)
        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        opt = torch.optim.AdamW(transformer_lora_parameters, 
                                lr=args.lr, 
                                weight_decay=args.adam_weight_decay,
                                betas=(args.adam_beta1, args.adam_beta2),
                                eps=args.adam_epsilon,
                                )
    else:
        opt = torch.optim.AdamW(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.adam_weight_decay,
                                betas=(args.adam_beta1, args.adam_beta2),
                                eps=args.adam_epsilon,
                                )

    ema = None
    if args.use_ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
    

    # Setup data:
    crop_func = crop_arr
    if not args.keep_raw_resolution:
        crop_func = center_crop_arr
    image_transform = transforms.Compose([
        transforms.Lambda(crop_func),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = DatasetFromVideoBlockFrame(
        video_dir_path=args.video_dir_path,
        processer=processor,
        image_transform=image_transform,
        max_input_length_limit=args.max_input_length_limit,
        keep_raw_resolution=args.keep_raw_resolution,
        frame_num=args.frame_num,
        frame_interval=args.frame_interval,
        data_reuse=args.data_reuse,
        data_limit=args.data_limit,
        flexible_interval=False,
        )
    collate_fn = TrainDataCollator_FrameBlock(
        pad_token_id=processor.text_tokenizer.eos_token_id, 
        hidden_size=model.llm.config.hidden_size, 
        keep_raw_resolution=args.keep_raw_resolution,
        frame_num=args.frame_num,
        sequence_parallel_size=args.sequence_parallel_size,
        batch_size=args.batch_size_per_device,
        )

    if args.prefetch_factor == 0:
        args.prefetch_factor = None
    loader = DataLoader(dataset,
                        collate_fn=collate_fn,
                        batch_size=args.batch_size_per_device,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        prefetch_factor=args.prefetch_factor,
                        timeout=args.timeout,
                        )

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,}")

    num_update_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(args.lr_scheduler,
                                 optimizer=opt,
                                 num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                                 num_training_steps=max_train_steps * args.gradient_accumulation_steps,
                                 )

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    
    if ema is not None:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode
    

    if ema is not None:
        model, ema, opt, lr_scheduler, loader = accelerator.prepare(model, ema, opt, lr_scheduler, loader)
    else:
        model, opt, lr_scheduler, loader = accelerator.prepare(model, opt, lr_scheduler, loader)
  
    # Variables for monitoring/logging purposes:
    train_steps, log_steps = 0, 0
    running_loss = 0
    running_grad = 0
    start_time = time()

    if args.auto_resume:
        print('auto resume')
        # find the latest checkpoint in args.results_dir
        checkpoint_files = glob.glob(os.path.join(args.results_dir, 'checkpoint-*'))
        if len(checkpoint_files) != 0:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('-')[-1]))
            print(f'latest_checkpoint: {latest_checkpoint}')
            train_steps = int(latest_checkpoint.split('-')[-1])
            log_steps = train_steps
            args.resume_from_checkpoint = latest_checkpoint
        else:
            print('no autoresume checkpoint found')
    if args.resume_from_checkpoint is not None:
        # if args.resume_from_checkpoint is dir
        if os.path.isdir(args.resume_from_checkpoint):
            logger.info(f'Load checkpoint from {args.resume_from_checkpoint}')
            accelerator.load_state(args.resume_from_checkpoint)
            # train_dataloader.resume()
            print(f'global_step: {train_steps}')
        # end with pytorch_model.bin
        elif args.resume_from_checkpoint.endswith('pytorch_model.bin'):
            cktp = torch.load(args.resume_from_checkpoint, map_location='cpu')
            # unwarp model to load ckpt
            missing_keys, unexpected_keys = accelerator.unwrap_model(agent_model).load_state_dict(cktp)
            with open('resume_missing_keys.txt', 'w') as f:
                f.write(str(missing_keys))
            with open('resume_unexpected_keys.txt', 'w') as f:
                f.write(str(unexpected_keys))
            print(f"load from {args.resume_from_checkpoint} done. missing_keys: resume_missing_keys.txt, unexpected_keys: resume_unexpected_keys.txt")

        gc.collect()

    # 如果还需要重置或更新 scheduler 参数
    # 具体做法取决于你的调度器类型
    lr_scheduler.scheduler.base_lrs = [args.lr for _ in lr_scheduler.scheduler.base_lrs]
    
    if accelerator.is_main_process:
        try:
            logger.info(f"Training for {args.epochs} epochs...")
        except:
            print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            try:
                logger.info(f"Beginning epoch {epoch}...")
            except:
                print(f"Beginning epoch {epoch}...")
        for data in loader:
            data = recursive_to_device(data, "cpu")
            for i in range(hccl_info.world_size):
                src = dist.get_rank() - hccl_info.rank + i
                tmp_data = deepcopy(data)
                tmp_data = broadcast_data(tmp_data, src=src, group=hccl_info.group, device=device)
                with accelerator.accumulate(model):
                    with torch.no_grad():
                        input_pixel_values = tmp_data['input_pixel_values']
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)
                        output_images = vae_encode(vae, tmp_data["output_images"], weight_dtype)

                    if accelerator.is_main_process:
                        value = model.module.x_embedder.proj.weight.data.abs().max().item()
                        print(f"x_embedder.proj.weight.data:\n{value:.16f}", flush=True)

                    model_kwargs = dict(
                        input_ids=tmp_data['input_ids'], 
                        input_img_latents=input_pixel_values, 
                        input_image_sizes=tmp_data['input_image_sizes'], 
                        attention_mask=tmp_data['attention_mask'], 
                        position_ids=tmp_data['position_ids'], 
                        denoise_image_sizes=tmp_data["denoise_image_sizes"],
                        time_emb_inx=tmp_data["time_emb_inx"],
                        past_key_values=None, 
                        return_past_key_values=False,
                        vae=vae,
                        )
                    loss_dict = training_losses_x1_noise_input(
                        model, 
                        output_images, 
                        model_kwargs, 
                        input_noise=args.input_noise, 
                        frame_blocks=tmp_data["frame_blocks"],
                        device=device,
                        )
                    loss = loss_dict["loss"].mean()
                    running_loss += loss.item()
                    accelerator.backward(loss)

                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)  # 计算参数梯度的2范数
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** 0.5  # 计算当前设备上的梯度范数

                    grad_norm_tensor = torch.tensor([grad_norm], device=accelerator.device)
                    avg_grad_norm_tensor = accelerator.reduce(grad_norm_tensor, reduction='mean')
                    avg_grad_norm = avg_grad_norm_tensor.item()

                    if args.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    running_grad += avg_grad_norm
                        
                    opt.step()
                    lr_scheduler.step()
                    opt.zero_grad()

                    log_steps += 1
                    train_steps += 1

                    if hccl_info.rank == 0:
                        accelerator.log({"training_loss": loss.item(), "model gradient": avg_grad_norm}, step=train_steps)
                    if train_steps % args.gradient_accumulation_steps == 0:
                        if accelerator.sync_gradients and ema is not None: 
                            update_ema(ema, model)
                        
                    if train_steps % (args.log_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                        torch.cuda.synchronize()
                        end_time = time()
                        steps_per_sec = log_steps / args.gradient_accumulation_steps / (end_time - start_time)
                        # Reduce loss history over all processes:
                        avg_loss = torch.tensor(running_loss / log_steps, device=device)
                        avg_grad = torch.tensor(running_grad / log_steps, device=device)
                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)   
                            dist.all_reduce(avg_grad, op=dist.ReduceOp.SUM)                     
                        avg_loss = avg_loss.item() / accelerator.num_processes 
                        avg_grad = avg_grad.item() / accelerator.num_processes
                            
                        if accelerator.is_main_process:
                            cur_lr = opt.param_groups[0]["lr"]
                            try:
                                logger.info(f"(step={int(train_steps/args.gradient_accumulation_steps):07d}) Train Loss: {avg_loss:.4f}, Model Grad: {avg_grad: 4f}, Train Steps/Sec: {steps_per_sec:.2f}, Epoch: {train_steps/len(loader)}, LR: {cur_lr}")
                            except:
                                print(f"(step={int(train_steps/args.gradient_accumulation_steps):07d}) Train Loss: {avg_loss:.4f}, Model Grad: {avg_grad: 4f}, Train Steps/Sec: {steps_per_sec:.2f}, Epoch: {train_steps/len(loader)}, LR: {cur_lr}")

                        # Reset monitoring variables:
                        running_loss = 0
                        running_grad = 0
                        log_steps = 0
                        start_time = time()


                if train_steps % (args.ckpt_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                    if accelerator.distributed_type == DistributedType.FSDP:
                        state_dict = accelerator.get_state_dict(model)
                        ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None
                    else:
                        if not args.use_lora:
                            if hasattr(model, "module"):
                                state_dict = model.module.state_dict()
                            else:
                                state_dict = model.state_dict()
                            ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None

                    global_step = int(train_steps/args.gradient_accumulation_steps)
                    save_path = os.path.join(args.results_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    try:
                        logger.info(f"Saved checkpoint to {save_path}")
                    except:
                        print(f"Saved checkpoint to {save_path}")
                        
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
    accelerator.end_training()
    model.eval()  
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name_or_path", type=str, default="LVM")
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--video_dir_path", type=str, default=None)
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--frame_interval", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size_per_device", type=int, default=1)
    parser.add_argument("--vae_path", type=str, default=None) 
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=20000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_input_length_limit", type=int, default=1024)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-5)
    parser.add_argument("--deepspeed_plugin", type=str, default=None)
    parser.add_argument("--sequence_parallel_size", type=int, default=1)
    parser.add_argument(
        "--keep_raw_resolution",
        action="store_true",
        help="multiple_resolutions",
    )
    parser.add_argument("--max_image_size", type=int, default=1344)

    parser.add_argument(
            "--use_lora",
            action="store_true",
        )
    parser.add_argument(
            "--lora_rank",
            type=int, 
            default=8
        )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether or not to use ema.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    ) 
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--auto_resume",
        type=bool,
        default=True,
        help="resume from previous checkpoint.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="The checkpoint used to resume.",
    )
    parser.add_argument(
        "--data_reuse",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="The waiting time for the worker to fetch the data in dataloader.",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="First resume model checkpoint path.",
    )
    parser.add_argument(
        "--data_limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--input_noise",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--interval_bound",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    MAX_IMAGE_SIZE = args.max_image_size
    assert args.max_image_size % 16 == 0, "Image size must be divisible by 16."

    main(args)


