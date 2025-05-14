import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    from LVM.transform.fa_transform import replace_attention
    print('use Ascend NPU')
except:
    torch_npu = None
    from LVM.transform.sdpa_transform import replace_attention
    print('use NVIDIA GPU')
import os
import argparse
from decord import cpu, gpu
from decord import VideoReader
from accelerate import Accelerator
from PIL import Image
from safetensors.torch import load_file

from LVM import LVMPipeline
from LVM.acceleration.parallel_states import init_npu_env, hccl_info


def print_model_parameters(model):
    """
    打印模型的总参数量和可训练参数量。
    
    Args:
        model (torch.nn.Module): 需要统计参数的 PyTorch 模型
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        num_params = param.numel()  # 计算当前参数的元素总数
        total_params += num_params
        if param.requires_grad:  # 判断参数是否需要梯度
            trainable_params += num_params
    
    print(f"总参数量(Total Parameters): {total_params:,}")
    print(f"可训练参数量(Trainable Parameters): {trainable_params:,}")


def main(args):
    init_npu_env(args)
    device = torch.cuda.current_device()
    print(hccl_info.rank, hccl_info.world_size, device, flush=True)
    pipe = LVMPipeline.from_pretrained(args.model_name, load_llm_ckpt=False, )
    if args.model_ckpt is not None:
        state_dicts = {}
        dir_path = os.path.join(args.model_ckpt, "pytorch_model.bin")

        if os.path.exists(os.path.join(args.model_ckpt, 'model.pt')):
            state_dicts = torch.load(os.path.join(args.model_ckpt, 'model.pt'), map_location="cpu")
            args.prediction_type = "v"
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

        pipe.model.load_state_dict(state_dicts, strict=False)
        print(f"load model from {args.model_ckpt} OK")
    pipe.model.llm = replace_attention(pipe.model.llm)
    print_model_parameters(pipe.model)

    k = 0
    video_paths = os.listdir(args.video_path_dir)
    video_extensions = (
        '.mp4', '.avi', '.mkv', '.flv', '.mov', '.wmv', '.mpg', '.mpeg',
        '.3gp', '.m4v', '.ts', '.webm', '.vob', '.rm', '.rmvb', '.ogv',
        '.ogg', '.drc', '.mng', '.qt', '.f4v', '.f4p', '.f4a', '.f4b',
        '.asf', '.amv', '.divx', '.mk3d', '.mts', '.m2ts', '.vob', '.ogm',
        '.svi', '.gifv', '.mxf', '.roq', '.nsv', '.viv', '.wtv', '.yuv'
    )  
    for video_path in video_paths:
        if video_path.lower().endswith(video_extensions):
            vr = VideoReader(os.path.join(args.video_path_dir, video_path), ctx=cpu(0))
            pil_image = []
            for i in range(args.clean_image_num):
                pil_image.append(Image.fromarray(vr[i*4].asnumpy()))
            images = pipe.prompt_condition_frame_block_autoregressive_inference(
                input_images=pil_image, 
                use_input_image_size_as_output=True,
                gen_nums=args.gen_nums,
                max_input_image_size=args.max_input_image_size, 
                img_guidance_scale=args.img_guidance_scale,  
                use_img_guidance=True,
                num_inference_steps=args.num_inference_steps,
                use_kv_cache=False,
                prediction_type=args.prediction_type,
                seed=42,
                clean_image_noise_level=args.clean_image_noise_level,
                max_frame_window=args.max_frame_window,
                )
            for i, image in enumerate(images):
                if args.local_rank == 0:
                    image.save(args.save_dir + f"/{k}_{i}.png")
            k += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="result/test/inference")
    parser.add_argument("--video_path_dir", type=str, default="vids")
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument('--world_size', type=int, default=1)  
    parser.add_argument('--local_rank', type=int, default=-1)    
    parser.add_argument('--sequence_parallel_size', type=int, default=1)
    parser.add_argument('--prediction_type', type=str, default="x1")    
    parser.add_argument('--height', type=int, default=256)    
    parser.add_argument('--width', type=int, default=480)    
    parser.add_argument('--num_inference_steps', type=int, default=50)   
    parser.add_argument('--img_guidance_scale', type=float, default=1)    
    parser.add_argument('--gen_nums', type=int, nargs='+', default=[4,4])
    parser.add_argument('--max_input_image_size', type=int, default=320)   
    parser.add_argument('--clean_image_noise_level', type=float, default=0.1)   
    parser.add_argument('--clean_image_num', type=int, default=1)  
    parser.add_argument('--max_frame_window', type=int, default=16)  
    args = parser.parse_args()  
    main(args)