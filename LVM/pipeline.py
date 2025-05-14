import os
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import gc

from PIL import Image
import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from diffusers.models import AutoencoderKL
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from safetensors.torch import load_file

from LVM import LVMProcessor, LVM, LVMScheduler
from LVM.acceleration.parallel_states import hccl_info


logger = logging.get_logger(__name__) 

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from LVM import LVMPipeline
        >>> pipe = FluxControlNetPipeline.from_pretrained(
        ...     base_model
        ... )
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=2.5,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""


class LVMPipeline:
    def __init__(
        self,
        vae: AutoencoderKL,
        model: LVM,
        processor: LVMProcessor,
        device: Union[str, torch.device] = None,
    ):
        self.vae = vae
        self.model = model
        self.processor = processor
        self.device = device

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                logger.info("Don't detect any available GPUs, using CPU instead, this may take long time to generate image!!!")
                self.device = torch.device("cpu")

        # self.model.to(torch.bfloat16)
        self.model.eval()
        self.vae.eval()

        self.model_cpu_offload = False

    @classmethod
    def from_pretrained(cls, model_name, vae_path: str=None, load_llm_ckpt=True):
        if not os.path.exists(model_name) or (not os.path.exists(os.path.join(model_name, 'model.safetensors')) and model_name == "Shitao/OmniGen-v1"):
            logger.info("Model not found, downloading...")
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5', 'model.pt'])
            logger.info(f"Downloaded model to {model_name}")
        model = LVM.from_pretrained(model_name, load_llm_ckpt=load_llm_ckpt)
        # print(hccl_info.world_size, "\n\n\n\n\n")
        processor = LVMProcessor.from_pretrained(model_name, sequence_parallel_size=hccl_info.world_size)

        if os.path.exists(os.path.join(model_name, "vae")):
            vae = AutoencoderKL.from_pretrained(os.path.join(model_name, "vae"))
        elif vae_path is not None:
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        else:
            logger.info(f"No VAE found in {model_name}, downloading stabilityai/sdxl-vae from HF")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)

        return cls(vae, model, processor)
    
    def merge_lora(self, lora_path: str):
        model = PeftModel.from_pretrained(self.model, lora_path)
        model.merge_and_unload()

        self.model = model
    
    def to(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        self.vae.to(device)
        self.device = device

    def vae_encode(self, x, dtype):
        if self.vae.config.shift_factor is not None:
            x = self.vae.encode(x).latent_dist.sample()
            x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            x = self.vae.encode(x).latent_dist.sample().mul_(self.vae.config.scaling_factor)
        x = x.to(dtype)
        return x
    
    def move_to_device(self, data):
        if isinstance(data, list):
            return [x.to(self.device) for x in data]
        return data.to(self.device)

    def enable_model_cpu_offload(self):
        self.model_cpu_offload = True
        self.model.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()  # Clear VRAM
        gc.collect()  # Run garbage collection to free system RAM
    
    def disable_model_cpu_offload(self):
        self.model_cpu_offload = False
        self.model.to(self.device)
        self.vae.to(self.device)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        input_images = None,
        height: int = 1024,
        width: int = 1024,
        gen_num: int = 1,
        num_inference_steps: int = 50,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = True,
        use_input_image_size_as_output: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = None,
        output_type: str = "pil",
        time_shifting_factor: float = 1.0,
        prediction_type: str = "v",
        clean_image_noise_level: float = None,
        ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                The list of input images. We will replace the "<|image_i|>" in prompt with the 1-th image in list.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image. The number must be a multiple of 16.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image. The number must be a multiple of 16.
            gen_num (`int`, *optional*, defaults to 1):
                The number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            max_input_image_size (`int`, *optional*, defaults to 1024): the maximum size of input image, which will be used to crop the input image to the maximum size
            use_kv_cache (`bool`, *optional*, defaults to True): enable kv cache to speed up the inference
            offload_kv_cache (`bool`, *optional*, defaults to True): offload the cached key and value to cpu, which can save memory but slow down the generation silightly
            offload_model (`bool`, *optional*, defaults to False): offload the model to cpu, which can save memory but slow down the generation
            use_input_image_size_as_output (bool, defaults to False): whether to use the input image size as the output image size, which can be used for single-image input, e.g., image editing task
            seed (`int`, *optional*):
                A random seed for generating output. 
            dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                data type for the model
            output_type (`str`, *optional*, defaults to "pil"):
                The type of the output image, which can be "pt" or "pil"
            time_shifting_factor (`float`, *optional*, defaults to 1):
        Examples:

        Returns:
            A list with the generated images.
        """
        # check inputs:
        prompt_img_len = len(input_images) if input_images is not None else 0
        if use_input_image_size_as_output:
            if not prompt_img_len == 1:
                print("if you want to make sure the output image have the same size as the input image, please only input one image instead of multiple input images")
        else:
            assert height%16 == 0 and width%16 == 0, "The height and width must be a multiple of 16."
        
        ori_input_images = input_images.copy() if input_images is not None else None

        output_images = []
        if img_guidance_scale == 1:
            use_img_guidance = False
        ori_use_img_guidance = use_img_guidance
        for gen_idx in range(gen_num):
            print(f"Generating image {gen_idx+1}/{gen_num}", flush=True)
            if len(output_images) != 0:
                if ori_input_images is None:
                    ori_input_images = [output_image]
                else:
                    ori_input_images.append(output_image)
            if ori_input_images is None:
                use_img_guidance = False
            else:
                use_img_guidance = ori_use_img_guidance

            prompt = ""
            if ori_input_images is not None:
                for i in range(len(ori_input_images)):
                    prompt += f"<img><|image_{i+1}|></img>"

            if isinstance(prompt, str):
                prompt = [prompt]
                input_images = [ori_input_images.copy()] if ori_input_images is not None else None
            
            # set model and processor
            if max_input_image_size != self.processor.max_image_size:
                self.processor = LVMProcessor(self.processor.text_tokenizer, max_image_size=max_input_image_size, sequence_parallel_size=hccl_info.world_size)
            self.model.to(dtype)
            if offload_model:
                self.enable_model_cpu_offload()
            else:
                self.disable_model_cpu_offload()

            input_data = self.processor(prompt, input_images, height=height, width=width, use_img_cfg=use_img_guidance, use_input_image_size_as_output=use_input_image_size_as_output)
            num_prompt = len(prompt)
            num_cfg = 1 if use_img_guidance else 0
            if use_input_image_size_as_output:
                height, width = input_data['input_pixel_values'][0].shape[-2:]
            latent_size_h, latent_size_w = height//8, width//8

            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None
            latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
            latents = torch.cat([latents]*(1+num_cfg), 0).to(dtype)

            if input_images is not None and self.model_cpu_offload: self.vae.to(self.device)
            input_img_latents = []
            for idx, img in enumerate(input_data['input_pixel_values']):
                img = self.vae_encode(img.to(self.device), dtype)
                if idx >= prompt_img_len:
                    img = (1 - clean_image_noise_level) * img + clean_image_noise_level * torch.randn_like(img)
                input_img_latents.append(img)

            if input_images is not None and self.model_cpu_offload:
                self.vae.to('cpu')
                torch.cuda.empty_cache()  # Clear VRAM
                gc.collect()  # Run garbage collection to free system RAM

            model_kwargs = dict(input_ids=self.move_to_device(input_data['input_ids']), 
                                input_img_latents=input_img_latents, 
                                input_image_sizes=input_data['input_image_sizes'], 
                                attention_mask=self.move_to_device(input_data["attention_mask"]), 
                                position_ids=self.move_to_device(input_data["position_ids"]), 
                                img_cfg_scale=img_guidance_scale,
                                use_img_cfg=use_img_guidance,
                                use_kv_cache=use_kv_cache,
                                offload_model=offload_model,
                                )
            
            func = self.model.forward_with_cfg

            if self.model_cpu_offload:
                for name, param in self.model.named_parameters():
                    if 'layers' in name and 'layers.0' not in name:
                        param.data = param.data.cpu()
                    else:
                        param.data = param.data.to(self.device)
                for buffer_name, buffer in self.model.named_buffers():
                    setattr(self.model, buffer_name, buffer.to(self.device))
            # else:
            #     self.model.to(self.device)

            scheduler = LVMScheduler(num_steps=num_inference_steps, time_shifting_factor=time_shifting_factor)
            samples = scheduler(latents, 
                                func, 
                                model_kwargs, 
                                use_kv_cache=use_kv_cache, 
                                offload_kv_cache=offload_kv_cache, 
                                prediction_type=prediction_type,
                                vae=self.vae,
                                )
            samples = samples.chunk((1+num_cfg), dim=0)[0]

            if self.model_cpu_offload:
                self.model.to('cpu')
                torch.cuda.empty_cache()  
                gc.collect()  

            self.vae.to(self.device)

            if gen_idx == 0:
                for input_img_latent in input_img_latents:
                    input_img_latent = input_img_latent.to(torch.float32)
                    if self.vae.config.shift_factor is not None:
                        input_img_latent = input_img_latent / self.vae.config.scaling_factor + self.vae.config.shift_factor
                    else:
                        input_img_latent = input_img_latent / self.vae.config.scaling_factor
                    input_img = self.vae.decode(input_img_latent).sample
                    input_img = (input_img * 0.5 + 0.5).clamp(0, 1)
                    input_img = (input_img * 255).to("cpu", dtype=torch.uint8)
                    input_img = input_img.permute(0, 2, 3, 1).numpy()
                    input_img = Image.fromarray(input_img[0])
                    output_images.append(input_img)

            samples = samples.to(torch.float32)
            if self.vae.config.shift_factor is not None:
                samples = samples / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                samples = samples / self.vae.config.scaling_factor   
            samples = self.vae.decode(samples).sample

            if self.model_cpu_offload:
                self.vae.to('cpu')
                torch.cuda.empty_cache()  
                gc.collect()  
            
            samples = (samples * 0.5 + 0.5).clamp(0, 1)

            output_samples = (samples * 255).to("cpu", dtype=torch.uint8)
            output_samples = output_samples.permute(0, 2, 3, 1).numpy()
            output_image = Image.fromarray(output_samples[0])
            output_images.append(output_image)

        torch.cuda.empty_cache()  # Clear VRAM
        gc.collect()              # Run garbage collection to free system RAM

        return output_images


    @torch.no_grad()
    def prompt_condition_frame_block_autoregressive_inference(
        self,
        input_images = None,
        height: int = 1024,
        width: int = 1024,
        gen_nums: list = [1],
        num_inference_steps: int = 50,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = True,
        use_input_image_size_as_output: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = None,
        output_type: str = "pil",
        time_shifting_factor: float = 1.0,
        prediction_type: str = "v",
        clean_image_noise_level: float = None,
        max_frame_window: int = 16,
        ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                The list of input images. We will replace the "<|image_i|>" in prompt with the 1-th image in list.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image. The number must be a multiple of 16.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image. The number must be a multiple of 16.
            gen_num (`int`, *optional*, defaults to 1):
                The number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            max_input_image_size (`int`, *optional*, defaults to 1024): the maximum size of input image, which will be used to crop the input image to the maximum size
            use_kv_cache (`bool`, *optional*, defaults to True): enable kv cache to speed up the inference
            offload_kv_cache (`bool`, *optional*, defaults to True): offload the cached key and value to cpu, which can save memory but slow down the generation silightly
            offload_model (`bool`, *optional*, defaults to False): offload the model to cpu, which can save memory but slow down the generation
            use_input_image_size_as_output (bool, defaults to False): whether to use the input image size as the output image size, which can be used for single-image input, e.g., image editing task
            seed (`int`, *optional*):
                A random seed for generating output. 
            dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                data type for the model
            output_type (`str`, *optional*, defaults to "pil"):
                The type of the output image, which can be "pt" or "pil"
            time_shifting_factor (`float`, *optional*, defaults to 1):
        Examples:

        Returns:
            A list with the generated images.
        """
        prompt_img_len = len(input_images) if input_images is not None else 0
        if use_input_image_size_as_output:
            if not prompt_img_len == 1:
                print("if you want to make sure the output image have the same size as the input image, please only input one image instead of multiple input images")
        else:
            assert height%16 == 0 and width%16 == 0, "The height and width must be a multiple of 16."

        output_images = []
        if img_guidance_scale == 1:
            use_img_guidance = False

        if input_images is None:
            use_img_guidance = False

        for k, gen_num in enumerate(gen_nums):
            if k > 0:
                input_images = output_images
            if len(input_images) + gen_num > max_frame_window:
                input_images = input_images[gen_num + len(input_images) - max_frame_window:]

            prompt_img_len = len(input_images) if input_images is not None else 0

            prompt = ""
            if input_images is not None:
                for i in range(len(input_images) + gen_num):
                    if i < len(input_images):
                        prompt += f"<img><|image_{i+1}|></img>"
                    else:
                        prompt += f"<|diffusion|><|image_{i+1}|>"

            frame_blocks = [len(input_images), gen_num]

            if use_img_guidance:
                prompt_ = ""
                for i in range(gen_num):
                    prompt_ += f"<|diffusion|><|image_{i+1}|>"
            
            if not use_img_guidance:
                if isinstance(prompt, str):
                    prompt = [prompt]
                    input_images = [input_images]
            else:
                if isinstance(prompt, str):
                    prompt = [prompt, prompt_]
                    input_images = [input_images, input_images[prompt_img_len:].copy()]

            # set model and processor
            if max_input_image_size != self.processor.max_image_size:
                self.processor = LVMProcessor(self.processor.text_tokenizer, max_image_size=max_input_image_size, sequence_parallel_size=hccl_info.world_size)
            self.model.to(dtype)
            if offload_model:
                self.enable_model_cpu_offload()
            else:
                self.disable_model_cpu_offload()

            input_data = self.processor.prompt_condition_frame_block_inference(
                prompt, 
                input_images, 
                height=height, 
                width=width, 
                use_img_cfg=use_img_guidance, 
                use_input_image_size_as_output=use_input_image_size_as_output,
                frame_blocks=frame_blocks,
                )
            num_cfg = 1 if use_img_guidance else 0
            if use_input_image_size_as_output:
                height, width = input_data['input_pixel_values'][0].shape[-2:]
            latent_size_h, latent_size_w = height//8, width//8

            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None

            latents = []
            for i in range(gen_num):
                latent = torch.randn(1, 4, latent_size_h, latent_size_w, device=self.device, generator=generator).to(dtype)
                latents.append(latent)
            latents = latents * (1+num_cfg)

            if input_images is not None and self.model_cpu_offload: self.vae.to(self.device)
            input_img_latents = []

            if use_img_guidance:
                shared_img_len = int((len(input_data['input_pixel_values']) - prompt_img_len) / 2)
            else:
                shared_img_len = int((len(input_data['input_pixel_values']) - prompt_img_len))
            for idx in range(prompt_img_len+shared_img_len):
                img = input_data['input_pixel_values'][idx]
                img = self.vae_encode(img.to(self.device), dtype)
                # if idx >= prompt_img_len:
                # if idx == prompt_img_len+shared_img_len-1 and gen_idx != 0:
                if k > 0:
                    img = (1 - clean_image_noise_level) * img + clean_image_noise_level * torch.randn_like(img)
                input_img_latents.append(img)
            if use_img_guidance:
                input_img_latents.extend(input_img_latents[prompt_img_len:prompt_img_len + shared_img_len])

            if input_images is not None and self.model_cpu_offload:
                self.vae.to('cpu')
                torch.cuda.empty_cache()  # Clear VRAM
                gc.collect()  # Run garbage collection to free system RAM

            model_kwargs = dict(
                input_ids=self.move_to_device(input_data['input_ids']), 
                input_img_latents=input_img_latents, 
                input_image_sizes=input_data['input_image_sizes'], 
                attention_mask=self.move_to_device(input_data["attention_mask"]), 
                position_ids=self.move_to_device(input_data["position_ids"]), 
                denoise_image_sizes=input_data['denoise_image_sizes'],
                time_emb_inx=input_data['time_emb_inx'],
                img_cfg_scale=img_guidance_scale,
                use_img_cfg=use_img_guidance,
                use_kv_cache=use_kv_cache,
                offload_model=offload_model,
                vae=self.vae,
                )
            
            func = self.model.frame_block_forward_with_cfg

            if self.model_cpu_offload:
                for name, param in self.model.named_parameters():
                    if 'layers' in name and 'layers.0' not in name:
                        param.data = param.data.cpu()
                    else:
                        param.data = param.data.to(self.device)
                for buffer_name, buffer in self.model.named_buffers():
                    setattr(self.model, buffer_name, buffer.to(self.device))
            # else:
            #     self.model.to(self.device)

            scheduler = LVMScheduler(
                num_steps=num_inference_steps, 
                time_shifting_factor=time_shifting_factor
                )
            samples = scheduler(
                latents, 
                func, 
                model_kwargs, 
                use_kv_cache=use_kv_cache, 
                offload_kv_cache=offload_kv_cache, 
                prediction_type=prediction_type,
                vae=self.vae,
                )
                
            samples = samples[:len(samples) // 2]

            if self.model_cpu_offload:
                self.model.to('cpu')
                torch.cuda.empty_cache()  
                gc.collect()  

            self.vae.to(self.device)

            if k == 0:
                for input_img_latent in input_img_latents:
                    input_img_latent = input_img_latent.to(torch.float32)
                    if self.vae.config.shift_factor is not None:
                        input_img_latent = input_img_latent / self.vae.config.scaling_factor + self.vae.config.shift_factor
                    else:
                        input_img_latent = input_img_latent / self.vae.config.scaling_factor
                    input_img = self.vae.decode(input_img_latent).sample
                    input_img = (input_img * 0.5 + 0.5).clamp(0, 1)
                    input_img = (input_img * 255).to("cpu", dtype=torch.uint8)
                    input_img = input_img.permute(0, 2, 3, 1).numpy()
                    input_img = Image.fromarray(input_img[0])
                    output_images.append(input_img)

            for sample in samples:
                sample = sample.to(torch.float32)
                if self.vae.config.shift_factor is not None:
                    sample = sample / self.vae.config.scaling_factor + self.vae.config.shift_factor
                else:
                    sample = sample / self.vae.config.scaling_factor   
                sample = self.vae.decode(sample).sample

                if self.model_cpu_offload:
                    self.vae.to('cpu')
                    torch.cuda.empty_cache()  
                    gc.collect()  
                
                sample = (sample * 0.5 + 0.5).clamp(0, 1)

                output_sample = (sample * 255).to("cpu", dtype=torch.uint8)
                output_sample = output_sample.permute(0, 2, 3, 1).numpy()
                output_image = Image.fromarray(output_sample[0])
                output_images.append(output_image)

        torch.cuda.empty_cache()  # Clear VRAM
        gc.collect()              # Run garbage collection to free system RAM

        return output_images
