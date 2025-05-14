import os
import re
from typing import Dict, List
import json

import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from OmniGen.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    # crop_arr,
)


class LVMProcessor:
    def __init__(self, 
                 text_tokenizer, 
                 max_image_size: int=1024,
                 sequence_parallel_size: int=1,
                 ):
        self.text_tokenizer = text_tokenizer
        self.max_image_size = max_image_size

        self.image_transform = transforms.Compose([
            transforms.Lambda(self.crop_arr),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.sequence_parallel_size = sequence_parallel_size
        self.collator = LVMCollator(sequence_parallel_size=sequence_parallel_size)

    def crop_arr(self, pil_image):
        while min(*pil_image.size) >= 2 * self.max_image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        if max(*pil_image.size) > self.max_image_size:
            scale = self.max_image_size / max(*pil_image.size)
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

    @classmethod
    def from_pretrained(cls, model_name, sequence_parallel_size=1):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           allow_patterns="*.json")
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)

        return cls(text_tokenizer, sequence_parallel_size=sequence_parallel_size)

    def process_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pass
        else:
            print("We get a ", type(image), "but we need a PIL.Image object")
            raise ValueError("Input must be a PIL.Image object")
        return self.image_transform(image)
    
    def process_multi_modal_prompt(self, text, input_images):
        text = self.add_prefix_instruction(text)
        if input_images is None or len(input_images) == 0:
            model_inputs = self.text_tokenizer(text)
            if model_inputs.input_ids[0] == 1:
                model_inputs.input_ids = model_inputs.input_ids[1:]
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)] 

        for i in range(0, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text) 
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        # total images must be the same as the number of image tags
        assert len(unique_image_ids) == len(input_images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"
        
        input_images = [input_images[x-1] for x in image_ids]

        all_input_ids = []
        img_inx = []
        idx = 0
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            if i != len(prompt_chunks) -1:
                start_inx = len(all_input_ids)
                size = input_images[i].size(-2) *  input_images[i].size(-1) // 16 // 16
                img_inx.append([start_inx, start_inx+size])
                all_input_ids.extend([0]*size)

        return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx}

    def process_multi_modal_prompt_frame_block(self, text, input_images, frame_blocks, height=None, width=None):
        if input_images is None:
            input_images = []
        if len(input_images) == 0 and (height is None and width is None):
            model_inputs = self.text_tokenizer(text)
            if model_inputs.input_ids[0] == 1:
                model_inputs.input_ids = model_inputs.input_ids[1:]
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)] 

        for i in range(0, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text) 
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        # total images must be the same as the number of image tags
        assert len(unique_image_ids) == len(input_images) + frame_blocks[-1], f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"
        
        input_images = [input_images[x-1] for x in image_ids[:frame_blocks[0]]]
        
        all_input_ids = []
        img_inx = []
        idx = 0
        for k, frame_block in enumerate(frame_blocks):
            if k != len(frame_blocks) - 1:
                for i in range(frame_block):
                    all_input_ids.extend(prompt_chunks[idx])
                    start_inx = len(all_input_ids)
                    size = input_images[idx].size(-2) *  input_images[idx].size(-1) // 16 // 16
                    img_inx.append([start_inx, start_inx+size])
                    all_input_ids.extend([0]*size)
                    idx += 1
            else:
                for i in range(frame_block):
                    all_input_ids.extend(prompt_chunks[idx])
                    all_input_ids.extend([0])
                    start_inx = len(all_input_ids)
                    if height is not None and width is not None:
                        size = height * width // 16 // 16
                    else:
                        size = input_images[0].size(-2) *  input_images[0].size(-1) // 16 // 16
                    img_inx.append([start_inx, start_inx+size])
                    all_input_ids.extend([0]*size)
                    idx += 1

        return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx}

    def process_multi_modal_prompt_training(self, text, input_images):
        if input_images is None or len(input_images) == 0:
            model_inputs = self.text_tokenizer(text)
            if model_inputs.input_ids[0] == 1:
                model_inputs.input_ids = model_inputs.input_ids[1:]
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)] 

        for i in range(0, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text) 
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        # total images must be the same as the number of image tags
        assert len(unique_image_ids) == len(input_images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"
        
        input_images = [input_images[x-1] for x in image_ids]
        
        all_input_ids = []
        img_inx = []
        idx = 0
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            if i != len(prompt_chunks) -1:
                if i % 2 == 0:
                    all_input_ids.extend([0])
                start_inx = len(all_input_ids)
                size = input_images[i].size(-2) *  input_images[i].size(-1) // 16 // 16
                img_inx.append([start_inx, start_inx+size])
                all_input_ids.extend([0]*size)

        return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx}

    def process_multi_modal_prompt_frame_block_training(self, text, input_images, frame_blocks):
        if input_images is None or len(input_images) == 0:
            model_inputs = self.text_tokenizer(text)
            if model_inputs.input_ids[0] == 1:
                model_inputs.input_ids = model_inputs.input_ids[1:]
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)] 

        for i in range(0, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text) 
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        # total images must be the same as the number of image tags
        assert len(unique_image_ids) == len(input_images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"
        
        input_images = [input_images[x-1] for x in image_ids]
        
        all_input_ids = []
        img_inx = []
        idx = 0
        for k, frame_block in enumerate(frame_blocks):
            if k != len(frame_blocks) - 1:
                for i in range(frame_block):
                    all_input_ids.extend(prompt_chunks[idx])
                    all_input_ids.extend([0])
                    start_inx = len(all_input_ids)
                    size = input_images[idx].size(-2) *  input_images[idx].size(-1) // 16 // 16
                    img_inx.append([start_inx, start_inx+size])
                    all_input_ids.extend([0]*size)
                    idx += 1
                for i in range(frame_block):
                    all_input_ids.extend(prompt_chunks[idx])
                    start_inx = len(all_input_ids)
                    size = input_images[idx].size(-2) *  input_images[idx].size(-1) // 16 // 16
                    img_inx.append([start_inx, start_inx+size])
                    all_input_ids.extend([0]*size)
                    idx += 1
            else:
                for i in range(frame_block):
                    all_input_ids.extend(prompt_chunks[idx])
                    all_input_ids.extend([0])
                    start_inx = len(all_input_ids)
                    size = input_images[idx].size(-2) *  input_images[idx].size(-1) // 16 // 16
                    img_inx.append([start_inx, start_inx+size])
                    all_input_ids.extend([0]*size)
                    idx += 1

        return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx}

    def add_prefix_instruction(self, prompt):
        assistant_prompt = '<|diffusion|>'
        prompt = f"{prompt}{assistant_prompt}"
        return prompt


    def __call__(self, 
                instructions: List[str], 
                input_images = None,
                height: int = 1024,
                width: int = 1024,
                use_img_cfg: bool = True,
                use_input_image_size_as_output: bool=False,
                ) -> Dict:

        if input_images is None:
            use_img_cfg = False
        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]
        
        input_data = []
        for i in range(len(instructions)):
            cur_instruction = instructions[i]
            cur_input_images = None if input_images is None else input_images[i]
            if cur_input_images is not None and len(cur_input_images) > 0:
                cur_input_images = [self.process_image(x) for x in cur_input_images]
            else:
                cur_input_images = None
                assert "<img><|image_1|></img>" not in cur_instruction
            
            mllm_input = self.process_multi_modal_prompt(cur_instruction, cur_input_images)
            img_cfg_mllm_input = None
            if use_img_cfg:
                img_cfg_mllm_input = self.process_multi_modal_prompt("", None)

            if use_input_image_size_as_output:
                input_data.append((mllm_input, img_cfg_mllm_input, [mllm_input['pixel_values'][0].size(-2), mllm_input['pixel_values'][0].size(-1)]))
            else:
                input_data.append((mllm_input, img_cfg_mllm_input, [height, width]))

        return self.collator(input_data)

    def prompt_condition_inference(
            self, 
            instructions: List[str], 
            input_images = None,
            height: int = 1024,
            width: int = 1024,
            use_img_cfg: bool = True,
            use_input_image_size_as_output: bool=False,
            ) -> Dict:

        if input_images is None:
            use_img_cfg = False
        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]
        
        input_data = []

        cur_instruction = instructions[0]
        cur_input_images = None if input_images is None else input_images[0]
        if cur_input_images is not None and len(cur_input_images) > 0:
            cur_input_images = [self.process_image(x) for x in cur_input_images]
        else:
            cur_input_images = None
            assert "<img><|image_1|></img>" not in cur_instruction
        
        mllm_input = self.process_multi_modal_prompt(cur_instruction, cur_input_images)

        img_cfg_mllm_input = None
        if use_img_cfg:
            cur_instruction = instructions[1]
            cur_input_images = None if input_images is None else input_images[1]
            if cur_input_images is not None and len(cur_input_images) > 0:
                cur_input_images = [self.process_image(x) for x in cur_input_images]
            else:
                cur_input_images = None
                assert "<img><|image_1|></img>" not in cur_instruction
            
            img_cfg_mllm_input = self.process_multi_modal_prompt(cur_instruction, cur_input_images)

        if use_input_image_size_as_output:
            input_data.append((mllm_input, img_cfg_mllm_input, [mllm_input['pixel_values'][0].size(-2), mllm_input['pixel_values'][0].size(-1)]))
        else:
            input_data.append((mllm_input, img_cfg_mllm_input, [height, width]))

        return self.collator(input_data)

    def prompt_condition_frame_block_inference(
            self, 
            instructions: List[str], 
            input_images = None,
            height: int = 1024,
            width: int = 1024,
            use_img_cfg: bool = True,
            use_input_image_size_as_output: bool=False,
            frame_blocks: List[int]=None,
            ) -> Dict:

        if input_images is None:
            use_img_cfg = False
        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]
        
        input_data = []

        cur_instruction = instructions[0]
        cur_input_images = None if input_images is None else input_images[0]
        if cur_input_images is not None and len(cur_input_images) > 0:
            cur_input_images = [self.process_image(x) for x in cur_input_images]
        else:
            cur_input_images = None
            assert "<img><|image_1|></img>" not in cur_instruction
        
        mllm_input = self.process_multi_modal_prompt_frame_block(
            cur_instruction, 
            cur_input_images, 
            frame_blocks
            )
        mllm_input["frame_blocks"] = frame_blocks
        input_data.append(mllm_input)

        img_cfg_mllm_input = None
        if use_img_cfg:
            cur_instruction = instructions[1]
            cur_input_images = None if input_images is None else input_images[1]
            if cur_input_images is not None and len(cur_input_images) > 0:
                cur_input_images = [self.process_image(x) for x in cur_input_images]
            else:
                cur_input_images = None
                assert "<img><|image_1|></img>" not in cur_instruction
            
            img_cfg_mllm_input = self.process_multi_modal_prompt_frame_block(
                cur_instruction, 
                cur_input_images, 
                [0, frame_blocks[-1]], 
                height=mllm_input["pixel_values"][0].size(-2),
                width=mllm_input["pixel_values"][0].size(-1),
                )
            img_cfg_mllm_input["frame_blocks"] = [0, frame_blocks[-1]]
            input_data.append(img_cfg_mllm_input)

        return self.collator.process_mllm_input_frame_block_call(input_data)




class LVMCollator:
    def __init__(self, pad_token_id=2, hidden_size=3072, sequence_parallel_size=1):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.sequence_parallel_size = sequence_parallel_size
    
    def create_position(self, attention_mask, num_tokens_for_output_images):
        position_ids = []
        text_length = attention_mask.size(-1)
        img_length = max(num_tokens_for_output_images)  
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            temp_position = [0]*(text_length-temp_l) + [i for i in range(temp_l+img_length+1)] # we add a time embedding into the sequence, so add one more token
            position_ids.append(temp_position)
        return torch.LongTensor(position_ids)

    def create_position_training(self, image_sizes):
        position_ids = []
        block_ls = []
        for b_inx in image_sizes.keys():
            temp_position = []
            # 第一张图前面的token是<|diffusion|>和time embedding，再前面的就都是padding
            pad_l = image_sizes[b_inx][0][0] - 2
            token_l = image_sizes[b_inx][-1][-1] - pad_l
            # 计算attention中的图片数量
            assert token_l % len(image_sizes[b_inx]) == 0
            block_l = token_l // len(image_sizes[b_inx])
            block_ls.append(block_l)
            start_inx = 0
            temp_position.extend([0]*pad_l)
            for i, image_size in enumerate(image_sizes[b_inx]):
                if i == 0:
                    temp_position.extend([i for i in range(start_inx, start_inx+block_l)])
                elif i % 2 == 0:
                    temp_position.extend([i for i in range(start_inx, start_inx+block_l)])
                    start_inx += block_l
                    temp_position.extend([i for i in range(start_inx, start_inx+block_l)])
                else:
                    continue
            position_ids.append(temp_position)

        return torch.LongTensor(position_ids), block_ls

    def create_position_frame_block_training(self, image_sizes, frame_blocks):
        position_ids = []
        block_ls = []
        for b_inx in image_sizes.keys():
            temp_position = []
            # 第一张图前面的token是<|diffusion|>和time embedding，再前面的就都是padding
            pad_l = image_sizes[b_inx][0][0] - 2
            token_l = image_sizes[b_inx][-1][-1] - pad_l
            # 计算attention中的图片数量
            assert token_l % len(image_sizes[b_inx]) == 0
            block_l = token_l // len(image_sizes[b_inx])
            block_ls.append(block_l)
            diffusion_start_inx = 0
            clean_start_inx = 0
            temp_position.extend([0]*pad_l)

            for k, frame_block in enumerate(frame_blocks[b_inx]):
                if k != len(frame_blocks[b_inx]) - 1:
                    for i in range(frame_block):
                        temp_position.extend([i for i in range(diffusion_start_inx, diffusion_start_inx+block_l)])
                        diffusion_start_inx += block_l
                    for i in range(frame_block):
                        temp_position.extend([i for i in range(clean_start_inx, clean_start_inx+block_l)])
                        clean_start_inx += block_l
                else:
                    for i in range(frame_block):
                        temp_position.extend([i for i in range(diffusion_start_inx, diffusion_start_inx+block_l)])
                        diffusion_start_inx += block_l

            position_ids.append(temp_position)

        return torch.LongTensor(position_ids), block_ls
    
    def create_position_frame_block_inference(self, image_sizes, frame_blocks):
        position_ids = []
        block_ls = []
        for b_inx in image_sizes.keys():
            temp_position = []
            # 第一张图前面的token是<|diffusion|>和time embedding，再前面的就都是padding
            if b_inx == 0:
                pad_l = image_sizes[b_inx][0][0] - 1
            else:
                pad_l = image_sizes[b_inx][0][0] - 2
            token_l = image_sizes[b_inx][-1][-1] - pad_l
            # 计算attention中的图片数量
            assert token_l % len(image_sizes[b_inx]) == 0
            block_l = token_l // len(image_sizes[b_inx])
            block_ls.append(block_l)
            diffusion_start_inx = 0
            clean_start_inx = 0
            temp_position.extend([0]*pad_l)

            for k, frame_block in enumerate(frame_blocks[b_inx]):
                if k != len(frame_blocks[b_inx]) - 1:
                    for i in range(frame_block):
                        temp_position.extend([i for i in range(clean_start_inx, clean_start_inx+block_l)])
                        clean_start_inx += block_l
                else:
                    diffusion_start_inx = clean_start_inx
                    for i in range(frame_block):
                        temp_position.extend([i for i in range(diffusion_start_inx, diffusion_start_inx+block_l)])
                        diffusion_start_inx += block_l

            position_ids.append(temp_position)

        return torch.LongTensor(position_ids), block_ls
    
    def create_mask(self, attention_mask, num_tokens_for_output_images):
        extended_mask = []
        padding_images = []
        text_length = attention_mask.size(-1)
        img_length = max(num_tokens_for_output_images)
        seq_len = text_length + img_length + 1 # we add a time embedding into the sequence, so add one more token
        inx = 0
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            pad_l = text_length - temp_l

            temp_mask = torch.tril(torch.ones(size=(temp_l+1, temp_l+1)))

            image_mask = torch.zeros(size=(temp_l+1, img_length))
            temp_mask = torch.cat([temp_mask, image_mask], dim=-1)

            image_mask = torch.ones(size=(img_length, temp_l+img_length+1))
            temp_mask = torch.cat([temp_mask, image_mask], dim=0)

            if pad_l > 0:
                pad_mask = torch.zeros(size=(temp_l+1+img_length, pad_l))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=-1)

                pad_mask = torch.ones(size=(pad_l, seq_len))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=0)

            true_img_length = num_tokens_for_output_images[inx]
            pad_img_length = img_length - true_img_length
            if pad_img_length > 0:
                temp_mask[:, -pad_img_length:] = 0
                temp_padding_imgs = torch.zeros(size=(1, pad_img_length, self.hidden_size))
            else:
                temp_padding_imgs = None
            
            extended_mask.append(temp_mask.unsqueeze(0))
            padding_images.append(temp_padding_imgs)
            inx += 1
        return torch.cat(extended_mask, dim=0).to(dtype=torch.uint8), padding_images

    def create_mask_training(self, attention_mask, block_ls):
        extended_mask = []
        seq_len = attention_mask.size(-1)
        inx = 0
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            pad_l = seq_len - temp_l
            block_l = block_ls[inx]
            image_num = temp_l // block_l // 2 + 1

            temp_mask = torch.tril(torch.ones(size=(temp_l, temp_l)))
            block_start_idx = 0
            block_end_idx = block_l
            block_img_start_idx = 2  # 0, 1是<|diffusion|>和time embedding
            block_img_end_idx = block_l
            for i in range(image_num):
                if i != image_num - 1:
                    temp_mask[block_end_idx:, block_start_idx:block_end_idx] = 0
                    temp_mask[block_img_start_idx:block_img_end_idx, block_img_start_idx:block_img_end_idx] = 1
                    block_start_idx += block_l
                    block_end_idx += block_l
                    block_img_start_idx = block_start_idx + 1
                    block_img_end_idx = block_end_idx - 1
                    temp_mask[block_img_start_idx:, block_img_start_idx:block_img_end_idx] = 1
                    block_start_idx += block_l
                    block_end_idx += block_l
                    block_img_start_idx = block_start_idx + 2
                    block_img_end_idx = block_end_idx
                else:
                    temp_mask[block_end_idx:, block_start_idx:block_end_idx] = 0
                    temp_mask[block_img_start_idx:block_img_end_idx, block_img_start_idx:block_img_end_idx] = 1

            if pad_l > 0:
                pad_mask = torch.zeros(size=(temp_l, pad_l))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=-1)

                pad_mask = torch.ones(size=(pad_l, seq_len))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=0)

            extended_mask.append(temp_mask.unsqueeze(0))
            inx += 1
        return torch.cat(extended_mask, dim=0).to(dtype=torch.bool)

    def create_mask_frame_block_training(self, attention_mask, block_ls, frame_blocks):
        extended_mask = []
        seq_len = attention_mask.size(-1)
        inx = 0
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            pad_l = seq_len - temp_l
            block_l = block_ls[inx]
            image_num = temp_l // block_l // 2 + 1

            temp_mask = torch.zeros(size=(temp_l, temp_l), dtype=torch.uint8)
            block_start_idx_row = 0
            block_end_idx_row = block_l
            block_start_idx_col = 0  # 0, 1是<|diffusion|>和time embedding
            block_end_idx_col = block_l
            for k, frame_block in enumerate(frame_blocks[inx]):
                if k != len(frame_blocks[inx]) - 1:
                    for i in range(frame_block):
                        temp_mask[block_start_idx_row:block_end_idx_row, block_start_idx_col] = 1
                        temp_mask[block_start_idx_row+1:block_end_idx_row, block_start_idx_col+1] = 1
                        temp_mask[block_start_idx_row+2:block_end_idx_row, block_start_idx_col+2:block_end_idx_col] = 1
                        block_start_idx_col += block_l
                        block_end_idx_col += block_l
                    block_start_idx_row += block_l
                    block_end_idx_row += block_l
                    for i in range(frame_block-1):
                        temp_mask[block_start_idx_row:block_end_idx_row, block_start_idx_col-frame_block*block_l:block_start_idx_col] = \
                        temp_mask[block_start_idx_row-block_l:block_end_idx_row-block_l, block_start_idx_col-frame_block*block_l:block_start_idx_col]
                        block_start_idx_row += block_l
                        block_end_idx_row += block_l
                    for i in range(frame_block):
                        temp_mask[block_start_idx_row:, block_start_idx_col] = 1
                        temp_mask[block_start_idx_row+1:, block_start_idx_col+1:block_end_idx_col-1] = 1
                        temp_mask[block_end_idx_row-1:, block_end_idx_col-1] = 1
                        block_start_idx_col += block_l
                        block_end_idx_col += block_l
                        block_start_idx_row += block_l
                        block_end_idx_row += block_l
                else:
                    for i in range(frame_block):
                        temp_mask[block_start_idx_row:block_end_idx_row, block_start_idx_col] = 1
                        temp_mask[block_start_idx_row+1:block_end_idx_row, block_start_idx_col+1] = 1
                        temp_mask[block_start_idx_row+2:block_end_idx_row, block_start_idx_col+2:block_end_idx_col] = 1
                        block_start_idx_col += block_l
                        block_end_idx_col += block_l
                    block_start_idx_row += block_l
                    block_end_idx_row += block_l
                    for i in range(frame_block-1):
                        temp_mask[block_start_idx_row:block_end_idx_row, block_start_idx_col-frame_block*block_l:block_start_idx_col] = \
                        temp_mask[block_start_idx_row-block_l:block_end_idx_row-block_l, block_start_idx_col-frame_block*block_l:block_start_idx_col]
                        block_start_idx_row += block_l
                        block_end_idx_row += block_l
                        
            if pad_l > 0:
                pad_mask = torch.zeros(size=(temp_l, pad_l), dtype=torch.uint8)
                temp_mask = torch.cat([pad_mask, temp_mask], dim=-1)

                pad_mask = torch.ones(size=(pad_l, seq_len), dtype=torch.uint8)
                temp_mask = torch.cat([pad_mask, temp_mask], dim=0)

            extended_mask.append(temp_mask.unsqueeze(0))
            inx += 1
        return torch.cat(extended_mask, dim=0).to(dtype=torch.bool)

    def create_mask_frame_block_inference(self, attention_mask, block_ls, frame_blocks):
        extended_mask = []
        seq_len = attention_mask.size(-1)
        inx = 0
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            pad_l = seq_len - temp_l
            block_l = block_ls[inx]
            image_num = temp_l // block_l // 2 + 1

            temp_mask = torch.zeros(size=(temp_l, temp_l))
            block_start_idx_row = 0
            block_end_idx_row = block_l
            block_start_idx_col = 0  # 0, 1是<|diffusion|>和time embedding
            block_end_idx_col = block_l
            for k, frame_block in enumerate(frame_blocks[inx]):
                if k != len(frame_blocks[inx]) - 1:
                    for i in range(frame_block):
                        temp_mask[block_start_idx_row:, block_start_idx_col] = 1
                        temp_mask[block_start_idx_row+1:, block_start_idx_col+1:block_end_idx_col-1] = 1
                        temp_mask[block_end_idx_row-1:, block_end_idx_col-1] = 1
                        block_start_idx_col += block_l
                        block_end_idx_col += block_l
                        block_start_idx_row += block_l
                        block_end_idx_row += block_l
                else:
                    for i in range(frame_block):
                        temp_mask[block_start_idx_row:block_end_idx_row, block_start_idx_col] = 1
                        temp_mask[block_start_idx_row+1:block_end_idx_row, block_start_idx_col+1] = 1
                        temp_mask[block_start_idx_row+2:block_end_idx_row, block_start_idx_col+2:block_end_idx_col] = 1
                        block_start_idx_col += block_l
                        block_end_idx_col += block_l
                    block_start_idx_row += block_l
                    block_end_idx_row += block_l
                    for i in range(frame_block-1):
                        temp_mask[block_start_idx_row:block_end_idx_row, block_start_idx_col-frame_block*block_l:block_start_idx_col] = \
                        temp_mask[block_start_idx_row-block_l:block_end_idx_row-block_l, block_start_idx_col-frame_block*block_l:block_start_idx_col]
                        block_start_idx_row += block_l
                        block_end_idx_row += block_l

            if pad_l > 0:
                pad_mask = torch.zeros(size=(temp_l, pad_l))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=-1)

                pad_mask = torch.ones(size=(pad_l, seq_len))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=0)

            extended_mask.append(temp_mask.unsqueeze(0))
            inx += 1
        return torch.cat(extended_mask, dim=0).to(dtype=torch.bool)

    def create_block_mask_training(self, attention_mask, block_ls):
        extended_mask = []
        seq_len = attention_mask.size(-1)
        inx = 0
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            pad_l = seq_len - temp_l
            block_l = block_ls[inx]
            image_num = temp_l // block_l // 2 + 1

            temp_mask = torch.tril(torch.ones(size=(temp_l, temp_l)))
            block_start_idx = 0
            block_end_idx = block_l
            block_img_start_idx = block_start_idx  # 0, 1是<|diffusion|>和time embedding
            block_img_end_idx = block_end_idx
            for i in range(image_num):
                if i != image_num - 1:
                    temp_mask[block_end_idx:, block_start_idx:block_end_idx] = 0
                    temp_mask[block_img_start_idx:block_img_end_idx, block_img_start_idx:block_img_end_idx] = 1
                    block_start_idx += block_l
                    block_end_idx += block_l
                    block_img_start_idx = block_start_idx
                    block_img_end_idx = block_end_idx
                    temp_mask[block_img_start_idx:, block_img_start_idx:block_img_end_idx] = 1
                    block_start_idx += block_l
                    block_end_idx += block_l
                    block_img_start_idx = block_start_idx
                    block_img_end_idx = block_end_idx
                else:
                    temp_mask[block_end_idx:, block_start_idx:block_end_idx] = 0
                    temp_mask[block_img_start_idx:block_img_end_idx, block_img_start_idx:block_img_end_idx] = 1

            if pad_l > 0:
                pad_mask = torch.zeros(size=(temp_l, pad_l))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=-1)

                pad_mask = torch.ones(size=(pad_l, seq_len))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=0)

            extended_mask.append(temp_mask.unsqueeze(0))
            inx += 1
        return torch.cat(extended_mask, dim=0).to(dtype=torch.uint8)
    
    def adjust_attention_for_input_images(self, attention_mask, image_sizes):
        for b_inx in image_sizes.keys():
            for start_inx, end_inx in image_sizes[b_inx]:
                attention_mask[b_inx][start_inx:end_inx, start_inx:end_inx] = 1

        return attention_mask
    
    def pad_input_ids(self, input_ids, image_sizes, num_tokens_for_output_images):
        max_l = max([len(input_ids) + num_tokens_for_output_images[i] + 1 for i, input_ids in enumerate(input_ids)])
        if max_l % self.sequence_parallel_size != 0:
            pub_pad_l = self.sequence_parallel_size - max_l % self.sequence_parallel_size
            max_l += pub_pad_l
        padded_ids = []
        attention_mask = []
        new_image_sizes = []

        for i in range(len(input_ids)):
            temp_ids = input_ids[i]
            temp_l = len(temp_ids)
            pad_l = max_l - temp_l - num_tokens_for_output_images[i] - 1
            if pad_l == 0:
                attention_mask.append([1]*temp_l)
                padded_ids.append(temp_ids)
            else:
                attention_mask.append([0]*pad_l+[1]*temp_l)
                padded_ids.append([self.pad_token_id]*pad_l+temp_ids)
            
            if i in image_sizes:
                new_inx = []
                for old_inx in image_sizes[i]:
                    new_inx.append([x+pad_l for x in old_inx])
                image_sizes[i] = new_inx
        
        return torch.LongTensor(padded_ids), torch.ByteTensor(attention_mask), image_sizes


    def pad_input_ids_training(self, input_ids, image_sizes):
        max_l = max([len(input_ids) for i, input_ids in enumerate(input_ids)])
        if max_l % self.sequence_parallel_size != 0:
            pub_pad_l = self.sequence_parallel_size - max_l % self.sequence_parallel_size
            max_l += pub_pad_l
        padded_ids = []
        attention_mask = []
        new_image_sizes = []

        for i in range(len(input_ids)):
            temp_ids = input_ids[i]
            temp_l = len(temp_ids)
            pad_l = max_l - temp_l
            if pad_l == 0:
                attention_mask.append([1]*max_l)
                padded_ids.append(temp_ids)
            else:
                attention_mask.append([0]*pad_l+[1]*temp_l)
                padded_ids.append([self.pad_token_id]*pad_l+temp_ids)
            
            if i in image_sizes:
                new_inx = []
                for old_inx in image_sizes[i]:
                    new_inx.append([x+pad_l for x in old_inx])
                image_sizes[i] = new_inx

        return torch.LongTensor(padded_ids), torch.ByteTensor(attention_mask), image_sizes


    def process_mllm_input(self, mllm_inputs, target_img_size):
        num_tokens_for_output_images = []
        for img_size in target_img_size:
            num_tokens_for_output_images.append(img_size[0]*img_size[1]//16//16)

        pixel_values, image_sizes = [], {}
        b_inx = 0
        for x in mllm_inputs:
            if x['pixel_values'] is not None:
                pixel_values.extend(x['pixel_values'])
                for size in x['image_sizes']:
                    if b_inx not in image_sizes:
                        image_sizes[b_inx] = [size]
                    else:
                        image_sizes[b_inx].append(size)
            b_inx += 1     
        pixel_values = [x.unsqueeze(0) for x in pixel_values]

        
        input_ids = [x['input_ids'] for x in mllm_inputs]
        padded_input_ids, attention_mask, image_sizes = self.pad_input_ids(input_ids, image_sizes, num_tokens_for_output_images)
        position_ids = self.create_position(attention_mask, num_tokens_for_output_images)
        attention_mask, padding_images = self.create_mask(attention_mask, num_tokens_for_output_images)
        attention_mask = self.adjust_attention_for_input_images(attention_mask, image_sizes)

        return padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes

    
    def process_mllm_input_training(self, mllm_inputs, block_aware=False):
        pixel_values, image_sizes = [], {}
        b_inx = 0
        for x in mllm_inputs:
            if x['pixel_values'] is not None:
                pixel_values.extend(x['pixel_values'])
                for size in x['image_sizes']:
                    if b_inx not in image_sizes:
                        image_sizes[b_inx] = [size]
                    else:
                        image_sizes[b_inx].append(size)
            b_inx += 1     
        pixel_values = [x.unsqueeze(0) for x in pixel_values]

        
        input_ids = [x['input_ids'] for x in mllm_inputs]
        padded_input_ids, attention_mask, image_sizes = self.pad_input_ids_training(input_ids, image_sizes)
        position_ids, block_ls = self.create_position_training(image_sizes)
        if not block_aware:
            attention_mask = self.create_mask_training(attention_mask, block_ls)
        else:
            attention_mask = self.create_block_mask_training(attention_mask, block_ls)
        return padded_input_ids, position_ids, attention_mask, pixel_values, image_sizes

    def process_mllm_input_frame_block_training(self, mllm_inputs, block_aware=False):
        pixel_values, image_sizes, frame_blocks = [], {}, {}
        b_inx = 0
        for x in mllm_inputs:
            if x['pixel_values'] is not None:
                pixel_values.extend(x['pixel_values'])
                for size in x['image_sizes']:
                    if b_inx not in image_sizes:
                        image_sizes[b_inx] = [size]
                        frame_blocks[b_inx] = x['frame_blocks']
                    else:
                        image_sizes[b_inx].append(size)
                        frame_blocks[b_inx] = x['frame_blocks']
            b_inx += 1     
        pixel_values = [x.unsqueeze(0) for x in pixel_values]
        
        input_ids = [x['input_ids'] for x in mllm_inputs]
        padded_input_ids, attention_mask, image_sizes = self.pad_input_ids_training(input_ids, image_sizes)
        position_ids, block_ls = self.create_position_frame_block_training(image_sizes, frame_blocks)
        attention_mask = self.create_mask_frame_block_training(attention_mask, block_ls, frame_blocks)

        return padded_input_ids, position_ids, attention_mask, pixel_values, image_sizes, frame_blocks 

    def process_mllm_input_frame_block_inference(self, mllm_inputs, block_aware=False):
        pixel_values, image_sizes, frame_blocks = [], {}, {}
        b_inx = 0
        for x in mllm_inputs:
            if x['pixel_values'] is not None:
                pixel_values.extend(x['pixel_values'])
                for size in x['image_sizes']:
                    if b_inx not in image_sizes:
                        image_sizes[b_inx] = [size]
                        frame_blocks[b_inx] = x['frame_blocks']
                    else:
                        image_sizes[b_inx].append(size)
                        frame_blocks[b_inx] = x['frame_blocks']
            b_inx += 1     
        pixel_values = [x.unsqueeze(0) for x in pixel_values]

        
        input_ids = [x['input_ids'] for x in mllm_inputs]
        padded_input_ids, attention_mask, image_sizes = self.pad_input_ids_training(input_ids, image_sizes)
        position_ids, block_ls = self.create_position_frame_block_inference(image_sizes, frame_blocks)
        # if not block_aware:
        attention_mask = self.create_mask_frame_block_inference(attention_mask, block_ls, frame_blocks)
        # else:
        #     attention_mask = self.create_block_mask_training(attention_mask, block_ls)

        return padded_input_ids, position_ids, attention_mask, pixel_values, image_sizes, frame_blocks 
    
    def __call__(self, features):
        mllm_inputs = [f[0] for f in features]
        img_cfg_mllm_input = [f[1] for f in features]
        target_img_size = [f[2] for f in features]

        
        if img_cfg_mllm_input[0] is not None:
            mllm_inputs = mllm_inputs + img_cfg_mllm_input
            target_img_size = target_img_size + target_img_size

        all_padded_input_ids, all_position_ids, all_attention_mask, all_padding_images, all_pixel_values, all_image_sizes = self.process_mllm_input(mllm_inputs, target_img_size)

        data = {"input_ids": all_padded_input_ids,
        "attention_mask": all_attention_mask,
        "position_ids": all_position_ids,
        "input_pixel_values": all_pixel_values,
        "input_image_sizes": all_image_sizes,
        "padding_images": all_padding_images,
        }
        return data

    def process_mllm_input_frame_block_call(self, features):
        mllm_inputs = features
        all_padded_input_ids, all_position_ids, all_attention_mask, all_pixel_values, all_image_sizes, frame_blocks = self.process_mllm_input_frame_block_inference(mllm_inputs)

        denoise_image_sizes = {}
        input_image_sizes = {}
        time_emb_inx = {}
        output_images = []
        input_images = []
        for b_inx in all_image_sizes.keys():
            denoise_image_sizes[b_inx] = []
            input_image_sizes[b_inx] = []
            time_emb_inx[b_inx] = []
            idx = 0
            for k, frame_block in enumerate(frame_blocks[b_inx]):
                if k != len(frame_blocks[b_inx]) - 1:
                    for i in range(frame_block):
                        input_image_sizes[b_inx].append(all_image_sizes[b_inx][idx])
                        input_images.append(all_pixel_values[idx])
                        idx += 1
                else:
                    for i in range(frame_block):
                        denoise_image_sizes[b_inx].append(all_image_sizes[b_inx][idx])
                        time_emb_inx[b_inx].append(all_image_sizes[b_inx][idx][0]-1)
                        idx += 1

        data = {"input_ids": all_padded_input_ids,
                "attention_mask": all_attention_mask,
                "position_ids": all_position_ids,
                "input_pixel_values": input_images,
                "input_image_sizes": input_image_sizes,
                "denoise_image_sizes": denoise_image_sizes,
                "output_images": output_images,
                "time_emb_inx": time_emb_inx,
                "frame_blocks": frame_blocks,
                }
        return data

