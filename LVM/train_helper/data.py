import os
import re
import cv2
import json
import copy
import time
import math
import torch
import pickle 
import random
import datasets
import argparse
import subprocess
import numpy as np
from PIL import Image
from decord import cpu, gpu
from decord import VideoReader
import torch.distributed as dist
from torchvision import transforms
from datasets import ClassLabel, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from LVM.acceleration.parallel_states import init_npu_env, hccl_info

from LVM import LVMProcessor
from LVM.processor import LVMCollator

from LVM.utils import (
    crop_arr,
    center_crop_arr,
)


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


def get_video_files(video_dir_path):
    # 常见的视频文件扩展名
    video_extensions = (
        '.mp4', '.avi', '.mkv', '.flv', '.mov', '.wmv', '.mpg', '.mpeg',
        '.3gp', '.m4v', '.ts', '.webm', '.vob', '.rm', '.rmvb', '.ogv',
        '.ogg', '.drc', '.mng', '.qt', '.f4v', '.f4p', '.f4a', '.f4b',
        '.asf', '.amv', '.divx', '.mk3d', '.mts', '.m2ts', '.vob', '.ogm',
        '.svi', '.gifv', '.mxf', '.roq', '.nsv', '.viv', '.wtv', '.yuv'
    )
    
    video_files = []
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                full_path = os.path.abspath(os.path.join(root, file))
                video_files.append(full_path)
    return video_files


def generate_random_list(num_frames):
    """
    随机生成一个列表，列表中所有整数均大于0，并且列表元素之和为 num_frames。

    参数:
        num_frames (int): 正整数，总和

    返回:
        list: 满足条件的随机列表
    """
    if num_frames <= 0:
        raise ValueError("num_frames 必须为正整数")

    # 随机确定列表的长度 k (1 到 num_frames 之间)
    k = random.randint(2, num_frames)

    # 从区间 [1, num_frames-1] 中随机选择 k-1 个不重复的切割点
    cuts = sorted(random.sample(range(1, num_frames), k-1))
    
    # 利用切割点把 num_frames 分成 k 段，每段的大小都大于 0
    parts = []
    previous = 0
    for cut in cuts:
        parts.append(cut - previous)
        previous = cut
    parts.append(num_frames - previous)
    
    return parts


class DatasetFromVideo(torch.utils.data.Dataset):
    def __init__(
        self,
        video_dir_path: str,
        processer: LVMProcessor,
        image_transform,
        max_input_length_limit: int = 18000,
        keep_raw_resolution: bool = True, 
        max_retry_times: int = 1000,
        frame_num: int = 3,
        frame_interval: int = 1,
        data_reuse: int = 1,
        accelerator=None,
        data_limit=None,
    ):
        
        self.image_transform = image_transform
        self.processer = processer
        self.max_input_length_limit = max_input_length_limit
        self.keep_raw_resolution = keep_raw_resolution
        self.max_retry_times = max_retry_times
        self.frame_num = frame_num
        self.frame_interval = frame_interval

         # 判断传入路径是否是以 .txt 结尾的文件
        if os.path.isfile(video_dir_path) and video_dir_path.lower().endswith('.txt'):
            # 从 txt 文件中读取每一行（去掉前后空白）
            with open(video_dir_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            valid_data = []

            # 计算采样所需最少帧数
            min_frames_needed = (self.frame_num - 1) * self.frame_interval + 1
            for line in lines:
                parts = line.split('\t')
                if len(parts) < 2:
                    # 如果一行里没有帧数数据，跳过
                    continue
                video_path, str_frame_count = parts[0], parts[1]
                try:
                    frame_count = int(str_frame_count)
                except ValueError:
                    # 帧数无法转换成int，跳过
                    continue
                
                if frame_count >= min_frames_needed:
                    # 满足最少帧数才保留
                    valid_data.append(video_path)
            
            self.data = valid_data
        else:
            # 如果不是 .txt 文件，则认为是文件夹，递归获取所有视频文件
            self.data = get_video_files(video_dir_path)

        # 复制数据列表 data_reuse 次
        self.data = self.data * data_reuse

        # 如果有数据限制，截取数据
        if data_limit is not None:
            if data_limit < len(self.data):
                self.data = self.data[:data_limit]
            else:
                print(f"数据量不足，数据量={len(self.data)}, data_limit={data_limit}")

        if accelerator is not None:
            block_data = len(self.data) // accelerator.num_processes
            if accelerator.process_index != accelerator.num_processes - 1:
                self.data = self.data[accelerator.process_index * block_data: (accelerator.process_index + 1) * block_data]
            else:
                self.data = self.data[accelerator.process_index * block_data:]

        self.accelerator = accelerator

    def process_image(self, image_file):
        return self.image_transform(image_file)

    def get_example(self, index):
        example = self.data[index]
        cap = cv2.VideoCapture(example)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 计算最大起始帧索引
        max_start_frame = total_frames - (self.frame_num - 1) * self.frame_interval
        if max_start_frame <= 0:
            raise ValueError("视频帧数不足，无法按照指定的间隔和数量采样帧。")
        # 随机选择起始帧
        start_frame = random.randint(0, max_start_frame - 1)
        frame_indices = [start_frame + i * self.frame_interval for i in range(self.frame_num)]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                raise Exception(f"Failed to read frame {idx}")
        frames = np.array(frames)

        width, height = frames[0].shape[1], frames[0].shape[0]
        if width < 128 or height < 128:
            raise ValueError(f"视频分辨率过低，宽：{width}，高：{height}。需要至少长或宽大于 320 的分辨率。")
        if width / height > 4 or height / width > 4:
            raise ValueError(f"视频分辨率异常，宽：{width}，高：{height}。")

        instruction = ""
        input_images = []
        for i in range(frames.shape[0]):
            if i < frames.shape[0] - 1:
                instruction += f"<|diffusion|><|image_{i+1}|><img><|image_{i+1}|></img>"
            else:
                instruction += f"<|diffusion|><|image_{i+1}|>"
            input_images.append(Image.fromarray(frames[i]))
    
        if input_images is not None:
            input_images = [self.process_image(x) for x in input_images]
        
        mllm_input = self.processer.process_multi_modal_prompt_training(instruction, input_images)
        cap.release()
            
        return mllm_input

    def __getitem__(self, index):
        for _ in range(self.max_retry_times):
            try:
                mllm_input = self.get_example(index)
                if len(mllm_input['input_ids']) + self.processer.sequence_parallel_size - 1 > self.max_input_length_limit:
                    raise RuntimeError(f"cur number of tokens={len(mllm_input['input_ids'])}, larger than max_input_length_limit={self.max_input_length_limit}")
                return mllm_input
            except Exception as e:
                print("error when loading data: ", e)
                print(self.data[index])
                index = random.randint(0, len(self.data)-1)
        raise RuntimeError("Too many bad data.")
    

    def __len__(self):
        return len(self.data)


class DatasetFromVideoBlockFrame(torch.utils.data.Dataset):
    def __init__(
        self,
        video_dir_path: str,
        processer: LVMProcessor,
        image_transform,
        max_input_length_limit: int = 18000,
        keep_raw_resolution: bool = True, 
        max_retry_times: int = 1000,
        frame_num: int = 3,
        frame_interval: int = 1,
        data_reuse: int = 1,
        accelerator=None,
        data_limit=None,
        flexible_interval=False,
        interval_bound=None,
    ):
        
        self.image_transform = image_transform
        self.processer = processer
        self.max_input_length_limit = max_input_length_limit
        self.keep_raw_resolution = keep_raw_resolution
        self.max_retry_times = max_retry_times
        self.frame_num = frame_num
        self.frame_interval = frame_interval
        self.flexible_interval = flexible_interval
        self.interval_bound = interval_bound

         # 判断传入路径是否是以 .txt 结尾的文件
        if os.path.isfile(video_dir_path) and video_dir_path.lower().endswith('.txt'):
            # 从 txt 文件中读取每一行（去掉前后空白）
            with open(video_dir_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            valid_data = []

            # 计算采样所需最少帧数
            min_frames_needed = (self.frame_num - 1) * self.frame_interval + 1
            for line in lines:
                parts = line.split('\t')
                if len(parts) < 2:
                    # 如果一行里没有帧数数据，跳过
                    continue
                video_path, str_frame_count = parts[0], parts[1]
                try:
                    frame_count = int(str_frame_count)
                except ValueError:
                    # 帧数无法转换成int，跳过
                    continue
                
                if frame_count >= min_frames_needed:
                    # 满足最少帧数才保留
                    valid_data.append(video_path)
            
            self.data = valid_data
        else:
            # 如果不是 .txt 文件，则认为是文件夹，递归获取所有视频文件
            self.data = get_video_files(video_dir_path)

        # 复制数据列表 data_reuse 次
        self.data = self.data * data_reuse

        # 如果有数据限制，截取数据
        if data_limit is not None:
            if data_limit < len(self.data):
                self.data = self.data[:data_limit]
            else:
                print(f"数据量不足，数据量={len(self.data)}, data_limit={data_limit}")

        if accelerator is not None:
            block_data = len(self.data) // accelerator.num_processes
            if accelerator.process_index != accelerator.num_processes - 1:
                self.data = self.data[accelerator.process_index * block_data: (accelerator.process_index + 1) * block_data]
            else:
                self.data = self.data[accelerator.process_index * block_data:]
        self.accelerator = accelerator

    def process_image(self, image_file):
        return self.image_transform(image_file)

    def get_example(self, index):
        example = self.data[index]
        cap = cv2.VideoCapture(example)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 计算最大起始帧索引
        if self.flexible_interval:
            max_frame_interval = (total_frames - 1) // (self.frame_num - 1)
            if max_frame_interval < self.frame_interval:
                raise ValueError("视频帧数不足")
            else:
                if self.interval_bound is not None:
                    max_frame_interval = min(max_frame_interval, self.interval_bound)
                frame_interval = random.randint(self.frame_interval, max_frame_interval)
        else:
            frame_interval = self.frame_interval

        max_start_frame = total_frames - (self.frame_num - 1) * frame_interval
        if max_start_frame <= 0:
            raise ValueError("视频帧数不足，无法按照指定的间隔和数量采样帧。")
        # 随机选择起始帧
        start_frame = random.randint(0, max_start_frame - 1)
        frame_indices = [start_frame + i * frame_interval for i in range(self.frame_num)]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                raise Exception(f"Failed to read frame {idx}")
        frames = np.array(frames)
        frame_blocks = generate_random_list(self.frame_num)

        width, height = frames[0].shape[1], frames[0].shape[0]
        if width < 128 or height < 128:
            raise ValueError(f"视频分辨率过低，宽：{width}，高：{height}。需要至少长或宽大于 320 的分辨率。")
        if width / height > 4 or height / width > 4:
            raise ValueError(f"视频分辨率异常，宽：{width}，高：{height}。")

        instruction = ""
        input_images = []
        i = 0
        j = 0
        for k, frame_block in enumerate(frame_blocks):
            if k != len(frame_blocks) - 1:
                for frame in range(frame_block):
                    instruction += f"<|diffusion|><|image_{i+1}|>"
                    input_images.append(Image.fromarray(frames[i]))
                    i += 1
                for frame in range(frame_block):
                    instruction += f"<img><|image_{j+1}|></img>"
                    j += 1
            else:
                for frame in range(frame_block):
                    instruction += f"<|diffusion|><|image_{i+1}|>"
                    input_images.append(Image.fromarray(frames[i]))
                    i += 1
    
        if input_images is not None:
            input_images = [self.process_image(x) for x in input_images]
        
        mllm_input = self.processer.process_multi_modal_prompt_frame_block_training(instruction, input_images, frame_blocks)
        cap.release()
        mllm_input["frame_blocks"] = frame_blocks
            
        return mllm_input


    def __getitem__(self, index):
        for _ in range(self.max_retry_times):
            try:
                mllm_input = self.get_example(index)
                if len(mllm_input['input_ids']) + self.processer.sequence_parallel_size - 1 > self.max_input_length_limit:
                    raise RuntimeError(f"cur number of tokens={len(mllm_input['input_ids'])}, larger than max_input_length_limit={self.max_input_length_limit}")
                return mllm_input
            except Exception as e:
                print("error when loading data: ", e)
                print(self.data[index])
                index = random.randint(0, len(self.data)-1)
        raise RuntimeError("Too many bad data.")
    

    def __len__(self):
        return len(self.data)


class TrainDataCollator(LVMCollator):
    def __init__(self, pad_token_id: int, 
                 hidden_size: int, 
                 keep_raw_resolution: bool, 
                 frame_num: int,
                 sequence_parallel_size=1, 
                 batch_size=1,
                 block_aware=False,
                 ):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.keep_raw_resolution = keep_raw_resolution
        self.sequence_parallel_size = sequence_parallel_size
        self.batch_size = batch_size
        self.frame_num = frame_num
        self.block_aware = block_aware

    def __call__(self, features):
        try:
            mllm_inputs = features
            # 如果 batch 不足，重复填充样本
            current_mllm_inputs = copy.deepcopy(mllm_inputs)
            if len(mllm_inputs)==1 and len(mllm_inputs) < self.batch_size:
                current_mllm_inputs.extend(mllm_inputs * (self.batch_size - 1))  # 重复填充到目标大小
            mllm_inputs = current_mllm_inputs
            all_padded_input_ids, all_position_ids, all_attention_mask, all_pixel_values, all_image_sizes = self.process_mllm_input_training(mllm_inputs, block_aware=self.block_aware)
            if not self.keep_raw_resolution:
                output_images = torch.cat(output_images, dim=0)
                if len(all_pixel_values) > 0:
                    all_pixel_values = torch.cat(all_pixel_values, dim=0)
                else:
                    all_pixel_values = None

            denoise_image_sizes = {}
            input_image_sizes = {}
            time_emb_inx = {}
            for b_inx in all_image_sizes.keys():
                denoise_image_sizes[b_inx] = [size for i, size in enumerate(all_image_sizes[b_inx]) if i % 2 == 0]
                input_image_sizes[b_inx] = [size for i, size in enumerate(all_image_sizes[b_inx]) if i % 2 == 1]  
                time_emb_inx[b_inx] = [size[0]-1 for i, size in enumerate(all_image_sizes[b_inx]) if i % 2 == 0]
            
            output_images = [image for i, image in enumerate(all_pixel_values) if i % (2 * self.frame_num - 1) % 2 == 0]
            input_images = [image for i, image in enumerate(all_pixel_values) if i % (2 * self.frame_num - 1) % 2 == 1]

            data = {"input_ids": all_padded_input_ids,
                    "attention_mask": all_attention_mask,
                    "position_ids": all_position_ids,
                    "input_pixel_values": input_images,
                    "input_image_sizes": input_image_sizes,
                    "denoise_image_sizes": denoise_image_sizes,
                    "output_images": output_images,
                    "time_emb_inx": time_emb_inx,
                    }
            return data
        except Exception as e:
            print(f"Error in collate_fn: {e}")
            raise e


class TrainDataCollator_FrameBlock(LVMCollator):
    def __init__(self, pad_token_id: int, 
                 hidden_size: int, 
                 keep_raw_resolution: bool, 
                 frame_num: int,
                 sequence_parallel_size=1, 
                 batch_size=1,
                 block_aware=False,
                 ):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.keep_raw_resolution = keep_raw_resolution
        self.sequence_parallel_size = sequence_parallel_size
        self.batch_size = batch_size
        self.frame_num = frame_num
        self.block_aware = block_aware

    def __call__(self, features):
        try:
            mllm_inputs = features
            # 如果 batch 不足，重复填充样本
            current_mllm_inputs = copy.deepcopy(mllm_inputs)
            if len(mllm_inputs)==1 and len(mllm_inputs) < self.batch_size:
                current_mllm_inputs.extend(mllm_inputs * (self.batch_size - 1))  # 重复填充到目标大小
            mllm_inputs = current_mllm_inputs
            all_padded_input_ids, all_position_ids, all_attention_mask, all_pixel_values, all_image_sizes, frame_blocks = self.process_mllm_input_frame_block_training(mllm_inputs, block_aware=self.block_aware)

            if not self.keep_raw_resolution:
                output_images = torch.cat(output_images, dim=0)
                if len(all_pixel_values) > 0:
                    all_pixel_values = torch.cat(all_pixel_values, dim=0)
                else:
                    all_pixel_values = None

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
                            denoise_image_sizes[b_inx].append(all_image_sizes[b_inx][idx])
                            input_image_sizes[b_inx].append(all_image_sizes[b_inx][idx+frame_block])
                            time_emb_inx[b_inx].append(all_image_sizes[b_inx][idx][0]-1)
                            output_images.append(all_pixel_values[idx+b_inx*self.frame_num])
                            input_images.append(all_pixel_values[idx+frame_block+b_inx*self.frame_num])
                            idx += 1
                        idx += frame_block
                    else:
                        for i in range(frame_block):
                            denoise_image_sizes[b_inx].append(all_image_sizes[b_inx][idx])
                            time_emb_inx[b_inx].append(all_image_sizes[b_inx][idx][0]-1)
                            output_images.append(all_pixel_values[idx+b_inx*self.frame_num])
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
        except Exception as e:
            print(f"Error in collate_fn: {e}")
            raise e
