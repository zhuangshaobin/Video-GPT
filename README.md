<h1 align="center">Video-GPT via Next Clip Diffusion</h1>


<p align="center">
    <a href="https://vectorspacelab.github.io/OmniGen/">
        <img alt="Build" src="https://img.shields.io/badge/Project%20Page-OmniGen-yellow">
    </a>
    <a href="https://arxiv.org/abs/2409.11340">
            <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2409.11340-b31b1b.svg">
    </a>
    <a href="https://huggingface.co/GrayShine/Video-GPT">
        <img alt="Build" src="https://img.shields.io/badge/HF%20Model-🤗-yellow">
    </a>    
</p>

<h4 align="center">
    <p>
        <a href=#1-news>News</a> |
        <a href=#3-methodology>Methodology</a> |
        <a href=#4-quick-start>Quick Start</a> |
        <a href="#5-finetune">Finetune</a> |
        <a href="#license">License</a> |
        <a href="#citation">Citation</a>
    <p>
</h4>



## 1. News
- 2025-5-16:✨✨We release our 4 stages prograssive training code (supporting Huawei's NPU and NVIDIA's GPU). You can refer to [LVM/script/train](LVM/script/train) and [LVM/train](LVM/train) for detailed training information.
- 2025-5-16:✨✨We release the inference code in [LVM/script/inference](LVM/script/inference) and [LVM/inference](LVM/inference).
- 2025-5-16:🔥🔥We release the first version of Video-GPT. Model Weight: [Video-GPT](https://huggingface.co/GrayShine/Video-GPT)


## 2. Overview

Video-GPT is a video self-supervised generative pre-trained model which treats video as new language for visual world modeling by next clip diffusion. It is designed to be simple, flexible, and easy to follow. We provide [inference code](#5-quick-start) so that everyone can explore more functionalities of Video-GPT.

Previous works on visual generation relies heavily on supervisory signals from textual modalities (such as Sora, WanX, HunyuanVideo, MovieGen). However, vision, as a natural ability of human beings, was formed even earlier than language. Therefore, we believe that the information of the visual modality itself is sufficient to support the model to model the world.

In addition, compared with the previous model architecture with many special designs for diffusion model (e.g., UNet, DiT, MM-DiT), we adopted the simplest vanilla transformer architecture. On the one hand, it is more conducive to the exploration of scaling law in the future. On the other hand, it is also more convenient for the community to follow up.

Due to the limited resources, Video-GPT still has room for improvement. We will continue to optimize it, and hope it inspires more universal video generative foundation models. 

If you have any questions, ideas, or interesting tasks you want Video-GPT to accomplish, feel free to discuss with us: hahahahaha@sjtu.edu.cn. We welcome any feedback to help us improve the model.


## 3. Methodology

You can see details in our [paper](https://arxiv.org/abs/2409.11340). 



## 4. Quick Start


### Using OmniGen
Install via Github:
```bash
git clone https://github.com/VectorSpaceLab/OmniGen.git
cd OmniGen
pip install -e .
```

You also can create a new environment to avoid conflicts:
```bash
# Create a python 3.10.13 conda env (you could also use virtualenv)
conda create -n omnigen python=3.10.13
conda activate omnigen

# Install pytorch with your CUDA version, e.g.
pip install torch==2.3.1+cu118 torchvision --extra-index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/VectorSpaceLab/OmniGen.git
cd OmniGen
pip install -e .
```

Here are some examples:
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  
# Note: Your local model path is also acceptable, such as 'pipe = OmniGenPipeline.from_pretrained(your_local_model_path)', where all files in your_local_model_path should be organized as https://huggingface.co/Shitao/OmniGen-v1/tree/main

## Text to Image
images = pipe(
    prompt="A curly-haired man in a red shirt is drinking tea.", 
    height=1024, 
    width=1024, 
    guidance_scale=2.5,
    seed=0,
)
images[0].save("example_t2i.png")  # save output PIL Image

## Multi-modal to Image
# In the prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
# You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
images = pipe(
    prompt="A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
    input_images=["./imgs/test_cases/two_man.jpg"],
    height=1024, 
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    seed=0
)
images[0].save("example_ti2i.png")  # save output PIL image
```
- If out of memory, you can set `offload_model=True`. If the inference time is too long when inputting multiple images, you can reduce the `max_input_image_size`.  For the required resources and the method to run OmniGen efficiently, please refer to [docs/inference.md#requiremented-resources](docs/inference.md#requiremented-resources).
- For more examples of image generation, you can refer to [inference.ipynb](inference.ipynb) and [inference_demo.ipynb](inference_demo.ipynb)
- For more details about the argument in inference, please refer to [docs/inference.md](docs/inference.md). 


### Using Diffusers

Coming soon.


### Gradio Demo

We construct an online demo in [Huggingface](https://huggingface.co/spaces/Shitao/OmniGen).

For the local gradio demo, you need to install `pip install gradio spaces`, and then you can run:
```python
pip install gradio spaces
python app.py
```

#### Use Google Colab
To use with Google Colab, please use the following command:

```
!git clone https://github.com/VectorSpaceLab/OmniGen.git
%cd OmniGen
!pip install -e .
!pip install gradio spaces
!python app.py --share
```

## 5. Finetune
We provide a training script `train.py` to fine-tune OmniGen. 
Here is a toy example about LoRA finetune:
```bash
accelerate launch --num_processes=1 train.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 2 \
    --condition_dropout_prob 0.01 \
    --lr 1e-3 \
    --use_lora \
    --lora_rank 8 \
    --json_file ./toy_data/toy_subject_data.jsonl \
    --image_path ./toy_data/images \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 10 \
    --epochs 200 \
    --log_every 1 \
    --results_dir ./results/toy_finetune_lora
```

Please refer to [docs/fine-tuning.md](docs/fine-tuning.md) for more details (e.g. full finetune).

### Contributors:
Thank all our contributors for their efforts and warmly welcome new members to join in!

<a href="https://github.com/VectorSpaceLab/OmniGen/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=VectorSpaceLab/OmniGen" />
</a>

## License
This repo is licensed under the [MIT License](LICENSE). 


## Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{xiao2024omnigen,
  title={Omnigen: Unified image generation},
  author={Xiao, Shitao and Wang, Yueze and Zhou, Junjie and Yuan, Huaying and Xing, Xingrun and Yan, Ruiran and Wang, Shuting and Huang, Tiejun and Liu, Zheng},
  journal={arXiv preprint arXiv:2409.11340},
  year={2024}
}
```





