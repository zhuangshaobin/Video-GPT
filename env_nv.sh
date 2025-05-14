#!/bin/bash
export LC_ALL=en_US.UTF-8 
export LANG=en_US.UTF-8 

this_scripts_path=$(cd `dirname $0`; pwd)
echo $this_scripts_path


cd ~

# this file's dir
cd $this_scripts_path

for ((i=0; i<3; i++)); do
    pip install -i https://mirrors.tencent.com/pypi/simple/ -U torch==2.3.1
    pip install -i https://mirrors.tencent.com/pypi/simple/ -U transformers==4.47.1
    pip install -i https://mirrors.tencent.com/pypi/simple/ accelerate==0.31.0
    pip install -i https://mirrors.tencent.com/pypi/simple/ -U diffusers==0.29.0
    pip install -i https://mirrors.tencent.com/pypi/simple/ -U huggingface-hub==0.25.0
    pip install -i https://mirrors.tencent.com/pypi/simple/ -U peft==0.14.0
    pip install -i https://mirrors.tencent.com/pypi/simple/ -U deepspeed==0.15.2
    pip install -i https://mirrors.tencent.com/pypi/simple/ -U decord
    pip install -i https://mirrors.tencent.com/pypi/simple/ -U natsort
    yum install -y ffmpeg ffmpeg-devel
done