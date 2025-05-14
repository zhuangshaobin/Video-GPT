#!/bin/bash
export LC_ALL=en_US.UTF-8 
export LANG=en_US.UTF-8 

this_scripts_path=$(cd `dirname $0`; pwd)
echo $this_scripts_path


cd ~

# this file's dir
cd $this_scripts_path

for ((i=0; i<3; i++)); do
    pip install -U torch==2.3.1
    pip install -U transformers==4.47.1
    pip install -U accelerate==0.31.0
    pip install -U diffusers==0.29.0
    pip install -U huggingface-hub==0.25.0
    pip install -U peft==0.14.0
    pip install -U deepspeed==0.15.2
    pip install -U decord
    pip install -U natsort
    yum install -y ffmpeg ffmpeg-devel
done