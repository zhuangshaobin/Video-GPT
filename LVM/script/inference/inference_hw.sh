source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export HCCL_RDMA_TIMEOUT=20
export ASCEND_PROCESS_LOG_PATH=/root/tmp_plog/

# avoid hccl timeout
export ASCEND_WORK_PATH=/tmp
export ASCEND_WORK_PATH=/tmp
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_ENTRY_LOG_ENABLE=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export HCCL_RDMA_TIMEOUT=20
export HCCL_CONNECT_TIMEOUT=3600
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo '--------------------------'
pkill -f python

# export CUDA_LAUNCH_BLOCKING=1

echo "start inference"

PROJ_PATH='.'
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_PORT=${MASTER_PORT:-23456}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    ${PROJ_PATH}/LVM/inference/LVM_video_frameblock_autoregressive_inference.py \
    --sequence_parallel_size=4 \
    --model_name "/path/to/Video-GPT/huggingface" \
    --model_ckpt "/path/to/Video-GPT/huggingface/transformer/ckpt" \
    --save_dir "result/test/inference" \
    --video_path_dir "vids" \
    --num_inference_steps=50 \
    --width 320 \
    --height 176 \
    --img_guidance_scale=1.5 \
    --gen_nums 24 24 24 24 \
    --max_input_image_size=320 \
    --clean_image_noise_level=0 \
    --clean_image_num=8 \
    --max_frame_window=80 \


echo '--------------------------'
echo inference task done
echo '--------------------------'