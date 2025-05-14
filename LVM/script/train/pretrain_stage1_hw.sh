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
echo '--------------------------'
pkill -f python

# export CUDA_LAUNCH_BLOCKING=1

echo "start training"

PROJ_PATH='.'
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_PORT=${MASTER_PORT:-23456}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    ${PROJ_PATH}/LVM/train/train_x1_stage1_noiseinput.py \
    --model_name_or_path "/path/to/Video-GPT/huggingface" \
    --results_dir "/path/to/experiment/stage1" \
    --video_dir_path "/path/to/pretrain/video/data" \
    --sequence_parallel_size=1 \
    --gradient_accumulation_steps=1 \
    --batch_size_per_device=1 \
    --mixed_precision "bf16" \
    --report_to "tensorboard" \
    --lr_warmup_steps 320000 \
    --lr_scheduler "constant_with_warmup" \
    --max_image_size 320 \
    --keep_raw_resolution \
    --lr=1e-4 \
    --max_grad_norm=1.0 \
    --ckpt_every=500 \
    --log_every=10 \
    --num_workers=4 \
    --prefetch_factor=2 \
    --epochs=1000000000000 \
    --max_input_length_limit=128000 \
    --frame_num=16 \
    --frame_interval=4 \
    --adam_weight_decay=0.1 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --adam_epsilon=1e-5 \
    --gradient_checkpointing=False \
    --timeout 30 \
    --deepspeed_plugin "LVM/acceleration/config/stage2_bf16_dp.json" \


echo '--------------------------'
echo training task done
echo '--------------------------'