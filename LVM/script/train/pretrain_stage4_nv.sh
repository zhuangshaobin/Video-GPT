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
    ${PROJ_PATH}/LVM/train/train_x1_stage34_noiseinput_frameblock_flexibleintervalbound.py \
    --model_name_or_path "/path/to/Video-GPT/huggingface" \
    --results_dir "/path/to/experiment/stage1" \
    --video_dir_path "/path/to/pretrain/video/data" \
    --model_ckpt "/path/to/pretrain/stage3/model/ckpt" \
    --sequence_parallel_size=1 \
    --gradient_accumulation_steps=1 \
    --batch_size_per_device=1 \
    --mixed_precision "bf16" \
    --report_to "tensorboard" \
    --lr_warmup_steps 176000 \
    --lr_scheduler "constant_with_warmup" \
    --max_image_size 320 \
    --keep_raw_resolution \
    --lr=1e-4 \
    --max_grad_norm=1.0 \
    --ckpt_every=500 \
    --log_every=10 \
    --num_workers=2 \
    --prefetch_factor=2 \
    --epochs=1000000000000 \
    --max_input_length_limit=1280000 \
    --frame_num=80 \
    --frame_interval=4 \
    --adam_weight_decay=0.1 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --adam_epsilon=1e-5 \
    --gradient_checkpointing=False \
    --timeout 100000 \
    --interval_bound 12 \
    --deepspeed_plugin "LVM/acceleration/config/stage2_bf16_dp.json" \


echo '--------------------------'
echo training task done
echo '--------------------------'