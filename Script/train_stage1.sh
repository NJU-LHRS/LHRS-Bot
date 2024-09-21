OUTPUT_PATH=""  # Path to save the output
DATA_PATH=""    # Path to the Stage 1dataset
SCRIPT_PATH=../main_pretrain_stage1.py
CONFIG_PATH=../Config/multi_modal_stage1.yaml

unset WANDB_RUN_ID
unset WANDB_RUN_NAME

deepspeed \
    --num_node=1 \
    --num_gpus=8 \
    $SCRIPT_PATH \
    -c \
    $CONFIG_PATH \
    --batch-size 16 \
    --workers 4 \
    --data-path $DATA_PATH \
    --output $OUTPUT_PATH \
    --accelerator "gpu" \
    --enable-amp True \
    --use-checkpoint \