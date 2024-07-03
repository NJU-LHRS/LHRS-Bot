MODEL_PATH="" # Path to the Stage2 model
OUTPUT_PATH=""  # Path to save the output
DATA_PATH=""  # Path to the Stage 3 dataset
CONFIG_PATH=../Config/multi_modal_stage3.yaml
SCRIPT_PATH=../main_pretrain_stage3.py

deepspeed \
    --num_node=1 \
    --num_gpus=8 \
    $SCRIPT_PATH \
    -c \
    $CONFIG_PATH \
    --batch-size 4 \
    --workers 2 \
    --data-path $DATA_PATH \
    --output $OUTPUT_PATH \
    --accelerator "gpu" \
    --enable-amp True \
    --use-checkpoint \