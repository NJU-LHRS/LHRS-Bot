SCRIPT_PATH=../main_vqa.py
OUTPUT_PATH="" # Output path
CONFI_PATH=../Config/multi_modal_eval.yaml
MODEL_PATH=""  # Model path
DATA_PATH=""   # Data path
DATA_TARGET="" # Data target path

deepspeed \
    --master_port=29500 \
    $SCRIPT_PATH \
    -c $CONFI_PATH \
    --batch-size 1 \
    --workers 0 \
    --output $OUTPUT_PATH \
    --model-path $MODEL_PATH \
    --data-path $DATA_PATH \
    --data-target $DATA_TARGET \
    --data-type "HR" \
    --enable-amp True \
    --accelerator "gpu" \