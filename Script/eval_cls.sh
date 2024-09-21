SCRIPT_PATH=../main_cls.py
OUTPUT_PATH="" # Output path
CONFI_PATH=../Config/multi_modal_eval.yaml
MODEL_PATH="" # Model path
DATA_PATH=$1

deepspeed \
    --master_port=29500 \
    $SCRIPT_PATH \
    -c $CONFI_PATH \
    --batch-size 16 \
    --workers 4 \
    --output $OUTPUT_PATH \
    --model-path $MODEL_PATH \
    --data-path $DATA_PATH \
    --enable-amp True \
    --accelerator "gpu" \