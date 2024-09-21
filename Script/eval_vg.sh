SCRIPT_PATH=/home/aiscuser/LHRS-Bot/main_vg.py
OUTPUT_PATH=/home/aiscuser/Output/VGEval/RSVG_DIOR
CONFI_PATH=/home/aiscuser/LHRS-Bot/Config/multi_modal_eval.yaml
MODEL_PATH=/mnt/default/projects/v-dmuhtar/data/LHRS/Stage3/checkpoints/FINAL.pt
DATA_PATH=/home/aiscuser/Datastore/InstructDataset/RSVG_DIOR_Image
DATA_TARGET=/home/aiscuser/Datastore/Eval/VGEvalDataset/RSVG_DIOR_test.json

source /opt/conda/bin/activate lhrs

deepspeed \
    --master_port=29500 \
    --include="node-0:$1" \
    $SCRIPT_PATH \
    -c $CONFI_PATH \
    --batch-size 1 \
    --workers 0 \
    --output $OUTPUT_PATH \
    --model-path $MODEL_PATH \
    --data-path $DATA_PATH \
    --data-target $DATA_TARGET \
    --enable-amp True \
    --accelerator "gpu" \