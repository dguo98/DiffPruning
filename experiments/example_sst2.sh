EXP_NAME=sst
SEED=0
PER_GPU_TRAIN_BATCH_SIZE=8
GRADIENT_ACC=1
LR=0.00001000
SPARSITY_PEN=0.00000012500
CONCRETE_LOWER=-1.500
CONCRETE_UPPER=1.500
ALPHA_INIT=5
FIX_LAYER=-1
BASE_DIR=/home/ubuntu/finetune-compression
LOCAL_DATA_DIR=${BASE_DIR}/glue_data
LOCAL_CKPT_DIR=${BASE_DIR}/logs/${EXP_NAME}
GPU=1
TASK=sst-2
DATA=SST-2

mkdir -p ${LOCAL_CKPT_DIR}

cd ${BASE_DIR}

# install transformer
CUDA_VISIBLE_DEVICES=${GPU} python ${BASE_DIR}/examples/run_glue_diffpruning.py --model_type bert --model_name_or_path bert-large-cased-whole-word-masking --task_name ${TASK} --output_dir ${LOCAL_CKPT_DIR} --do_train --do_eval --data_dir ${LOCAL_DATA_DIR}/${DATA} --sparsity_pen ${SPARSITY_PEN} --concrete_lower ${CONCRETE_LOWER} --concrete_upper ${CONCRETE_UPPER} --num_train_epochs 3 --save_steps 5000 --seed ${SEED} --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} --learning_rate ${LR}  --gradient_accumulation_steps ${GRADIENT_ACC} --fix_layer ${FIX_LAYER} --max_seq_length 128 --per_gpu_eval_batch_size 8 --overwrite_output_dir --logging_steps 5000 1>${LOCAL_CKPT_DIR}/${EXP_NAME}.out 2>${LOCAL_CKPT_DIR}/${EXP_NAME}.err

EXP2=${EXP_NAME}_2nd_mag
mkdir -p ${LOCAL_CKPT_DIR}/${EXP2}
EXP3=${EXP_NAME}_3rd_fixmask
EVAL_CHECKPOINT=${BASE_DIR}/logs/${EXP_NAME}/checkpoint-last-info.pt

mkdir -p ${LOCAL_CKPT_DIR}/${EXP3}
CUDA_VISIBLE_DEVICES=${GPU} python ${BASE_DIR}/examples/run_glue_mag.py --model_type bert --model_name_or_path bert-large-cased-whole-word-masking --task_name ${TASK} --output_dir ${LOCAL_CKPT_DIR}/${EXP2} --do_train --do_eval --data_dir ${LOCAL_DATA_DIR}/${DATA} --sparsity_pen 0.000000125 --concrete_lower -1.5 --concrete_upper 1.5 --num_train_epochs 3 --save_steps 5000 --seed ${SEED} --eval_checkpoint ${EVAL_CHECKPOINT} --save_checkpoint ${LOCAL_CKPT_DIR}/mag0.5p.pt --evaluate_during_training --logging_steps 5000 --target_sparsity 0.005 --overwrite_output_dir 1>${LOCAL_CKPT_DIR}/${EXP2}_mag0.5p.out 2>${LOCAL_CKPT_DIR}/${EXP2}_mag0.5p.err
# learning rate default = 5e-5, device_id = 0 (relative)
CUDA_VISIBLE_DEVICES=${GPU} python ${BASE_DIR}/examples/run_glue_fixmask_finetune.py --model_type bert --model_name_or_path bert-large-cased-whole-word-masking --task_name ${TASK} --output_dir ${LOCAL_CKPT_DIR}/${EXP3} --do_train --do_eval --data_dir ${LOCAL_DATA_DIR}/${DATA} --sparsity_pen ${SPARSITY_PEN} --concrete_lower -1.5 --concrete_upper 1.5 --num_train_epochs 3 --save_steps 5000 --seed ${SEED} --mask_checkpoint ${LOCAL_CKPT_DIR}/mag0.5p.pt --evaluate_during_training --logging_steps 5000 --overwrite_output_dir --finetune 1 1>${LOCAL_CKPT_DIR}/${EXP3}_mag0.5.out 2>${LOCAL_CKPT_DIR}/${EXP3}_mag0.5.err
rm ${LOCAL_CKPT_DIR}/mag0.5p.pt
rm -r ${LOCAL_CKPT_DIR}/${EXP3}

