# Parameter-Efficient Transfer Learning with Diff Pruning

While task-specific finetuning of pretrained networks has led to significant empirical advances in NLP, the large size of networks makes finetuning difficult to deploy in multi-task, memory-constrained settings. We propose diff pruning as a simple approach to enable parameter-efficient transfer learning within the pretrain-finetune framework. This approach views finetuning as learning a task-specific diff vector that is applied on top of the pretrained parameter vector, which remains fixed and is shared across different tasks. The diff vector is adaptively pruned during training with a differentiable approximation to the L0-norm penalty to encourage sparsity. Diff pruning becomes parameter-efficient as the number of tasks increases, as it requires storing only the nonzero positions and weights of the diff vector for each task, while the cost of storing the shared pretrained model remains constant. It further does not require access to all tasks during training, which makes it attractive in settings where tasks arrive in stream or the set of tasks is unknown. We find that models finetuned with diff pruning can match the performance of fully finetuned baselines on the GLUE benchmark while only modifying 0.5% of the pretrained model's parameters per task. <br> <br>

[[Arxiv 2020 Paper]](https://arxiv.org/abs/2012.07463)

## Environment
This codebase was tested with the following environment configurations. <br>
* Ubuntu 16.0
* CUDA 10.0
* Python 3.6
* Pytorch 3.6
* Nvidia V100 (32 GB) GPU

## Installation & Dataset Processing
First, install relevant python packages. We update sentencepiece to a specified version for compatibility issues.
```
cd examples
pip install -r requirements.txt
pip install -v sentencepiece==0.1.91
cd ..
```
Next, install HuggingFace Transformers from (our modified) source code. 
```
cd transformers
pip install --editable .
cd ..
```
Finally, we download GLUE and SQuAD v1.1 datasets.
```
python download_glue.py
mkdir squad
cd squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
cd ..
```
## Running Diff Pruning Experiments
Here, we provide an example for running on CoLA dataset of GLUE benchmark. <br.
See more examples of full scripts (e.g. SST-2) in [Experiments Folder](https://github.com/dguo98/diff-pruning/blob/main/experiments) <br>

First, specify environmental variables and hyperparameters. Feel free to adapt based on your needs.
```
EXP_NAME=cola
SEED=0
PER_GPU_TRAIN_BATCH_SIZE=8
GRADIENT_ACC=1
LR=0.00001000
SPARSITY_PEN=0.00000012500
CONCRETE_LOWER=-1.500
CONCRETE_UPPER=1.500
ALPHA_INIT=5
FIX_LAYER=-1
BASE_DIR=/home/ubuntu/diffpruning
LOCAL_DATA_DIR=${BASE_DIR}/glue_data
LOCAL_CKPT_DIR=${BASE_DIR}/logs/${EXP_NAME}
GPU=0
TASK=cola
DATA=CoLA

mkdir -p ${LOCAL_CKPT_DIR}

cd ${BASE_DIR}
```
Next, run structured pruning.
```
CUDA_VISIBLE_DEVICES=${GPU} python ${BASE_DIR}/examples/run_glue_diffpruning.py --model_type bert --model_name_or_path bert-large-cased-whole-word-masking --task_name ${TASK} --output_dir ${LOCAL_CKPT_DIR} --do_train --do_eval --data_dir ${LOCAL_DATA_DIR}/${DATA} --sparsity_pen ${SPARSITY_PEN} --concrete_lower ${CONCRETE_LOWER} --concrete_upper ${CONCRETE_UPPER} --num_train_epochs 3 --save_steps 5000 --seed ${SEED} --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} --learning_rate ${LR}  --gradient_accumulation_steps ${GRADIENT_ACC} --fix_layer ${FIX_LAYER} --max_seq_length 128 --per_gpu_eval_batch_size 8 --overwrite_output_dir --logging_steps 5000 1>${LOCAL_CKPT_DIR}/${EXP_NAME}.out 2>${LOCAL_CKPT_DIR}/${EXP_NAME}.err
```
Finally, run magnitude pruning and fixmask finetuning.
```
EXP2=${EXP_NAME}_2nd_mag
mkdir -p ${LOCAL_CKPT_DIR}/${EXP2}
EXP3=${EXP_NAME}_3rd_fixmask
EVAL_CHECKPOINT=${BASE_DIR}/logs/${EXP_NAME}/checkpoint-last-info.pt
mkdir -p ${LOCAL_CKPT_DIR}/${EXP3}

CUDA_VISIBLE_DEVICES=${GPU} python ${BASE_DIR}/examples/run_glue_mag.py --model_type bert --model_name_or_path bert-large-cased-whole-word-masking --task_name ${TASK} --output_dir ${LOCAL_CKPT_DIR}/${EXP2} --do_train --do_eval --data_dir ${LOCAL_DATA_DIR}/${DATA} --sparsity_pen 0.000000125 --concrete_lower -1.5 --concrete_upper 1.5 --num_train_epochs 3 --save_steps 5000 --seed ${SEED} --eval_checkpoint ${EVAL_CHECKPOINT} --save_checkpoint ${LOCAL_CKPT_DIR}/mag0.5p.pt --evaluate_during_training --logging_steps 5000 --target_sparsity 0.005 --overwrite_output_dir 1>${LOCAL_CKPT_DIR}/${EXP2}_mag0.5p.out 2>${LOCAL_CKPT_DIR}/${EXP2}_mag0.5p.err

# learning rate default = 5e-5, device_id = 0 (relative)
CUDA_VISIBLE_DEVICES=${GPU} python ${BASE_DIR}/examples/run_glue_fixmask_finetune.py --model_type bert --model_name_or_path bert-large-cased-whole-word-masking --task_name ${TASK} --output_dir ${LOCAL_CKPT_DIR}/${EXP3} --do_train --do_eval --data_dir ${LOCAL_DATA_DIR}/${DATA} --sparsity_pen ${SPARSITY_PEN} --concrete_lower -1.5 --concrete_upper 1.5 --num_train_epochs 3 --save_steps 5000 --seed ${SEED} --mask_checkpoint ${LOCAL_CKPT_DIR}/mag0.5p.pt --evaluate_during_training --logging_steps 5000 --overwrite_output_dir --finetune 1 1>${LOCAL_CKPT_DIR}/${EXP3}_mag0.5.out 2>${LOCAL_CKPT_DIR}/${EXP3}_mag0.5.err
```
# Notes
This is still a WIP repository. We plan to release a faster, and cleaner version in the near future. 

# Citing Diff Pruning
```
@misc{guo2020parameterefficient,
      title={Parameter-Efficient Transfer Learning with Diff Pruning}, 
      author={Demi Guo and Alexander M. Rush and Yoon Kim},
      year={2020},
      eprint={2012.07463},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
