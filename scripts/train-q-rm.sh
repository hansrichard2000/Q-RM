export TASK="gsm8k"
export DATA_FILE="data/${TASK}/train/pairwise/results.jsonl"  # data format: [{"instruction": "xxx", "chosen": "xxx", "rejected": "xxx"},]
export MODEL_TYPE="llama3.1"
export MODEL_NAME="llama-3.1-8b-instruct"
export LORA_RANK=128
export LR=1e-5
export STRATEGY="mean-score-beta-0.2-gamma-2.0-lr-${LR}"
export MODEL_DIR="/root/ReinforcementLearning/Q-RM/src/weights/Llama-3.1-8B-Instruct/original/"  # need to download checkpoint from https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/tree/main/original
export SAVE_DIR="results/${MODEL_NAME}/verifier/${TASK}-pairwise-${STRATEGY}/lora-${LORA_RANK}/"
export LOG_DIR="log/${MODEL_NAME}/verifier/${TASK}-pairwise-${STRATEGY}/lora-${LORA_RANK}/"
torchrun --nproc_per_node 1 verifier_train_pairwise.py \
    --strategy ${STRATEGY} \
    --ckpt_dir ${MODEL_DIR} \
    --save_dir ${SAVE_DIR} \
    --train_file ${DATA_FILE} \
    --model_type ${MODEL_TYPE} \
    --max_seq_len 1024 \
    --max_batch_size 2 \
    --epochs 1 \
    --lr ${LR} \
    --dtype bfloat16 \
    --beta 0.2 \
    --gamma 2.0 \
    --log_dir ${LOG_DIR} \
    --lora_rank ${LORA_RANK} \
    --lora_dtype bfloat16 \
    --use_chat_template