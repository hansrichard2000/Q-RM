cd ..
export TASK="gsm8k"
export DATA_FILE="path/to/pairwise/file/results.jsonl"  # data format: [{"instruction": "xxx", "chosen": "xxx", "rejected": "xxx"},]
export STRATEGY="${TASK}-beta-0.1-lr-1e-6"

export MAX_SEQ_LEN=1024
export MAX_BATCH_SIZE=2
export FORWARD_BATCH_SIZE=16

export POLICY_MODEL_TYPE="llama3"
export POLICY_MODEL_NAME="llama-3-70b-instruct"
export POLICY_CKPT_DIR="path/to/model/${POLICY_MODEL_TYPE}/${POLICY_MODEL_NAME}"  # need to download checkpoint from https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/tree/main/original
export POLICY_CONFIG_DIR="path/to/model/${POLICY_MODEL_TYPE}/${POLICY_MODEL_NAME}"
export POLICY_SAVE_DIR="results/${POLICY_MODEL_NAME}/policy/dpo/${STRATEGY}/lora-128/"

torchrun --nproc_per_node 8 policy_train_dpo.py \
  --train_file ${DATA_FILE} \
  --save_dir ${POLICY_SAVE_DIR} \
  --policy_ckpt_dir ${POLICY_CKPT_DIR} \
  --policy_model_type ${POLICY_MODEL_TYPE} \
  --policy_tokenizer_file ${POLICY_CONFIG_DIR} \
  --policy_config_file ${POLICY_CONFIG_DIR} \
  --reference_ckpt_dir ${POLICY_CKPT_DIR} \
  --max_seq_len ${MAX_SEQ_LEN} \
  --max_batch_size ${MAX_BATCH_SIZE} \
  --forward_batch_size ${FORWARD_BATCH_SIZE} \
  --lr 1e-6 \
  --dtype bfloat16 \
  --lora_rank 128 \
  --epochs 1 \
  --beta 0.1 \
  --use_chat_template