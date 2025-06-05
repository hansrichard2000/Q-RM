cd ..
export TASK="gsm8k"
export DATA_FILE="data/${TASK}/train/ppo/results.jsonl"  # data format [{"instruction": "xxx"},]
export MODEL_TYPE="llama3"
export STRATEGY="${TASK}-dpo-rm-samples-1-lr-1e-6-chunk-size-3072"
export LABEL_FILE="data/${TASK}/test/test.jsonl"  # data format [{"instruction": "xxx", "label": "xxx"},]

export POLICY_MODEL_NAME="llama-3.2-3b-instruct"
export POLICY_LORA_RANK=-1
export POLICY_LORA_DTYPE="bfloat16"
export POLICY_MAX_BATCH_SIZE=8
export POLICY_SAVE_DIR="results/${POLICY_MODEL_NAME}/policy/policy-gradient/${STRATEGY}/full/"
export POLICY_CKPT_DIR="path/to/model/${MODEL_TYPE}/${POLICY_MODEL_NAME}/"  # need to download from https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main/original
export POLICY_CONFIG_DIR="path/to/model/${MODEL_TYPE}/${POLICY_MODEL_NAME}/"
export LOG_DIR="log/${POLICY_MODEL_NAME}/policy/policy-gradient/${STRATEGY}/full/"

export VERIFIER_MODEL_NAME="llama-3-70b-instruct"
export VERIFIER_CKPT_DIR="results/${VERIFIER_MODEL_NAME}/policy/dpo/${TASK}-beta-0.1-lr-1e-6/lora-128/epoch-1/"
export VERIFIER_CONFIG_DIR="path/to/model/${MODEL_TYPE}/${VERIFIER_MODEL_NAME}"
export VERIFIER_REFERENCE_CKPT_DIR="path/to/model/${MODEL_TYPE}/${VERIFIER_MODEL_NAME}"

torchrun --nproc_per_node 8 policy_train_policy_gradient_dpo_rm.py \
  --task ${TASK} \
  --label_file ${LABEL_FILE} \
  --log_dir ${LOG_DIR} \
  --train_file ${DATA_FILE} \
  --save_dir ${POLICY_SAVE_DIR} \
  --policy_ckpt_dir ${POLICY_CKPT_DIR} \
  --policy_model_type ${MODEL_TYPE} \
  --policy_config_file ${POLICY_CONFIG_DIR} \
  --policy_tokenizer_file ${POLICY_CONFIG_DIR} \
  --verifier_ckpt_dir ${VERIFIER_CKPT_DIR} \
  --verifier_model_type ${MODEL_TYPE} \
  --verifier_config_file ${VERIFIER_CONFIG_DIR} \
  --verifier_tokenizer_file ${VERIFIER_CONFIG_DIR} \
  --verifier_reference_ckpt_dir ${VERIFIER_REFERENCE_CKPT_DIR} \
  --lora_rank ${POLICY_LORA_RANK} \
  --lora_dtype ${POLICY_LORA_DTYPE} \
  --max_batch_size ${POLICY_MAX_BATCH_SIZE} \
  --max_generate_batch_size 512 \
  --max_forward_batch_size 16 \
  --max_seq_len 1024 \
  --epochs 1 \
  --chunk_size 3072 \
  --begin_epoch 0 \
  --beta 0.1 \
  --lr 1e-6 \
  --num_samples_per_prompt 1 \
  --use_chat_template