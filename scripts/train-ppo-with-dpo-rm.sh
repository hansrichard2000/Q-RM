cd ..
export TASK="gsm8k"
export DATA_FILE="data/${TASK}/train/ppo/results.jsonl"  # data format [{"instruction": "xxx"},]
export LABEL_FILE="data/${TASK}/test/test.jsonl"  # data format [{"instruction": "xxx", "label": "xxx"},]
export MODEL_TYPE="llama3"
export STRATEGY="${TASK}-dpo-rm-samples-1-lr-5e-6-chunk-size-3072"

export ACTOR_MODEL_NAME="llama-3.2-3b-instruct"
export ACTOR_LORA_RANK=-1
export ACTOR_LORA_DTYPE="bfloat16"
export ACTOR_MAX_BATCH_SIZE=10
export ACTOR_SAVE_DIR="results/${ACTOR_MODEL_NAME}/policy/ppo/${STRATEGY}/full/"
export POLICY_CKPT_DIR="path/to/model/${MODEL_TYPE}/${ACTOR_MODEL_NAME}/"  # need to download from https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main/original
export POLICY_CONFIG_DIR="path/to/model/${MODEL_TYPE}/${ACTOR_MODEL_NAME}/"
export LOG_DIR="log/${POLICY_MODEL_NAME}/policy/ppo/${STRATEGY}/full/"

export VERIFIER_MODEL_NAME="llama-3-70b-instruct"
export VERIFIER_CKPT_DIR="results/${VERIFIER_MODEL_NAME}/policy/dpo/${TASK}-beta-0.1-lr-1e-6/full/epoch-1/"
export VERIFIER_CONFIG_DIR="path/to/model/${MODEL_TYPE}/${VERIFIER_MODEL_NAME}"
export VERIFIER_REFERENCE_CKPT_DIR="path/to/model/${MODEL_TYPE}/${VERIFIER_MODEL_NAME}"

export CRITIC_MODEL_NAME="llama-3-70b-instruct"
export CRITIC_LORA_RANK=-1
export CRITIC_LORA_DTYPE="bfloat16"
export CRITIC_MAX_BATCH_SIZE=1
export CRITIC_SAVE_DIR="results/${CRITIC_MODEL_NAME}/verifier/ppo/${STRATEGY}/lora-${CRITIC_LORA_RANK}/"
export CRITIC_CKPT_DIR=${VERIFIER_CKPT_DIR}
export CRITIC_CONFIG_DIR=${VERIFIER_CONFIG_DIR}

export REFERENCE_CKPT_DIR=${ACTOR_CKPT_DIR}

torchrun --nproc_per_node 8 policy_train_ppo_dpo_rm.py \
  --task ${TASK} \
  --label_file ${LABEL_FILE} \
  --train_file ${DATA_FILE} \
  --log_dir ${LOG_DIR} \
  --actor_ckpt_dir ${ACTOR_CKPT_DIR} \
  --actor_model_type ${MODEL_TYPE} \
  --actor_save_dir ${ACTOR_SAVE_DIR} \
  --actor_config_file ${ACTOR_CONFIG_DIR} \
  --actor_tokenizer_file ${ACTOR_CONFIG_DIR} \
  --critic_ckpt_dir ${CRITIC_CKPT_DIR} \
  --critic_model_type ${MODEL_TYPE} \
  --critic_save_dir ${CRITIC_SAVE_DIR} \
  --critic_config_file ${CRITIC_CONFIG_DIR} \
  --critic_tokenizer_file ${CRITIC_CONFIG_DIR} \
  --verifier_ckpt_dir ${VERIFIER_CKPT_DIR} \
  --verifier_model_type ${MODEL_TYPE} \
  --verifier_reference_ckpt_dir ${VERIFIER_REFERENCE_CKPT_DIR} \
  --verifier_config_file ${VERIFIER_CONFIG_DIR} \
  --verifier_tokenizer_file ${VERIFIER_CONFIG_DIR} \
  --reference_ckpt_dir ${REFERENCE_CKPT_DIR} \
  --actor_lora_rank ${ACTOR_LORA_RANK} \
  --actor_lora_dtype ${ACTOR_LORA_DTYPE} \
  --critic_lora_rank ${CRITIC_LORA_RANK} \
  --critic_lora_dtype ${CRITIC_LORA_DTYPE} \
  --actor_max_batch_size ${ACTOR_MAX_BATCH_SIZE} \
  --critic_max_batch_size ${CRITIC_MAX_BATCH_SIZE} \
  --max_generate_batch_size 512 \
  --max_forward_batch_size 16 \
  --max_seq_len 1024 \
  --chunk_size 3072 \
  --inner_epochs 1 \
  --lr 5e-6 \
  --kl_coef 0.01 \
  --beta 0.1 \
  --clip_range 0.2 \
  --use_chat_template