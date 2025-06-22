cd ..
export TASK="gsm8k"
export DATA_FILE="/data/${TASK}/test/pairwise/results.jsonl"  # data format: [{"instruction": "xxx", "chosen": "xxx", "rejected": "xxx"},]
export MODEL_TYPE="llama3"
export MODEL_NAME="llama-3-70b-instruct"
export STRATEGY="mean-score-beta-0.2-gamma-2.0-lr-1e-5"
export CONFIG_DIR="../../models/${MODEL_TYPE}/${MODEL_NAME}/"

for EPOCH in 1
do
  export MODEL_DIR="results/${MODEL_NAME}/verifier/${TASK}-pairwise-${STRATEGY}/lora-128/epoch-${EPOCH}"
  export LOG_DIR="log/${MODEL_NAME}/verifier/${TASK}-pairwise-${STRATEGY}/lora-128/epoch-${EPOCH}"
  torchrun --nproc_per_node 8 verifier_evaluate_pairwise.py \
    --strategy ${STRATEGY} \
    --ckpt_dir ${MODEL_DIR} \
    --log_dir ${LOG_DIR} \
    --label_file ${DATA_FILE} \
    --config_file ${CONFIG_DIR} \
    --tokenizer_file ${CONFIG_DIR} \
    --model_type ${MODEL_TYPE} \
    --max_seq_len 1024 \
    --max_batch_size 18 \
    --dtype bfloat16 \
    --use_chat_template

  python draw_token_scores.py \
    --data_file ${LOG_DIR}/token_results.json \
    --model_type ${MODEL_TYPE} \
    --tokenizer_file ${CONFIG_DIR} \
    --idx 1
done