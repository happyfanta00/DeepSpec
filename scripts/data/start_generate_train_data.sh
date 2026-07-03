python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3-4B \
    --server-address \
        127.0.0.1:30000 \
        127.0.0.1:30001 \
        127.0.0.1:30002 \
        127.0.0.1:30003 \
        127.0.0.1:30004 \
        127.0.0.1:30005 \
        127.0.0.1:30006 \
        127.0.0.1:30007 \
    --concurrency 32 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl