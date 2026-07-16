#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m fusion.train \
  --data_root ../data \
  --output_dir outputs/bert_wav2vec_cross_attn \
  --text_model bert-base-multilingual-cased \
  --audio_model facebook/wav2vec2-base \
  --epochs 10 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --auto_resume \
  --max_text_length 256 \
  --max_audio_seconds 30 \
  --fusion_dim 256 \
  --cross_attention_heads 4 \
  --cross_attention_layers 2
