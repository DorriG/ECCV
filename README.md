# Multimodal Ambivalence and Hesitancy Recognition with Bidirectional Audio--Text Cross-Attention

This directory implements a local training version of an audio+text bimodal classification model:

- text: uses Hugging Face BERT, with `bert-base-multilingual-cased` as the default
- audio: uses wav2vec2, with `facebook/wav2vec2-base` as the default
- fusion: first projects the two modalities into the same dimensional space, then uses bidirectional cross-attention to align text tokens with audio-frame features
- target: binary classification, where `0 = No A-H` and `1 = A-H`
- metrics: outputs `CL_ACC`, `CONFUSION_MATRIX`, `F1_POS`, `F1_NEG`, `W_F1`, `MACRO_F1`, and `Average_precision_POS`

## Directory Structure

```text
fusion/
  README.md
  requirements.txt
  pyproject.toml
  scripts/train_default.sh
  fusion/
    dataset.py      # Reads split/*.txt and maps Videos/*.mp4 to audios/*.wav
    model.py        # BERT + wav2vec2 + bidirectional cross-attention
    engine.py       # Training, validation, checkpoint saving, and predictions
    metrics.py      # BAH/CASP-aligned metrics
    train.py        # Training entry point
    evaluate.py     # Checkpoint evaluation entry point
```

## Installation

It is recommended to install the environment in this directory:

```bash
cd /home/.../fusion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you already have a working environment, you can also run only:

```bash
cd /home/.../fusion
pip install -r requirements.txt
```

## Data Format

By default, the code reads:

```text
/home/.../data/split/train.txt
/home/.../data/split/val.txt
/home/.../data/split/test.txt
/home/.../data/audios
```

Each line should follow this format:

```text
Videos/.../xxx_Video.mp4,label,transcript
```

The code automatically maps `Videos/.../xxx_Video.mp4` to:

```text
data/audios/.../xxx_Video.wav
```

## Quick Training

Run the following command from the `fusion` directory:

```bash
python -m fusion.train \
  --data_root ../data \
  --output_dir outputs/bert_wav2vec_cross_attn \
  --epochs 10 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --auto_resume \
  --max_audio_seconds 30
```

You can also run the script directly:

```bash
bash scripts/train_default.sh
```

By default, the BERT and wav2vec2 encoders are frozen, and only the projection layers, cross-attention module, and classification head are trained. This usually saves GPU memory and is suitable for first verifying the pipeline. To fine-tune the last few layers, add:

```bash
python -m fusion.train \
  --data_root ../data \
  --output_dir outputs/finetune_last2 \
  --unfreeze_text_layers 2 \
  --unfreeze_audio_layers 2 \
  --encoder_lr 1e-5 \
  --lr 3e-4
```

If GPU memory is limited, you can reduce the following settings:

```bash
--batch_size 1 --eval_batch_size 1 --max_audio_seconds 15 --gradient_accumulation_steps 4
```

## Output Files

After training finishes, `output_dir` contains:

```text
best_model.pt
last_checkpoint.pt
last_loaded_best_model.pt
run_config.json
training_history.csv
metrics_val_epoch_*.json
metrics_val_epoch_*.csv
metrics_test.json
metrics_test.csv
predictions_test.csv
tokenizer/
audio_feature_extractor/
```

`predictions_test.csv` contains the sample ID, video path, audio path, ground-truth label, predicted label, A-H probability, and logits.

## Resume Training from a Checkpoint

After each training epoch, `last_checkpoint.pt` is saved. It contains the model, optimizer, scheduler, AMP scaler, completed epoch, best metric, and training history. If the server disconnects, restart training with the same `output_dir` and add `--auto_resume` to continue from the most recent completed epoch:

```bash
python -m fusion.train \
  --data_root ../data \
  --output_dir outputs/bert_wav2vec_cross_attn \
  --epochs 10 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --auto_resume
```

You can also explicitly specify a checkpoint:

```bash
python -m fusion.train \
  --data_root ../data \
  --output_dir outputs/bert_wav2vec_cross_attn \
  --epochs 10 \
  --resume outputs/bert_wav2vec_cross_attn/last_checkpoint.pt
```

If you want to retrain from scratch, use a new `--output_dir` or omit `--auto_resume`.

## Evaluate a Checkpoint Separately

```bash
python -m fusion.evaluate \
  --checkpoint outputs/bert_wav2vec_cross_attn/best_model.pt \
  --data_root ../data \
  --output_dir outputs/bert_wav2vec_cross_attn/eval_test
```

## Common Arguments

- `--text_model`: BERT model name, default `bert-base-multilingual-cased`
- `--audio_model`: wav2vec2 model name, default `facebook/wav2vec2-base`
- `--max_text_length`: maximum number of BERT tokens, default 256
- `--max_audio_seconds`: maximum number of seconds from each audio sample passed into wav2vec, default 30
- `--train_audio_crop`: training audio cropping strategy, default `random`
- `--eval_audio_crop`: validation/test audio cropping strategy, default `center`
- `--class_weight`: default `balanced`, used as the cross-entropy weight when classes are imbalanced
- `--metric_for_best`: validation metric used to save the best checkpoint, default `MACRO_F1`
- `--mixed_precision`: enables AMP on CUDA
- `--auto_resume`: automatically resumes training if `output_dir/last_checkpoint.pt` exists
- `--resume`: explicitly specifies a checkpoint to resume training from
- `--limit_train/--limit_val/--limit_test`: small-sample smoke test settings

## Smoke Test

First, use a very small sample to check the data, model, and output directory:

```bash
python -m fusion.train \
  --data_root ../data \
  --output_dir outputs/smoke \
  --epochs 1 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --max_audio_seconds 5 \
  --limit_train 4 \
  --limit_val 2 \
  --limit_test 2
```

The first run downloads the BERT and wav2vec2 weights from Hugging Face. If the server cannot access the internet, cache the models in advance and use `--cache_dir` to point to the cache directory.
