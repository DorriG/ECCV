# Multimodal Ambivalence and Hesitancy Recognition with Bidirectional Audio--Text Cross-Attention

这个目录实现了一个本地训练版的 audio+text 双模态分类模型：

- text：使用 Hugging Face BERT，默认 `bert-base-multilingual-cased`
- audio：使用 wav2vec2，默认 `facebook/wav2vec2-base`
- fusion：先把两个模态投影到同一维度，再用双向 cross-attention 对齐文本 token 和音频帧特征
- target：二分类，`0 = No A-H`，`1 = A-H`
- metrics：输出 `CL_ACC`、`CONFUSION_MATRIX`、`F1_POS`、`F1_NEG`、`W_F1`、`MACRO_F1`、`Average_precision_POS`，并额外给出 CASP 风格的 `CASP_ACC2` 和 `CASP_F1_WEIGHTED`

CASP 原仓库 README 主要规定了训练流程和 PyTorch 环境；其代码里的二分类评价核心是 `Accuracy` 和 weighted `F1 score`。BAH 数据已有 zero-shot 代码使用的指标命名更细，所以这里同时保留 BAH 指标名和 CASP 风格字段。

## 目录结构

```text
fusion/
  README.md
  requirements.txt
  pyproject.toml
  scripts/train_default.sh
  fusion/
    dataset.py      # 读取 split/*.txt，映射 Videos/*.mp4 到 audios/*.wav
    model.py        # BERT + wav2vec2 + bidirectional cross-attention
    engine.py       # 训练、验证、保存 checkpoint 和 predictions
    metrics.py      # BAH/CASP 对齐指标
    train.py        # 训练入口
    evaluate.py     # checkpoint 评估入口
```

## 安装

建议在本目录安装：

```bash
cd /home/dorri/dorri_workshop/12_eccv/fusion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

如果你已有可用环境，也可以只执行：

```bash
cd /home/dorri/dorri_workshop/12_eccv/fusion
pip install -r requirements.txt
```

## 数据格式

默认读取：

```text
/home/dorri/dorri_workshop/12_eccv/data/split/train.txt
/home/dorri/dorri_workshop/12_eccv/data/split/val.txt
/home/dorri/dorri_workshop/12_eccv/data/split/test.txt
/home/dorri/dorri_workshop/12_eccv/data/audios
```

每行格式：

```text
Videos/.../xxx_Video.mp4,label,transcript
```

代码会自动把 `Videos/.../xxx_Video.mp4` 映射为：

```text
data/audios/.../xxx_Video.wav
```

## 快速训练

从 `fusion` 目录运行：

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

也可以直接运行脚本：

```bash
bash scripts/train_default.sh
```

默认冻结 BERT 和 wav2vec2 编码器，只训练投影层、cross-attention 和分类头。这通常更省显存，也适合先确认 pipeline。想微调最后几层可以加：

```bash
python -m fusion.train \
  --data_root ../data \
  --output_dir outputs/finetune_last2 \
  --unfreeze_text_layers 2 \
  --unfreeze_audio_layers 2 \
  --encoder_lr 1e-5 \
  --lr 3e-4
```

如果显存紧张，可以降低：

```bash
--batch_size 1 --eval_batch_size 1 --max_audio_seconds 15 --gradient_accumulation_steps 4
```

## 输出文件

训练完成后，`output_dir` 包含：

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

`predictions_test.csv` 会包含 sample id、视频路径、音频路径、真实标签、预测标签、A-H 概率和 logits。

## 断点续训

训练每个 epoch 结束后会保存 `last_checkpoint.pt`，里面包含模型、optimizer、scheduler、AMP scaler、已完成 epoch、best 指标和 history。服务器断开后，用同一个 `output_dir` 重新启动并加 `--auto_resume` 即可从最近一个完整 epoch 后继续：

```bash
python -m fusion.train \
  --data_root ../data \
  --output_dir outputs/bert_wav2vec_cross_attn \
  --epochs 10 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --auto_resume
```

也可以显式指定 checkpoint：

```bash
python -m fusion.train \
  --data_root ../data \
  --output_dir outputs/bert_wav2vec_cross_attn \
  --epochs 10 \
  --resume outputs/bert_wav2vec_cross_attn/last_checkpoint.pt
```

如果想从头重新训练，请换一个新的 `--output_dir`，或者不要加 `--auto_resume`。

## 单独评估 checkpoint

```bash
python -m fusion.evaluate \
  --checkpoint outputs/bert_wav2vec_cross_attn/best_model.pt \
  --data_root ../data \
  --output_dir outputs/bert_wav2vec_cross_attn/eval_test
```

## 常用参数

- `--text_model`：BERT 模型名，默认 `bert-base-multilingual-cased`
- `--audio_model`：wav2vec2 模型名，默认 `facebook/wav2vec2-base`
- `--max_text_length`：BERT 最大 token 数，默认 256
- `--max_audio_seconds`：每条音频最多送入 wav2vec 的秒数，默认 30
- `--train_audio_crop`：训练音频裁剪方式，默认 `random`
- `--eval_audio_crop`：验证/测试音频裁剪方式，默认 `center`
- `--class_weight`：默认 `balanced`，用于类别不平衡时的交叉熵权重
- `--metric_for_best`：保存 best checkpoint 的验证指标，默认 `MACRO_F1`
- `--mixed_precision`：CUDA 上启用 AMP
- `--auto_resume`：如果 `output_dir/last_checkpoint.pt` 存在，自动断点续训
- `--resume`：显式指定一个 checkpoint 继续训练
- `--limit_train/--limit_val/--limit_test`：小样本 smoke test

## Smoke Test

先用极小样本检查数据、模型和输出目录：

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

第一次运行会从 Hugging Face 下载 BERT 和 wav2vec2 权重；如果服务器不能联网，请提前把模型缓存好，并用 `--cache_dir` 指向缓存目录。
