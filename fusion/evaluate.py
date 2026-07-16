from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer

from .dataset import BAHTextAudioDataset
from .dataset import TextAudioCollator
from .dataset import build_dataloader
from .engine import evaluate
from .engine import save_predictions
from .metrics import save_metrics
from .model import BertWav2VecCrossAttentionClassifier
from .utils import resolve_device
from .utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained fusion checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("../data"))
    parser.add_argument("--test_file", type=Path, default=None)
    parser.add_argument("--audio_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _checkpoint_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    return dict(raw_args)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = _checkpoint_args(checkpoint["args"])
    seed_everything(int(train_args.get("seed", 42)))

    data_root = args.data_root
    test_file = args.test_file or data_root / "split" / "test.txt"
    audio_root = args.audio_root or data_root / "audios"
    output_dir = args.output_dir or args.checkpoint.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    text_model = train_args["text_model"]
    audio_model = train_args["audio_model"]
    tokenizer_path = args.checkpoint.parent / "tokenizer"
    audio_processor_path = args.checkpoint.parent / "audio_feature_extractor"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path.exists() else text_model
    )
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
        audio_processor_path if audio_processor_path.exists() else audio_model
    )
    sampling_rate = int(getattr(audio_feature_extractor, "sampling_rate", 16000))

    dataset = BAHTextAudioDataset(
        manifest_path=test_file,
        audio_root=audio_root,
        target_sampling_rate=sampling_rate,
        max_audio_seconds=float(train_args.get("max_audio_seconds", 30.0)),
        crop_mode=str(train_args.get("eval_audio_crop", "center")),
        limit=args.limit,
    )
    collator = TextAudioCollator(
        tokenizer=tokenizer,
        audio_feature_extractor=audio_feature_extractor,
        sampling_rate=sampling_rate,
        max_text_length=int(train_args.get("max_text_length", 256)),
    )
    loader = build_dataloader(
        dataset=dataset,
        collator=collator,
        batch_size=args.batch_size or int(train_args.get("eval_batch_size", 4)),
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = resolve_device(args.no_cuda)
    mixed_precision = args.mixed_precision and device.type == "cuda"
    model = BertWav2VecCrossAttentionClassifier(
        text_model_name=text_model,
        audio_model_name=audio_model,
        fusion_dim=int(train_args.get("fusion_dim", 256)),
        cross_attention_heads=int(train_args.get("cross_attention_heads", 4)),
        cross_attention_layers=int(train_args.get("cross_attention_layers", 2)),
        dropout=float(train_args.get("dropout", 0.2)),
        unfreeze_text_layers=int(train_args.get("unfreeze_text_layers", 0)),
        unfreeze_audio_layers=int(train_args.get("unfreeze_audio_layers", 0)),
        cache_dir=train_args.get("cache_dir") or None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    _, metrics, rows = evaluate(
        model=model,
        loader=loader,
        device=device,
        class_weights=None,
        mixed_precision=mixed_precision,
        desc="test",
    )
    save_metrics(metrics, output_dir / "metrics_test.json", output_dir / "metrics_test.csv")
    save_predictions(rows, output_dir / "predictions_test.csv")

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
