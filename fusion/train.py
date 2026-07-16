from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer

from .dataset import BAHTextAudioDataset
from .dataset import TextAudioCollator
from .dataset import build_dataloader
from .engine import build_optimizer_and_scheduler
from .engine import compute_class_weights
from .engine import evaluate
from .engine import save_checkpoint
from .engine import save_predictions
from .engine import train_one_epoch
from .metrics import MACRO_F1
from .metrics import save_metrics
from .model import BertWav2VecCrossAttentionClassifier
from .utils import count_parameters
from .utils import resolve_device
from .utils import save_json
from .utils import seed_everything
from .utils import write_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT + wav2vec cross-attention model for BAH A/H classification."
    )
    parser.add_argument("--data_root", type=Path, default=Path("../data"))
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--val_file", type=Path, default=None)
    parser.add_argument("--test_file", type=Path, default=None)
    parser.add_argument("--audio_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/cross_attn"))

    parser.add_argument("--text_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--audio_model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_text_length", type=int, default=256)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--train_audio_crop", choices=["random", "center", "first"], default="random")
    parser.add_argument("--eval_audio_crop", choices=["center", "first"], default="center")

    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--cross_attention_heads", type=int, default=4)
    parser.add_argument("--cross_attention_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--unfreeze_text_layers",
        type=int,
        default=0,
        help="0 freezes BERT, N fine-tunes the last N layers, -1 fine-tunes all.",
    )
    parser.add_argument(
        "--unfreeze_audio_layers",
        type=int,
        default=0,
        help="0 freezes wav2vec, N fine-tunes the last N layers, -1 fine-tunes all.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--metric_for_best", type=str, default=MACRO_F1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a full checkpoint, usually output_dir/last_checkpoint.pt.",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically resume from output_dir/last_checkpoint.pt when it exists.",
    )
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    return parser.parse_args()


def resolve_data_paths(args: argparse.Namespace) -> argparse.Namespace:
    data_root = args.data_root
    args.train_file = args.train_file or data_root / "split" / "train.txt"
    args.val_file = args.val_file or data_root / "split" / "val.txt"
    args.test_file = args.test_file or data_root / "split" / "test.txt"
    args.audio_root = args.audio_root or data_root / "audios"
    return args


def build_loaders(
    args: argparse.Namespace,
    tokenizer: Any,
    audio_feature_extractor: Any,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    BAHTextAudioDataset,
]:
    sampling_rate = int(getattr(audio_feature_extractor, "sampling_rate", 16000))
    collator = TextAudioCollator(
        tokenizer=tokenizer,
        audio_feature_extractor=audio_feature_extractor,
        sampling_rate=sampling_rate,
        max_text_length=args.max_text_length,
    )
    train_dataset = BAHTextAudioDataset(
        manifest_path=args.train_file,
        audio_root=args.audio_root,
        target_sampling_rate=sampling_rate,
        max_audio_seconds=args.max_audio_seconds,
        crop_mode=args.train_audio_crop,
        limit=args.limit_train,
    )
    val_dataset = BAHTextAudioDataset(
        manifest_path=args.val_file,
        audio_root=args.audio_root,
        target_sampling_rate=sampling_rate,
        max_audio_seconds=args.max_audio_seconds,
        crop_mode=args.eval_audio_crop,
        limit=args.limit_val,
    )
    test_dataset = BAHTextAudioDataset(
        manifest_path=args.test_file,
        audio_root=args.audio_root,
        target_sampling_rate=sampling_rate,
        max_audio_seconds=args.max_audio_seconds,
        crop_mode=args.eval_audio_crop,
        limit=args.limit_test,
    )
    train_loader = build_dataloader(
        train_dataset,
        collator=collator,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        collator=collator,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_dataloader(
        test_dataset,
        collator=collator,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader, train_dataset


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        return args.resume

    latest_checkpoint = args.output_dir / "last_checkpoint.pt"
    if args.auto_resume and latest_checkpoint.exists():
        return latest_checkpoint
    return None


def load_training_state(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    metric_for_best: str,
) -> tuple[int, float, list[dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    loaded_full_state = True
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        loaded_full_state = False
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        loaded_full_state = False
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    completed_epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics", {})
    best_metric = float(
        checkpoint.get("best_metric", metrics.get(metric_for_best, float("-inf")))
    )
    raw_history = checkpoint.get("history", [])
    history = raw_history if isinstance(raw_history, list) else []
    rng_state = checkpoint.get("rng_state")
    if isinstance(rng_state, dict):
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"].detach().cpu())
        cuda_rng_state = rng_state.get("cuda")
        if torch.cuda.is_available() and cuda_rng_state:
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in cuda_rng_state]
            )

    if loaded_full_state:
        print(f"resumed training from {checkpoint_path} at epoch {completed_epoch}")
    else:
        print(
            f"loaded model weights from {checkpoint_path}; optimizer/scheduler state "
            "was missing, so training will continue with fresh optimizer state"
        )
    return completed_epoch + 1, best_metric, history


def main() -> None:
    args = resolve_data_paths(parse_args())
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_args = args_to_dict(args)

    device = resolve_device(args.no_cuda)
    mixed_precision = args.mixed_precision and device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.text_model, cache_dir=args.cache_dir)
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.audio_model,
        cache_dir=args.cache_dir,
    )
    train_loader, val_loader, test_loader, train_dataset = build_loaders(
        args,
        tokenizer,
        audio_feature_extractor,
    )

    model = BertWav2VecCrossAttentionClassifier(
        text_model_name=args.text_model,
        audio_model_name=args.audio_model,
        fusion_dim=args.fusion_dim,
        cross_attention_heads=args.cross_attention_heads,
        cross_attention_layers=args.cross_attention_layers,
        dropout=args.dropout,
        unfreeze_text_layers=args.unfreeze_text_layers,
        unfreeze_audio_layers=args.unfreeze_audio_layers,
        cache_dir=args.cache_dir,
    ).to(device)

    param_counts = count_parameters(model)
    save_json(
        {
            "args": checkpoint_args,
            "device": str(device),
            "parameters": param_counts,
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
        },
        args.output_dir / "run_config.json",
    )

    train_steps = math.ceil(
        len(train_loader) / max(args.gradient_accumulation_steps, 1)
    ) * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        train_steps=train_steps,
        warmup_ratio=args.warmup_ratio,
    )
    scaler = GradScaler(device="cuda", enabled=mixed_precision)

    class_weights = None
    if args.class_weight == "balanced":
        class_weights = compute_class_weights(
            [example.label for example in train_dataset.examples],
            device=device,
        )

    best_metric = float("-inf")
    history: list[dict[str, Any]] = []
    best_checkpoint = args.output_dir / "best_model.pt"
    latest_checkpoint = args.output_dir / "last_checkpoint.pt"
    start_epoch = 1

    resume_path = resolve_resume_path(args)
    if resume_path is not None:
        start_epoch, best_metric, history = load_training_state(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            metric_for_best=args.metric_for_best,
        )

    if start_epoch > args.epochs:
        print(
            f"checkpoint already completed epoch {start_epoch - 1}; "
            f"target epochs is {args.epochs}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            class_weights=class_weights,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
        )
        val_loss, val_metrics, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            class_weights=class_weights,
            mixed_precision=mixed_precision,
            desc=f"val epoch {epoch}",
        )
        current_metric = float(val_metrics.get(args.metric_for_best, -val_loss))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{
                key: value
                for key, value in val_metrics.items()
                if isinstance(value, (int, float))
            },
        }
        history.append(row)
        write_history(history, args.output_dir / "training_history.csv")
        save_metrics(
            val_metrics,
            args.output_dir / f"metrics_val_epoch_{epoch}.json",
            args.output_dir / f"metrics_val_epoch_{epoch}.csv",
        )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} {args.metric_for_best}={current_metric:.4f}"
        )

        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(
                model=model,
                output_path=best_checkpoint,
                args=checkpoint_args,
                metrics=val_metrics,
                epoch=epoch,
                best_metric=best_metric,
                history=history,
            )
            print(f"saved best checkpoint: {best_checkpoint}")

        save_checkpoint(
            model=model,
            output_path=latest_checkpoint,
            args=checkpoint_args,
            metrics=val_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            history=history,
        )
        print(f"saved latest checkpoint: {latest_checkpoint}")

    final_checkpoint = best_checkpoint if best_checkpoint.exists() else latest_checkpoint
    checkpoint = torch.load(final_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_metrics, prediction_rows = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        class_weights=class_weights,
        mixed_precision=mixed_precision,
        desc="test",
    )
    save_checkpoint(
        model=model,
        output_path=args.output_dir / "last_loaded_best_model.pt",
        args=checkpoint_args,
        metrics=test_metrics,
    )
    save_metrics(
        test_metrics,
        args.output_dir / "metrics_test.json",
        args.output_dir / "metrics_test.csv",
    )
    save_predictions(prediction_rows, args.output_dir / "predictions_test.csv")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    audio_feature_extractor.save_pretrained(args.output_dir / "audio_feature_extractor")

    print("test metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
