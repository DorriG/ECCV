from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler
from torch.amp import autocast
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .metrics import bah_perfs
from .metrics import json_ready
from .metrics import logits_to_predictions
from .utils import move_to_device


def compute_class_weights(labels: list[int], device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def optimizer_parameter_groups(
    model: torch.nn.Module,
    lr: float,
    encoder_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    encoder_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("text_encoder.") or name.startswith("audio_encoder."):
            encoder_params.append(parameter)
        else:
            head_params.append(parameter)

    groups: list[dict[str, Any]] = []
    if head_params:
        groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
    if encoder_params:
        groups.append(
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay}
        )
    return groups


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float,
    encoder_lr: float,
    weight_decay: float,
    train_steps: int,
    warmup_ratio: float,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, lr, encoder_lr, weight_decay)
    )
    warmup_steps = int(round(train_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=train_steps,
    )
    return optimizer, scheduler


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    class_weights: torch.Tensor | None,
    scaler: GradScaler,
    mixed_precision: bool,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_to_device(batch, device)
        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_values=batch["audio_values"],
                audio_attention_mask=batch["audio_attention_mask"],
                labels=batch["labels"],
                class_weights=class_weights,
            )
            loss = output["loss"] / gradient_accumulation_steps

        scaler.scale(loss).backward()
        batch_size = batch["labels"].shape[0]
        total_loss += float(loss.item()) * gradient_accumulation_steps * batch_size
        total_items += batch_size

        should_step = (
            step % gradient_accumulation_steps == 0 or step == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        progress.set_postfix(loss=total_loss / max(total_items, 1))

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    mixed_precision: bool = False,
    desc: str = "eval",
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    logits_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_to_device(batch, device)
        with autocast(device_type=device.type, enabled=mixed_precision):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_values=batch["audio_values"],
                audio_attention_mask=batch["audio_attention_mask"],
                labels=batch["labels"],
                class_weights=class_weights,
            )

        labels = batch["labels"]
        logits = output["logits"]
        batch_size = labels.shape[0]
        total_loss += float(output["loss"].item()) * batch_size
        total_items += batch_size
        logits_chunks.append(logits.detach().cpu().numpy())
        label_chunks.append(labels.detach().cpu().numpy())

        for idx in range(batch_size):
            meta_rows.append(
                {
                    "sample_id": batch["sample_ids"][idx],
                    "video_path": batch["video_paths"][idx],
                    "audio_path": batch["audio_paths"][idx],
                    "transcript": batch["transcripts"][idx],
                }
            )

    logits_np = np.concatenate(logits_chunks, axis=0)
    labels_np = np.concatenate(label_chunks, axis=0)
    hard_preds, pos_scores = logits_to_predictions(logits_np)
    metrics = bah_perfs(labels_np, hard_preds, pos_scores)
    metrics["loss"] = total_loss / max(total_items, 1)

    for row, label, pred, score, logits_row in zip(
        meta_rows,
        labels_np,
        hard_preds,
        pos_scores,
        logits_np,
    ):
        row["label"] = int(label)
        row["prediction"] = int(pred)
        row["prob_no_ah"] = float(1.0 - score)
        row["prob_ah"] = float(score)
        row["logit_no_ah"] = float(logits_row[0])
        row["logit_ah"] = float(logits_row[1])

    return metrics["loss"], metrics, meta_rows


def save_predictions(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    model: torch.nn.Module,
    output_path: Path,
    args: dict[str, Any],
    metrics: dict[str, Any] | None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scaler: GradScaler | None = None,
    epoch: int | None = None,
    best_metric: float | None = None,
    history: list[dict[str, Any]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "args": args,
        "metrics": json_ready(metrics or {}),
        "label_map": {"No A-H": 0, "A-H": 1},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
    if history is not None:
        checkpoint["history"] = history
    checkpoint["rng_state"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    torch.save(checkpoint, output_path)
