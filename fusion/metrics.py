from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


MACRO_F1 = "MACRO_F1"
W_F1 = "W_F1"
CL_ACC = "CL_ACC"
CFUSE_MARIX = "CONFUSION_MATRIX"
F1_POS = "F1_POS"
F1_NEG = "F1_NEG"
AP_POS = "Average_precision_POS"
CASP_ACC2 = "CASP_ACC2"
CASP_F1_WEIGHTED = "CASP_F1_WEIGHTED"


def logits_to_predictions(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    hard_preds = probs.argmax(axis=1)
    pos_scores = probs[:, 1]
    return hard_preds.astype(int), pos_scores.astype(float)


def bah_perfs(
    gt: np.ndarray,
    hard_preds: np.ndarray,
    pos_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    if not isinstance(gt, np.ndarray):
        gt = np.asarray(gt)
    if not isinstance(hard_preds, np.ndarray):
        hard_preds = np.asarray(hard_preds)

    assert gt.shape == hard_preds.shape, f"{gt.shape} | {hard_preds.shape}"
    gt = gt.astype(int)
    hard_preds = hard_preds.astype(int)

    cl_acc = float((gt == hard_preds).mean().item())
    conf_mtx = confusion_matrix(
        y_true=gt,
        y_pred=hard_preds,
        labels=[0, 1],
        normalize="true",
    )
    conf_mtx = np.nan_to_num(conf_mtx, nan=0.0)

    f1_s = f1_score(gt, hard_preds, labels=[0, 1], average=None, zero_division=0)
    f1_neg = float(f1_s[0].item())
    f1_pos = float(f1_s[1].item())
    macro_f1 = float(np.mean(f1_s).item())
    wf1 = float(f1_score(gt, hard_preds, average="weighted", zero_division=0))

    ap_pos = 0.0
    if pos_scores is not None:
        pos_scores = np.asarray(pos_scores, dtype=float)
        if pos_scores.shape == gt.shape and len(np.unique(gt)) == 2:
            ap_pos = float(average_precision_score(gt, pos_scores))

    return {
        CL_ACC: cl_acc,
        CFUSE_MARIX: conf_mtx,
        F1_POS: f1_pos,
        F1_NEG: f1_neg,
        W_F1: wf1,
        MACRO_F1: macro_f1,
        AP_POS: ap_pos,
        CASP_ACC2: cl_acc,
        CASP_F1_WEIGHTED: wf1,
    }


def metrics_to_flat_row(metrics: dict[str, Any]) -> dict[str, Any]:
    conf = np.asarray(metrics[CFUSE_MARIX])
    return {
        CL_ACC: metrics[CL_ACC],
        F1_POS: metrics[F1_POS],
        F1_NEG: metrics[F1_NEG],
        W_F1: metrics[W_F1],
        MACRO_F1: metrics[MACRO_F1],
        AP_POS: metrics[AP_POS],
        CASP_ACC2: metrics[CASP_ACC2],
        CASP_F1_WEIGHTED: metrics[CASP_F1_WEIGHTED],
        "TN_RATE": float(conf[0, 0]),
        "FP_RATE": float(conf[0, 1]),
        "FN_RATE": float(conf[1, 0]),
        "TP_RATE": float(conf[1, 1]),
    }


def json_ready(metrics: dict[str, Any]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            ready[key] = value.tolist()
        elif isinstance(value, np.generic):
            ready[key] = value.item()
        else:
            ready[key] = value
    return ready


def save_metrics(metrics: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(metrics), f, indent=2, ensure_ascii=False)

    row = metrics_to_flat_row(metrics)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

