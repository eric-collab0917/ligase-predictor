#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ESM-2 based regression model for ligase optimal pH and temperature.

Features:
- Multi-task regression (pH + temperature)
- Masked loss: supports samples with only pH or only temperature
- Huber loss (robust to outliers) or MSE
- Early stopping on validation loss
- MPS/CUDA/CPU auto-detection
- Two-stage training: load pretrained weights and finetune

Usage:
  # Stage 1: Pretrain on all enzymes
  python scripts/train_ligase_ph_temp.py \
      --csv data/processed/enzyme_ph_temp_all.csv \
      --outdir outputs/enzyme_ph_temp_pretrain \
      --epochs 15 --batch-size 16

  # Stage 2: Finetune on ligases
  python scripts/train_ligase_ph_temp.py \
      --csv data/processed/ligase_ph_temp_auto.csv \
      --outdir outputs/ligase_ph_temp_finetuned \
      --pretrained outputs/enzyme_ph_temp_pretrain/best_ligase_ph_temp.pt \
      --epochs 20 --lr 5e-4
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ligase_ph_temp import (
    LigasePhTempModel,
    clean_sequence,
    compute_regression_metrics,
    get_device,
    masked_cross_entropy_loss,
    masked_huber_loss,
    masked_mse_loss,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def snap_series(series: pd.Series, step: float) -> pd.Series:
    if step <= 0:
        return series
    return np.round(series / step) * step


def build_quantile_bin_edges(values: np.ndarray, num_bins: int) -> list:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0 or num_bins <= 1:
        return []
    q = np.linspace(0.0, 1.0, num_bins + 1)
    raw = np.quantile(values, q)
    edges = [0.0]
    for x in raw[1:-1]:
        x = float(x)
        if x > edges[-1] + 1e-6:
            edges.append(x)
    edges.append(14.000001)
    return edges if len(edges) >= 3 else []


def assign_bins(values: pd.Series, edges: list) -> np.ndarray:
    if not edges:
        return np.full(len(values), -1, dtype=np.int64)
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(len(arr), -1, dtype=np.int64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    out[mask] = np.digitize(arr[mask], np.asarray(edges[1:-1], dtype=np.float64), right=False)
    return out


def compute_bin_weights(bin_idx: np.ndarray, num_bins: int, power: float) -> tuple:
    valid = bin_idx >= 0
    counts = np.bincount(bin_idx[valid], minlength=num_bins) if np.any(valid) else np.zeros(num_bins, dtype=np.int64)
    counts = np.maximum(counts, 1)
    weights = (counts.max() / counts.astype(np.float64)) ** float(power)
    weights = weights / max(weights.mean(), 1e-12)
    sample_weights = np.ones(len(bin_idx), dtype=np.float32)
    sample_weights[valid] = weights[bin_idx[valid]].astype(np.float32)
    return counts.tolist(), weights.astype(np.float32), sample_weights


def prepare_clean_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ph_col: str,
    temp_col: str,
    ph_round_to: float,
    ph_winsor_lower: float,
    ph_winsor_upper: float,
):
    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df[ph_col] = pd.to_numeric(train_df[ph_col], errors="coerce")
    val_df[ph_col] = pd.to_numeric(val_df[ph_col], errors="coerce")
    train_df[temp_col] = pd.to_numeric(train_df[temp_col], errors="coerce")
    val_df[temp_col] = pd.to_numeric(val_df[temp_col], errors="coerce")

    train_df.loc[~train_df[ph_col].between(0, 14, inclusive="both"), ph_col] = np.nan
    val_df.loc[~val_df[ph_col].between(0, 14, inclusive="both"), ph_col] = np.nan
    train_df.loc[~train_df[temp_col].between(0, 120, inclusive="both"), temp_col] = np.nan
    val_df.loc[~val_df[temp_col].between(0, 120, inclusive="both"), temp_col] = np.nan

    ph_clip = None
    if 0.0 <= ph_winsor_lower < ph_winsor_upper <= 1.0:
        valid = train_df[ph_col].dropna().to_numpy(dtype=np.float64)
        if len(valid) > 0:
            lo = float(np.quantile(valid, ph_winsor_lower))
            hi = float(np.quantile(valid, ph_winsor_upper))
            train_df.loc[train_df[ph_col].notna(), ph_col] = train_df.loc[train_df[ph_col].notna(), ph_col].clip(lo, hi)
            ph_clip = [lo, hi]

    if ph_round_to > 0:
        mask = train_df[ph_col].notna()
        train_df.loc[mask, ph_col] = snap_series(train_df.loc[mask, ph_col], ph_round_to)

    return train_df, val_df, ph_clip


def compute_auto_ph_loss_weight(train_df: pd.DataFrame, ph_col: str, temp_col: str, base_weight: float, enabled: bool, max_weight: float) -> float:
    if not enabled:
        return float(base_weight)
    ph_vals = train_df[ph_col].dropna().to_numpy(dtype=np.float64)
    temp_vals = train_df[temp_col].dropna().to_numpy(dtype=np.float64)
    if len(ph_vals) < 2 or len(temp_vals) < 2:
        return float(base_weight)
    ph_std = float(np.std(ph_vals))
    temp_std = float(np.std(temp_vals))
    if ph_std <= 1e-8:
        return float(base_weight)
    ratio = temp_std / ph_std
    auto_weight = min(max_weight, max(1.0, ratio ** 0.5))
    return float(base_weight * auto_weight)


def metric_for_selection(metrics: dict, key: str) -> float:
    if key == "val_loss":
        val = float(metrics.get("val_loss", np.inf))
        return val if np.isfinite(val) else float("inf")
    if key == "ph_mae":
        val = float(metrics.get("ph_mae", np.inf))
        return val if np.isfinite(val) else float("inf")
    if key == "ph_rmse":
        val = float(metrics.get("ph_rmse", np.inf))
        return val if np.isfinite(val) else float("inf")
    val = float(metrics.get("val_loss", np.inf))
    return val if np.isfinite(val) else float("inf")


class PhTempDataset(Dataset):
    """Dataset for pH/temperature regression with masking support."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_col: str = "sequence",
        ph_col: str = "opt_ph",
        temp_col: str = "opt_temp",
        ph_bin_col: str = "ph_bin",
        ph_weight_col: str = "ph_weight",
    ):
        self.samples = []
        for _, row in df.iterrows():
            seq = clean_sequence(row.get(seq_col, ""))
            if not seq:
                continue

            # Parse pH
            ph_val = row.get(ph_col, "")
            has_ph = False
            ph_target = 0.0
            try:
                ph_target = float(ph_val)
                if 0 <= ph_target <= 14:
                    has_ph = True
                else:
                    ph_target = 0.0
                    has_ph = False
            except (ValueError, TypeError):
                pass

            # Parse temperature
            temp_val = row.get(temp_col, "")
            has_temp = False
            temp_target = 0.0
            try:
                temp_target = float(temp_val)
                if 0 <= temp_target <= 120:
                    has_temp = True
                else:
                    temp_target = 0.0
                    has_temp = False
            except (ValueError, TypeError):
                pass

            # Skip if neither pH nor temperature is available
            if not has_ph and not has_temp:
                continue

            self.samples.append(
                {
                    "sequence": seq,
                    "ph_target": ph_target,
                    "has_ph": float(has_ph),
                    "ph_bin": int(row.get(ph_bin_col, -1)) if has_ph else -1,
                    "ph_weight": float(row.get(ph_weight_col, 1.0)) if has_ph else 1.0,
                    "temp_target": temp_target,
                    "has_temp": float(has_temp),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(tokenizer, max_length: int):
    def _collate(batch):
        seqs = [x["sequence"] for x in batch]
        enc = tokenizer(
            seqs,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ph_target = torch.tensor(
            [x["ph_target"] for x in batch], dtype=torch.float32
        )
        has_ph = torch.tensor([x["has_ph"] for x in batch], dtype=torch.float32)
        ph_bin = torch.tensor([x["ph_bin"] for x in batch], dtype=torch.long)
        ph_weight = torch.tensor([x["ph_weight"] for x in batch], dtype=torch.float32)
        temp_target = torch.tensor(
            [x["temp_target"] for x in batch], dtype=torch.float32
        )
        has_temp = torch.tensor(
            [x["has_temp"] for x in batch], dtype=torch.float32
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "ph_target": ph_target,
            "has_ph": has_ph,
            "ph_bin": ph_bin,
            "ph_weight": ph_weight,
            "temp_target": temp_target,
            "has_temp": has_temp,
        }

    return _collate


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_type: str = "huber",
    ph_loss_weight: float = 1.0,
    temp_loss_weight: float = 1.0,
    huber_delta: float = 1.0,
    ph_aux_loss_weight: float = 0.0,
    ph_class_weight: torch.Tensor = None,
    ph_label_smoothing: float = 0.0,
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ph_target = batch["ph_target"].to(device)
        has_ph = batch["has_ph"].to(device)
        ph_bin = batch["ph_bin"].to(device)
        ph_weight = batch["ph_weight"].to(device)
        temp_target = batch["temp_target"].to(device)
        has_temp = batch["has_temp"].to(device)

        ph_pred, temp_pred, aux = model(input_ids, attention_mask, return_aux=True)

        loss = torch.tensor(0.0, device=device)

        # pH loss
        if has_ph.sum() > 0:
            if loss_type == "huber":
                loss_ph = masked_huber_loss(
                    ph_pred, ph_target, has_ph, delta=huber_delta, sample_weight=ph_weight
                )
            else:
                loss_ph = masked_mse_loss(ph_pred, ph_target, has_ph, sample_weight=ph_weight)
            loss = loss + ph_loss_weight * loss_ph

            if ph_aux_loss_weight > 0.0 and "ph_bin_logits" in aux:
                loss_ph_aux = masked_cross_entropy_loss(
                    aux["ph_bin_logits"],
                    ph_bin,
                    has_ph,
                    class_weight=ph_class_weight,
                    sample_weight=ph_weight,
                    label_smoothing=ph_label_smoothing,
                )
                loss = loss + ph_aux_loss_weight * loss_ph_aux

        # Temperature loss
        if has_temp.sum() > 0:
            if loss_type == "huber":
                loss_temp = masked_huber_loss(
                    temp_pred, temp_target, has_temp, delta=huber_delta
                )
            else:
                loss_temp = masked_mse_loss(temp_pred, temp_target, has_temp)
            loss = loss + temp_loss_weight * loss_temp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    loss_type: str = "huber",
    ph_loss_weight: float = 1.0,
    temp_loss_weight: float = 1.0,
    huber_delta: float = 1.0,
    ph_aux_loss_weight: float = 0.0,
    ph_class_weight: torch.Tensor = None,
    ph_label_smoothing: float = 0.0,
):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    ph_true_list, ph_pred_list = [], []
    temp_true_list, temp_pred_list = [], []
    ph_bin_true_list, ph_bin_pred_list = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ph_target = batch["ph_target"].to(device)
        has_ph = batch["has_ph"].to(device)
        ph_bin = batch["ph_bin"].to(device)
        ph_weight = batch["ph_weight"].to(device)
        temp_target = batch["temp_target"].to(device)
        has_temp = batch["has_temp"].to(device)

        ph_pred, temp_pred, aux = model(input_ids, attention_mask, return_aux=True)

        loss = torch.tensor(0.0, device=device)

        if has_ph.sum() > 0:
            if loss_type == "huber":
                loss_ph = masked_huber_loss(
                    ph_pred, ph_target, has_ph, delta=huber_delta, sample_weight=ph_weight
                )
            else:
                loss_ph = masked_mse_loss(ph_pred, ph_target, has_ph, sample_weight=ph_weight)
            loss = loss + ph_loss_weight * loss_ph

            if ph_aux_loss_weight > 0.0 and "ph_bin_logits" in aux:
                loss_ph_aux = masked_cross_entropy_loss(
                    aux["ph_bin_logits"],
                    ph_bin,
                    has_ph,
                    class_weight=ph_class_weight,
                    sample_weight=ph_weight,
                    label_smoothing=ph_label_smoothing,
                )
                loss = loss + ph_aux_loss_weight * loss_ph_aux

            # Collect for metrics
            mask = has_ph > 0.5
            ph_true_list.append(ph_target[mask].cpu().numpy())
            ph_pred_list.append(ph_pred[mask].cpu().numpy())
            if "ph_bin_logits" in aux:
                ph_bin_true_list.append(ph_bin[mask].cpu().numpy())
                ph_bin_pred_list.append(aux["ph_bin_logits"][mask].argmax(dim=1).cpu().numpy())

        if has_temp.sum() > 0:
            if loss_type == "huber":
                loss_temp = masked_huber_loss(
                    temp_pred, temp_target, has_temp, delta=huber_delta
                )
            else:
                loss_temp = masked_mse_loss(temp_pred, temp_target, has_temp)
            loss = loss + temp_loss_weight * loss_temp

            mask = has_temp > 0.5
            temp_true_list.append(temp_target[mask].cpu().numpy())
            temp_pred_list.append(temp_pred[mask].cpu().numpy())

        total_loss += float(loss.item())
        n_batches += 1

    metrics = {"val_loss": total_loss / max(n_batches, 1)}

    # pH metrics
    if ph_true_list:
        ph_true = np.concatenate(ph_true_list)
        ph_pred = np.concatenate(ph_pred_list)
        ph_metrics = compute_regression_metrics(ph_true, ph_pred)
        metrics["ph_mae"] = ph_metrics["mae"]
        metrics["ph_rmse"] = ph_metrics["rmse"]
        metrics["ph_r2"] = ph_metrics["r2"]
        metrics["ph_pearson_r"] = ph_metrics["pearson_r"]
        if ph_bin_true_list:
            y_true = np.concatenate(ph_bin_true_list)
            y_pred = np.concatenate(ph_bin_pred_list)
            metrics["ph_bin_acc"] = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else float("nan")
        else:
            metrics["ph_bin_acc"] = float("nan")
    else:
        metrics["ph_mae"] = float("nan")
        metrics["ph_rmse"] = float("nan")
        metrics["ph_r2"] = float("nan")
        metrics["ph_pearson_r"] = float("nan")
        metrics["ph_bin_acc"] = float("nan")

    # Temperature metrics
    if temp_true_list:
        temp_true = np.concatenate(temp_true_list)
        temp_pred = np.concatenate(temp_pred_list)
        temp_metrics = compute_regression_metrics(temp_true, temp_pred)
        metrics["temp_mae"] = temp_metrics["mae"]
        metrics["temp_rmse"] = temp_metrics["rmse"]
        metrics["temp_r2"] = temp_metrics["r2"]
        metrics["temp_pearson_r"] = temp_metrics["pearson_r"]
    else:
        metrics["temp_mae"] = float("nan")
        metrics["temp_rmse"] = float("nan")
        metrics["temp_r2"] = float("nan")
        metrics["temp_pearson_r"] = float("nan")

    return metrics


def fmt_float(v, decimals: int = 4):
    try:
        if v is None or np.isnan(v):
            return "nan"
    except Exception:
        pass
    return f"{float(v):.{decimals}f}"


def main():
    ap = argparse.ArgumentParser(
        description="Train ligase pH/temperature regression model"
    )
    ap.add_argument("--csv", required=True, help="Input CSV with sequence/pH/temp")
    ap.add_argument("--outdir", default="./outputs/ligase_ph_temp")
    ap.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--ph-col", default="opt_ph")
    ap.add_argument("--temp-col", default="opt_temp")
    ap.add_argument("--valid-size", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true"
    )
    ap.add_argument(
        "--no-freeze-backbone", dest="freeze_backbone", action="store_false"
    )
    ap.set_defaults(freeze_backbone=True)
    ap.add_argument("--unfreeze-last-n-layers", type=int, default=0)
    ap.add_argument(
        "--loss-type", choices=["mse", "huber"], default="huber"
    )
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--ph-loss-weight", type=float, default=1.0)
    ap.add_argument("--temp-loss-weight", type=float, default=1.0)
    ap.add_argument(
        "--auto-ph-loss-balance",
        action="store_true",
        help="Scale pH loss weight based on target spread to reduce temperature dominance",
    )
    ap.add_argument(
        "--max-auto-ph-loss-weight",
        type=float,
        default=8.0,
        help="Upper cap for automatically scaled pH loss weight",
    )
    ap.add_argument(
        "--ph-round-to",
        type=float,
        default=0.0,
        help="Optional snapping step for pH labels, e.g. 0.1 or 0.5",
    )
    ap.add_argument(
        "--ph-winsor-lower",
        type=float,
        default=-1.0,
        help="Lower quantile for pH winsorization on training labels; negative disables it",
    )
    ap.add_argument(
        "--ph-winsor-upper",
        type=float,
        default=-1.0,
        help="Upper quantile for pH winsorization on training labels; negative disables it",
    )
    ap.add_argument(
        "--ph-num-bins",
        type=int,
        default=0,
        help="If >1, add an auxiliary pH bin classification head with this many bins",
    )
    ap.add_argument(
        "--ph-bin-weight-power",
        type=float,
        default=0.5,
        help="Power used for inverse-frequency pH bin reweighting",
    )
    ap.add_argument(
        "--ph-aux-loss-weight",
        type=float,
        default=0.0,
        help="Loss weight for auxiliary pH bin classification",
    )
    ap.add_argument(
        "--ph-label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for auxiliary pH bin classification",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument(
        "--early-stop-metric",
        choices=["val_loss", "ph_mae", "ph_rmse"],
        default="val_loss",
        help="Metric used for model selection and early stopping",
    )
    ap.add_argument(
        "--pretrained",
        default="",
        help="Path to pretrained checkpoint for finetuning (two-stage training)",
    )
    ap.add_argument(
        "--finetune-lr-scale",
        type=float,
        default=0.5,
        help="Scale factor for learning rate when finetuning (default: 0.5)",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv)
    if args.seq_col not in df.columns:
        raise KeyError(f"Sequence column not found: {args.seq_col}")

    df = df.copy()
    df[args.seq_col] = df[args.seq_col].astype(str).map(clean_sequence)
    df = df[df[args.seq_col].str.len() > 0].reset_index(drop=True)

    # Count valid samples
    def has_valid_ph(row):
        try:
            v = float(row.get(args.ph_col, ""))
            return 0 <= v <= 14
        except (ValueError, TypeError):
            return False

    def has_valid_temp(row):
        try:
            v = float(row.get(args.temp_col, ""))
            return 0 <= v <= 120
        except (ValueError, TypeError):
            return False

    n_ph = sum(1 for _, r in df.iterrows() if has_valid_ph(r))
    n_temp = sum(1 for _, r in df.iterrows() if has_valid_temp(r))
    n_both = sum(
        1 for _, r in df.iterrows() if has_valid_ph(r) and has_valid_temp(r)
    )
    print(f"[Info] Total rows: {len(df)}, with pH: {n_ph}, with temp: {n_temp}, with both: {n_both}")

    if n_ph + n_temp == 0:
        raise ValueError("No valid pH or temperature data found.")

    # Split
    train_df_raw, val_df_raw = train_test_split(
        df, test_size=args.valid_size, random_state=args.seed, shuffle=True
    )
    print(f"[Info] train={len(train_df_raw)}, val={len(val_df_raw)}")

    train_df, val_df, ph_clip = prepare_clean_targets(
        train_df=train_df_raw,
        val_df=val_df_raw,
        ph_col=args.ph_col,
        temp_col=args.temp_col,
        ph_round_to=args.ph_round_to,
        ph_winsor_lower=args.ph_winsor_lower,
        ph_winsor_upper=args.ph_winsor_upper,
    )

    ph_edges = []
    ph_counts = []
    ph_class_weight = None
    if args.ph_num_bins and args.ph_num_bins > 1:
        ph_train_values = train_df[args.ph_col].dropna().to_numpy(dtype=np.float64)
        ph_edges = build_quantile_bin_edges(ph_train_values, args.ph_num_bins)
        if ph_edges:
            train_df["ph_bin"] = assign_bins(train_df[args.ph_col], ph_edges)
            val_df["ph_bin"] = assign_bins(val_df[args.ph_col], ph_edges)
            num_bins = len(ph_edges) - 1
            ph_counts, ph_class_weight_np, train_ph_weight_np = compute_bin_weights(
                train_df["ph_bin"].to_numpy(dtype=np.int64),
                num_bins=num_bins,
                power=args.ph_bin_weight_power,
            )
            train_df["ph_weight"] = train_ph_weight_np
            val_df["ph_weight"] = 1.0
            ph_class_weight = torch.tensor(ph_class_weight_np, dtype=torch.float32)
            print(f"[Info] pH auxiliary bins={num_bins}, edges={ph_edges}, counts={ph_counts}")
        else:
            train_df["ph_bin"] = -1
            val_df["ph_bin"] = -1
            train_df["ph_weight"] = 1.0
            val_df["ph_weight"] = 1.0
            print("[Warn] Could not build stable pH bin edges; disabling pH auxiliary head.")
    else:
        train_df["ph_bin"] = -1
        val_df["ph_bin"] = -1
        train_df["ph_weight"] = 1.0
        val_df["ph_weight"] = 1.0

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = PhTempDataset(
        train_df,
        seq_col=args.seq_col,
        ph_col=args.ph_col,
        temp_col=args.temp_col,
    )
    val_ds = PhTempDataset(
        val_df,
        seq_col=args.seq_col,
        ph_col=args.ph_col,
        temp_col=args.temp_col,
    )
    print(f"[Info] train samples (with labels): {len(train_ds)}, val samples: {len(val_ds)}")

    if len(train_ds) < 10:
        raise ValueError(f"Too few training samples: {len(train_ds)}")

    collate_fn = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Model
    device = get_device(args.device)
    print(f"[Info] Using device: {device}")

    effective_ph_loss_weight = compute_auto_ph_loss_weight(
        train_df=train_df,
        ph_col=args.ph_col,
        temp_col=args.temp_col,
        base_weight=args.ph_loss_weight,
        enabled=bool(args.auto_ph_loss_balance),
        max_weight=args.max_auto_ph_loss_weight,
    )
    print(
        f"[Info] pH loss weight={effective_ph_loss_weight:.4f} "
        f"(base={args.ph_loss_weight}, auto_balance={args.auto_ph_loss_balance})"
    )

    model = LigasePhTempModel(
        model_name=args.model_name,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        hidden_dim=args.hidden_dim,
        num_ph_bins=(len(ph_edges) - 1) if ph_edges else 0,
    ).to(device)

    # Load pretrained weights if specified (for two-stage training)
    actual_lr = args.lr
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"[Info] Loading pretrained weights from: {args.pretrained}")
        pretrained_ckpt = torch.load(args.pretrained, map_location="cpu")
        # Load only the regression heads (ph_head, temp_head), not the backbone
        pretrained_state = pretrained_ckpt.get("model_state", pretrained_ckpt)
        model_state = model.state_dict()
        
        # Filter and load matching keys
        loaded_keys = []
        for k, v in pretrained_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded_keys.append(k)
        
        model.load_state_dict(model_state)
        print(f"[Info] Loaded {len(loaded_keys)} parameter tensors from pretrained model")
        
        # Use lower learning rate for finetuning
        actual_lr = args.lr * args.finetune_lr_scale
        print(f"[Info] Finetuning mode: lr scaled from {args.lr} to {actual_lr}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[Info] Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable, lr=actual_lr, weight_decay=args.weight_decay
    )
    if ph_class_weight is not None:
        ph_class_weight = ph_class_weight.to(device)

    # Training loop
    best_metric = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_type=args.loss_type,
            ph_loss_weight=effective_ph_loss_weight,
            temp_loss_weight=args.temp_loss_weight,
            huber_delta=args.huber_delta,
            ph_aux_loss_weight=args.ph_aux_loss_weight,
            ph_class_weight=ph_class_weight,
            ph_label_smoothing=args.ph_label_smoothing,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_type=args.loss_type,
            ph_loss_weight=effective_ph_loss_weight,
            temp_loss_weight=args.temp_loss_weight,
            huber_delta=args.huber_delta,
            ph_aux_loss_weight=args.ph_aux_loss_weight,
            ph_class_weight=ph_class_weight,
            ph_label_smoothing=args.ph_label_smoothing,
        )

        row = {"epoch": epoch, "train_loss": tr_loss, **val_metrics}
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={tr_loss:.4f} val_loss={val_metrics['val_loss']:.4f} | "
            f"pH: MAE={fmt_float(val_metrics['ph_mae'])} "
            f"R²={fmt_float(val_metrics['ph_r2'])} "
            f"BinAcc={fmt_float(val_metrics.get('ph_bin_acc'))} | "
            f"Temp: MAE={fmt_float(val_metrics['temp_mae'])} R²={fmt_float(val_metrics['temp_r2'])}"
        )

        current_metric = metric_for_selection(val_metrics, args.early_stop_metric)

        # Early stopping / model selection
        if current_metric < best_metric - 1e-6:
            best_metric = current_metric
            best_epoch = epoch
            bad_epochs = 0

            ckpt = {
                "model_state": model.state_dict(),
                "config": {
                    "model_name": args.model_name,
                    "max_length": args.max_length,
                    "dropout": args.dropout,
                    "hidden_dim": args.hidden_dim,
                    "freeze_backbone": args.freeze_backbone,
                    "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
                    "loss_type": args.loss_type,
                    "huber_delta": args.huber_delta,
                    "ph_loss_weight": effective_ph_loss_weight,
                    "temp_loss_weight": args.temp_loss_weight,
                    "base_ph_loss_weight": args.ph_loss_weight,
                    "auto_ph_loss_balance": bool(args.auto_ph_loss_balance),
                    "max_auto_ph_loss_weight": args.max_auto_ph_loss_weight,
                    "ph_round_to": args.ph_round_to,
                    "ph_winsor_lower": args.ph_winsor_lower,
                    "ph_winsor_upper": args.ph_winsor_upper,
                    "ph_num_bins": (len(ph_edges) - 1) if ph_edges else 0,
                    "ph_bin_edges": ph_edges,
                    "ph_bin_weight_power": args.ph_bin_weight_power,
                    "ph_aux_loss_weight": args.ph_aux_loss_weight,
                    "ph_label_smoothing": args.ph_label_smoothing,
                    "early_stop_metric": args.early_stop_metric,
                },
                "best_epoch": best_epoch,
                "best_val_loss": val_metrics["val_loss"],
                "best_selection_metric": best_metric,
                "best_metrics": {
                    "ph_mae": val_metrics.get("ph_mae"),
                    "ph_rmse": val_metrics.get("ph_rmse"),
                    "ph_r2": val_metrics.get("ph_r2"),
                    "ph_pearson_r": val_metrics.get("ph_pearson_r"),
                    "ph_bin_acc": val_metrics.get("ph_bin_acc"),
                    "temp_mae": val_metrics.get("temp_mae"),
                    "temp_rmse": val_metrics.get("temp_rmse"),
                    "temp_r2": val_metrics.get("temp_r2"),
                    "temp_pearson_r": val_metrics.get("temp_pearson_r"),
                },
            }
            torch.save(ckpt, outdir / "best_ligase_ph_temp.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[Info] Early stopping at epoch {epoch}, best epoch={best_epoch}")
                break

    # Save history
    pd.DataFrame(history).to_csv(outdir / "train_history.csv", index=False)

    # Save config
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "max_length": args.max_length,
                "dropout": args.dropout,
                "hidden_dim": args.hidden_dim,
                "freeze_backbone": args.freeze_backbone,
                "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
                "loss_type": args.loss_type,
                "huber_delta": args.huber_delta,
                "ph_loss_weight": effective_ph_loss_weight,
                "temp_loss_weight": args.temp_loss_weight,
                "base_ph_loss_weight": args.ph_loss_weight,
                "auto_ph_loss_balance": bool(args.auto_ph_loss_balance),
                "max_auto_ph_loss_weight": args.max_auto_ph_loss_weight,
                "ph_round_to": args.ph_round_to,
                "ph_winsor_lower": args.ph_winsor_lower,
                "ph_winsor_upper": args.ph_winsor_upper,
                "ph_num_bins": (len(ph_edges) - 1) if ph_edges else 0,
                "ph_bin_edges": ph_edges,
                "ph_bin_counts": ph_counts,
                "ph_bin_weight_power": args.ph_bin_weight_power,
                "ph_aux_loss_weight": args.ph_aux_loss_weight,
                "ph_label_smoothing": args.ph_label_smoothing,
                "ph_clip_range": ph_clip,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_selection_metric": best_metric,
                "early_stop_metric": args.early_stop_metric,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[Done] Best model: {outdir / 'best_ligase_ph_temp.pt'}")
    print(f"[Done] Best {args.early_stop_metric}: {best_metric:.4f} at epoch {best_epoch}")
    print(f"[Done] History: {outdir / 'train_history.csv'}")


if __name__ == "__main__":
    main()
