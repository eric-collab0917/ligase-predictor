#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a multi-label classification model for ligase subcellular localization
using ESM-2 backbone + classification head.

Usage:
  python scripts/train_ligase_subcellular.py \
      --csv data/processed/ligase_subcellular_auto.csv \
      --outdir outputs/ligase_subcellular_v1 \
      --model-name facebook/esm2_t6_8M_UR50D
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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ligase_subcellular import (
    LigaseSubcellularModel,
    build_label_map,
    clean_sequence,
    compute_pos_weight,
    get_device,
    is_explicit_none_label,
    multilabel_f1_score,
    parse_multilabel_cell,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SubcellularDataset(Dataset):
    def __init__(
        self,
        df,
        label_map,
        seq_col="sequence",
        label_col="subcellular_labels",
        sep=";",
    ):
        self.samples = []
        for _, row in df.iterrows():
            seq = clean_sequence(row[seq_col])
            if not seq:
                continue

            tokens = parse_multilabel_cell(row.get(label_col, ""), sep=sep)
            has_labels = len(tokens) > 0
            label_vec = np.zeros(len(label_map), dtype=np.float32)
            label_mask = np.zeros(len(label_map), dtype=np.float32)

            if has_labels and not is_explicit_none_label(tokens):
                mapped = 0
                for t in tokens:
                    if t in label_map:
                        label_vec[label_map[t]] = 1.0
                        mapped += 1
                if mapped == 0:
                    has_labels = False
                else:
                    # Enable ALL positions as targets: known positives=1, rest=0
                    label_mask[:] = 1.0
            elif has_labels and is_explicit_none_label(tokens):
                # Explicit "NONE" means known-negative for all labels
                label_mask[:] = 1.0

            self.samples.append({
                "sequence": seq,
                "label_vec": label_vec,
                "label_mask": label_mask,
                "has_labels": float(has_labels and label_mask.sum() > 0),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(tokenizer, max_length):
    def _collate(batch):
        seqs = [x["sequence"] for x in batch]
        enc = tokenizer(
            seqs,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        label_vec = torch.tensor(
            np.stack([x["label_vec"] for x in batch], axis=0), dtype=torch.float32
        )
        label_mask = torch.tensor(
            np.stack([x["label_mask"] for x in batch], axis=0), dtype=torch.float32
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label_vec": label_vec,
            "label_mask": label_mask,
        }
    return _collate


def masked_bce_loss(logits, targets, mask, pos_weight=None):
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
    )
    w = mask.to(loss.dtype)
    denom = torch.clamp(w.sum(), min=1.0)
    return (loss * w).sum() / denom


def train_one_epoch(model, loader, optimizer, device, pos_weight=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_vec = batch["label_vec"].to(device)
        label_mask = batch["label_mask"].to(device)

        logits = model(input_ids, attention_mask)
        loss = masked_bce_loss(logits, label_vec, label_mask, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(input_ids)

    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader, device, pos_weight=None, threshold=0.5,
             search_threshold=False, label_names=None):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_masks = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_vec = batch["label_vec"].to(device)
        label_mask = batch["label_mask"].to(device)

        logits = model(input_ids, attention_mask)
        loss = masked_bce_loss(logits, label_vec, label_mask, pos_weight=pos_weight)
        total_loss += loss.item() * len(input_ids)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_masks.append(label_mask.cpu().numpy())
        all_labels.append(label_vec.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = {"loss": total_loss / max(len(loader.dataset), 1)}

    # Find valid samples for each label dimension
    valid_mask = (all_masks.sum(axis=1) > 0)
    if valid_mask.sum() == 0:
        metrics["micro_f1"] = float("nan")
        metrics["macro_f1"] = float("nan")
        return metrics

    # Filter to only valid samples and labels with coverage
    valid_labels = all_labels[valid_mask]
    valid_probs = all_probs[valid_mask]
    valid_masks = all_masks[valid_mask]

    # Per-label coverage
    per_label_coverage = valid_masks.sum(axis=0)
    valid_cols = per_label_coverage > 0

    if valid_cols.sum() == 0:
        metrics["micro_f1"] = float("nan")
        metrics["macro_f1"] = float("nan")
        return metrics

    filtered_labels = valid_labels[:, valid_cols]
    filtered_probs = valid_probs[:, valid_cols]

    # Search threshold or use fixed
    best_threshold = threshold
    if search_threshold:
        best_f1 = -1.0
        for thr in np.arange(0.2, 0.85, 0.05):
            pred_bin = (filtered_probs >= thr).astype(np.int32)
            micro = float(F.binary_cross_entropy_with_logits(
                torch.tensor(filtered_probs),
                torch.tensor(filtered_labels),
                reduction="mean",
            ).item())
            f1_res = multilabel_f1_score(filtered_labels, pred_bin)
            if f1_res["micro"] > best_f1:
                best_f1 = f1_res["micro"]
                best_threshold = thr

    pred_bin = (filtered_probs >= best_threshold).astype(np.int32)
    f1_res = multilabel_f1_score(filtered_labels, pred_bin)
    metrics["micro_f1"] = f1_res["micro"]
    metrics["macro_f1"] = f1_res["macro"]
    metrics["threshold"] = float(best_threshold)

    if label_names:
        for i, name in enumerate(label_names):
            if i in f1_res["per_label"]:
                metrics[f"f1_{name}"] = f1_res["per_label"][i]

    return metrics


def split_df(df, valid_size, seed, label_col):
    # Try to create a rough stratification based on label presence
    df = df.copy()
    has_label = df[label_col].fillna("").apply(
        lambda x: "present" if (str(x).strip() and str(x).strip().lower() not in {"nan", "none", "null"}) else "none"
    )
    stratify = has_label if has_label.nunique() > 1 else None
    try:
        return train_test_split(df, test_size=valid_size, random_state=seed,
                                shuffle=True, stratify=stratify)
    except Exception:
        return train_test_split(df, test_size=valid_size, random_state=seed,
                                shuffle=True, stratify=None)


def fmt_float(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "nan"
    except Exception:
        pass
    return f"{float(v):.4f}"


def build_label_map_from_df(df, label_col, sep=";"):
    items = []
    for _, row in df.iterrows():
        tokens = parse_multilabel_cell(row.get(label_col, ""), sep=sep)
        if tokens and not is_explicit_none_label(tokens):
            items.extend(tokens)
    return build_label_map(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with sequence/labels")
    ap.add_argument("--outdir", default="./outputs/ligase_subcellular_v1")
    ap.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--label-col", default="subcellular_labels")
    ap.add_argument("--sep", default=";")
    ap.add_argument("--valid-size", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    ap.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    ap.set_defaults(freeze_backbone=True)
    ap.add_argument("--unfreeze-last-n-layers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--search-thresholds", dest="search_thresholds", action="store_true")
    ap.add_argument("--no-search-thresholds", dest="search_thresholds", action="store_false")
    ap.set_defaults(search_thresholds=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.seq_col not in df.columns:
        raise KeyError(f"Sequence column not found: {args.seq_col}")
    if args.label_col not in df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")

    df = df.copy()
    df[args.seq_col] = df[args.seq_col].astype(str).map(clean_sequence)
    df = df[df[args.seq_col].str.len() > 0].reset_index(drop=True)
    print(f"[Info] Valid sequences: {len(df)}")

    label_map = build_label_map_from_df(df, args.label_col, args.sep)
    if len(label_map) == 0:
        raise ValueError("No valid subcellular labels found.")
    id2label = [x for x, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    print(f"[Info] Subcellular classes ({len(label_map)}): {id2label}")

    train_df, val_df = split_df(df, args.valid_size, args.seed, args.label_col)
    print(f"[Info] train={len(train_df)}, val={len(val_df)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = SubcellularDataset(
        train_df, label_map,
        seq_col=args.seq_col, label_col=args.label_col, sep=args.sep,
    )
    val_ds = SubcellularDataset(
        val_df, label_map,
        seq_col=args.seq_col, label_col=args.label_col, sep=args.sep,
    )

    collate_fn = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )

    device = get_device(args.device)
    pos_weight = compute_pos_weight(
        np.stack([s["label_vec"] for s in train_ds.samples], axis=0)
    )
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
    print(f"[Info] pos_weight min={pos_weight.min():.2f} max={pos_weight.max():.2f}")

    model = LigaseSubcellularModel(
        model_name=args.model_name,
        num_subcellular=len(label_map),
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        hidden_dim=args.hidden_dim,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_micro_f1 = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device,
                                   pos_weight=pos_weight_tensor)
        val_metrics = evaluate(
            model, val_loader, device,
            pos_weight=pos_weight_tensor,
            threshold=args.threshold,
            search_threshold=args.search_thresholds,
            label_names=id2label,
        )

        log = {
            "epoch": epoch,
            "train_loss": tr_loss,
            **val_metrics,
        }
        history.append(log)
        print(
            f"[Epoch {epoch:2d}] "
            f"tr_loss={tr_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"micro_f1={val_metrics.get('micro_f1', float('nan')):.4f} "
            f"macro_f1={val_metrics.get('macro_f1', float('nan')):.4f}"
        )

        val_loss = val_metrics["loss"]
        micro_f1 = val_metrics.get("micro_f1", float("nan"))

        # Early stopping on val_loss
        improved = False
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            if not np.isnan(micro_f1) and micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
            best_epoch = epoch
            improved = True
            bad_epochs = 0

            # Save best checkpoint
            found_thr = val_metrics.get("threshold", args.threshold)
            ckpt = {
                "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "config": {
                    "model_name": args.model_name,
                    "num_subcellular": len(label_map),
                    "max_length": args.max_length,
                    "dropout": args.dropout,
                    "hidden_dim": args.hidden_dim,
                    "freeze_backbone": args.freeze_backbone,
                    "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
                    "threshold": found_thr,
                },
                "label_maps": {
                    "subcellular_to_idx": label_map,
                },
                "best_metrics": val_metrics,
                "best_epoch": epoch,
            }
            torch.save(ckpt, outdir / "best_ligase_subcellular.pt")

            # Save train history
            pd.DataFrame(history).to_csv(outdir / "train_history.csv", index=False)

            # Save label schema
            with open(outdir / "label_schema.json", "w", encoding="utf-8") as f:
                json.dump({
                    "num_classes": len(label_map),
                    "labels": id2label,
                    "label_map": label_map,
                }, f, ensure_ascii=False, indent=2)

            # Save config
            with open(outdir / "config.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"[Info] Early stopping at epoch {epoch}")
            break

    print(f"\n[Done] Best epoch={best_epoch}, val_loss={best_val_loss:.4f}, "
          f"micro_f1={best_micro_f1:.4f}")
    print(f"Checkpoint saved to {outdir / 'best_ligase_subcellular.pt'}")


if __name__ == "__main__":
    main()
