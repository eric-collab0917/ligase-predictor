#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visual evaluation for ligase subcellular localization.

Outputs:
- overall_metrics.json
- per_label_metrics.csv
- val_predictions_subcellular.csv
- fig_label_frequency.png
- fig_training_curves.png
- fig_per_label_f1_support.png
- fig_ovr_confusion_grid.png
- fig_precision_recall_curves.png
- fig_threshold_sweep.png
- fig_label_cooccurrence.png
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
try:
    import matplotlib.pyplot as plt  # noqa: E402
except Exception:
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ligase_subcellular import (  # noqa: E402
    LigaseSubcellularModel,
    clean_sequence,
    get_device,
    is_explicit_none_label,
    parse_multilabel_cell,
)


BG = "#f8fafc"
TEXT = "#0f172a"
GRID = "#cbd5e1"
TEAL = "#0f766e"
TEAL_LIGHT = "#99f6e4"
BLUE = "#2563eb"
ORANGE = "#ea580c"
ROSE = "#e11d48"
SLATE = "#475569"


def load_json(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_df(df, valid_size, seed, label_col):
    df = df.copy()
    has_label = df[label_col].fillna("").apply(
        lambda x: "present"
        if (str(x).strip() and str(x).strip().lower() not in {"nan", "none", "null"})
        else "none"
    )
    stratify = has_label if has_label.nunique() > 1 else None
    try:
        return train_test_split(
            df,
            test_size=valid_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except Exception:
        return train_test_split(
            df,
            test_size=valid_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )


def slugify(text):
    out = []
    for ch in str(text).lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def safe_float(v):
    try:
        if v is None or np.isnan(v):
            return None
    except Exception:
        pass
    return float(v)


def ensure_pyplot():
    if plt is None:
        raise RuntimeError("matplotlib is not available in the current Python environment.")


def parse_target_and_mask(value, label_map, sep=";"):
    tokens = parse_multilabel_cell(value, sep=sep)
    vec = np.zeros(len(label_map), dtype=np.int32)
    mask = np.zeros(len(label_map), dtype=np.int32)

    if not tokens:
        return vec, mask, False

    if is_explicit_none_label(tokens):
        mask[:] = 1
        return vec, mask, True

    mapped = 0
    for token in tokens:
        if token in label_map:
            vec[label_map[token]] = 1
            mapped += 1

    if mapped == 0:
        return vec, mask, False

    mask[:] = 1
    return vec, mask, True


class EvalDataset(Dataset):
    def __init__(self, df, label_map, seq_col, label_col, id_col, sep):
        self.samples = []
        for row_idx, row in df.reset_index(drop=False).iterrows():
            seq = clean_sequence(row[seq_col])
            if not seq:
                continue
            label_vec, label_mask, has_labels = parse_target_and_mask(
                row.get(label_col, ""), label_map=label_map, sep=sep
            )
            self.samples.append(
                {
                    "row_idx": int(row["index"]),
                    "id": str(row.get(id_col, f"row_{row_idx}")),
                    "sequence": seq,
                    "sequence_length": len(seq),
                    "label_vec": label_vec.astype(np.float32),
                    "label_mask": label_mask.astype(np.float32),
                    "has_labels": bool(has_labels),
                }
            )

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
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label_vec": torch.tensor(
                np.stack([x["label_vec"] for x in batch], axis=0), dtype=torch.float32
            ),
            "label_mask": torch.tensor(
                np.stack([x["label_mask"] for x in batch], axis=0), dtype=torch.float32
            ),
            "meta": batch,
        }

    return _collate


@torch.no_grad()
def infer_predictions(model, loader, device):
    all_probs = []
    all_true = []
    all_masks = []
    records = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_true.append(batch["label_vec"].cpu().numpy())
        all_masks.append(batch["label_mask"].cpu().numpy())
        records.extend(batch["meta"])

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_true, axis=0),
        np.concatenate(all_masks, axis=0),
        records,
    )


def plot_label_frequency(df, label_map, label_col, sep, out_png):
    ensure_pyplot()
    id2label = [x for x, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    counts = {label: 0 for label in id2label}

    for raw in df[label_col].fillna(""):
        tokens = parse_multilabel_cell(raw, sep=sep)
        if not tokens or is_explicit_none_label(tokens):
            continue
        for token in tokens:
            if token in counts:
                counts[token] += 1

    freq_df = pd.DataFrame(
        {"label": list(counts.keys()), "count": list(counts.values())}
    ).sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    bars = ax.barh(freq_df["label"], freq_df["count"], color=TEAL, alpha=0.9)
    ax.set_title("Subcellular Label Frequency", fontsize=15, fontweight="bold", color=TEXT)
    ax.set_xlabel("Positive samples", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID)

    x_max = max(freq_df["count"].max(), 1)
    for bar, count in zip(bars, freq_df["count"]):
        ax.text(
            bar.get_width() + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)}",
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history_path, out_png):
    ensure_pyplot()
    hist = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    fig.patch.set_facecolor(BG)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID)
        ax.tick_params(colors=TEXT)

    axes[0].plot(hist["epoch"], hist["train_loss"], color=TEAL, lw=2.2, label="Train loss")
    axes[0].plot(hist["epoch"], hist["loss"], color=ORANGE, lw=2.2, label="Val loss")
    axes[0].set_title("Loss", fontsize=13, fontweight="bold", color=TEXT)
    axes[0].set_xlabel("Epoch", color=TEXT)
    axes[0].legend(frameon=False)

    axes[1].plot(hist["epoch"], hist["micro_f1"], color=BLUE, lw=2.2, label="Micro F1")
    axes[1].plot(hist["epoch"], hist["macro_f1"], color=ROSE, lw=2.2, label="Macro F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation F1", fontsize=13, fontweight="bold", color=TEXT)
    axes[1].set_xlabel("Epoch", color=TEXT)
    axes[1].legend(frameon=False)

    axes[2].plot(hist["epoch"], hist["threshold"], color=SLATE, lw=2.2)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Best threshold by epoch", fontsize=13, fontweight="bold", color=TEXT)
    axes[2].set_xlabel("Epoch", color=TEXT)

    fig.suptitle("Training Dynamics", fontsize=15, fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.int32)
    metrics = {
        "threshold": float(threshold),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_ratio": float((y_true == y_pred).all(axis=1).mean()),
        "label_cardinality_true": float(y_true.sum(axis=1).mean()),
        "label_cardinality_pred": float(y_pred.sum(axis=1).mean()),
    }
    jaccard = []
    for i in range(y_true.shape[0]):
        inter = int(np.logical_and(y_true[i] == 1, y_pred[i] == 1).sum())
        union = int(np.logical_or(y_true[i] == 1, y_pred[i] == 1).sum())
        jaccard.append(1.0 if union == 0 else inter / union)
    metrics["mean_jaccard"] = float(np.mean(jaccard))
    return metrics, y_pred


def build_per_label_metrics(y_true, y_prob, y_pred, id2label):
    rows = []
    for idx, label in enumerate(id2label):
        yt = y_true[:, idx].astype(np.int32)
        yp = y_pred[:, idx].astype(np.int32)
        pr = float(precision_score(yt, yp, zero_division=0))
        rc = float(recall_score(yt, yp, zero_division=0))
        f1 = float(f1_score(yt, yp, zero_division=0))

        try:
            ap = float(average_precision_score(yt, y_prob[:, idx]))
        except Exception:
            ap = float("nan")
        try:
            auc = float(roc_auc_score(yt, y_prob[:, idx]))
        except Exception:
            auc = float("nan")

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append(
            {
                "label": label,
                "support": int(yt.sum()),
                "predicted_positive": int(yp.sum()),
                "precision": pr,
                "recall": rc,
                "f1": f1,
                "average_precision": ap,
                "roc_auc": auc,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def plot_per_label_f1_support(per_label_df, out_png):
    ensure_pyplot()
    d = per_label_df.sort_values(["f1", "support"], ascending=[True, True]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor(BG)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID)
        ax.tick_params(colors=TEXT)

    axes[0].barh(d["label"], d["f1"], color=BLUE, alpha=0.9)
    axes[0].set_xlim(0, 1.0)
    axes[0].set_title("Per-label F1", fontsize=13, fontweight="bold", color=TEXT)
    axes[0].set_xlabel("F1", color=TEXT)
    for i, val in enumerate(d["f1"]):
        axes[0].text(min(val + 0.02, 0.98), i, f"{val:.2f}", va="center", ha="left", fontsize=9, color=TEXT)

    axes[1].barh(d["label"], d["support"], color=TEAL, alpha=0.9)
    axes[1].set_title("Validation support", fontsize=13, fontweight="bold", color=TEXT)
    axes[1].set_xlabel("Positive samples", color=TEXT)
    for i, val in enumerate(d["support"]):
        axes[1].text(val + max(d["support"].max(), 1) * 0.01, i, str(int(val)), va="center", ha="left", fontsize=9, color=TEXT)

    fig.suptitle("Per-label Performance and Class Balance", fontsize=15, fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_ovr_confusion_grid(per_label_df, out_png):
    ensure_pyplot()
    n = len(per_label_df)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 3.6 * nrows))
    fig.patch.set_facecolor(BG)
    axes = np.array(axes).reshape(nrows, ncols)

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, (_, row) in zip(axes.flat, per_label_df.iterrows()):
        ax.set_visible(True)
        ax.set_facecolor(BG)
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8, color=TEXT)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=8, color=TEXT)
        ax.set_title(f"{row['label']}\nF1={row['f1']:.2f}", fontsize=10, fontweight="bold", color=TEXT)
        vmax = cm.max() if cm.size else 0.0
        for i in range(2):
            for j in range(2):
                color = "white" if vmax > 0 and cm[i, j] > 0.55 * vmax else TEXT
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    fig.suptitle("One-vs-rest Confusion Matrices", fontsize=15, fontweight="bold", color=TEXT, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curves(y_true, y_prob, id2label, out_png):
    ensure_pyplot()
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap = plt.cm.get_cmap("tab20", len(id2label))
    for idx, label in enumerate(id2label):
        yt = y_true[:, idx].astype(np.int32)
        if yt.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(yt, y_prob[:, idx])
        ap = average_precision_score(yt, y_prob[:, idx])
        ax.plot(recall, precision, lw=2.0, color=cmap(idx), label=f"{label} (AP={ap:.2f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", color=TEXT)
    ax.set_ylabel("Precision", color=TEXT)
    ax.set_title("Precision-Recall Curves by Label", fontsize=15, fontweight="bold", color=TEXT)
    ax.grid(linestyle="--", alpha=0.35, color=GRID)
    ax.tick_params(colors=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(y_true, y_prob, out_png):
    ensure_pyplot()
    sweep = build_threshold_sweep_df(y_true, y_prob)
    best_idx = int(sweep["micro_f1"].idxmax())

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID)
        ax.tick_params(colors=TEXT)

    axes[0].plot(sweep["threshold"], sweep["micro_f1"], color=BLUE, lw=2.2, label="Micro F1")
    axes[0].plot(sweep["threshold"], sweep["macro_f1"], color=ROSE, lw=2.2, label="Macro F1")
    axes[0].plot(sweep["threshold"], sweep["samples_f1"], color=TEAL, lw=2.2, label="Samples F1")
    axes[0].axvline(sweep.loc[best_idx, "threshold"], color=SLATE, linestyle="--", lw=1.6)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xlabel("Global threshold", color=TEXT)
    axes[0].set_title("F1 vs threshold", fontsize=13, fontweight="bold", color=TEXT)
    axes[0].legend(frameon=False)

    axes[1].plot(sweep["threshold"], sweep["mean_pred_labels"], color=ORANGE, lw=2.2)
    axes[1].set_xlabel("Global threshold", color=TEXT)
    axes[1].set_title("Predicted labels per sample", fontsize=13, fontweight="bold", color=TEXT)

    fig.suptitle("Threshold Sweep", fontsize=15, fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def cooccurrence_matrix(y_bin):
    return y_bin.T @ y_bin


def plot_label_cooccurrence(y_true, y_pred, id2label, out_png):
    ensure_pyplot()
    true_co = cooccurrence_matrix(y_true.astype(np.int32))
    pred_co = cooccurrence_matrix(y_pred.astype(np.int32))
    delta = pred_co - true_co

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8))
    fig.patch.set_facecolor(BG)
    mats = [
        ("True co-occurrence", true_co, "YlGnBu"),
        ("Predicted co-occurrence", pred_co, "YlGnBu"),
        ("Prediction minus truth", delta, "coolwarm"),
    ]

    for ax, (title, mat, cmap) in zip(axes, mats):
        ax.set_facecolor(BG)
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(id2label)))
        ax.set_yticks(np.arange(len(id2label)))
        ax.set_xticklabels(id2label, rotation=50, ha="right", fontsize=8, color=TEXT)
        ax.set_yticklabels(id2label, fontsize=8, color=TEXT)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle("Label Co-occurrence Structure", fontsize=15, fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_prediction_table(records, y_true, y_prob, y_pred, id2label):
    rows = []
    slugs = [slugify(label) for label in id2label]
    for i, meta in enumerate(records):
        true_labels = [id2label[j] for j in range(len(id2label)) if y_true[i, j] == 1]
        pred_labels = [id2label[j] for j in range(len(id2label)) if y_pred[i, j] == 1]
        inter = int(np.logical_and(y_true[i] == 1, y_pred[i] == 1).sum())
        union = int(np.logical_or(y_true[i] == 1, y_pred[i] == 1).sum())
        row = {
            "id": meta["id"],
            "row_idx": int(meta["row_idx"]),
            "sequence_length": int(meta["sequence_length"]),
            "true_labels": ";".join(true_labels),
            "pred_labels": ";".join(pred_labels),
            "num_true_labels": int(y_true[i].sum()),
            "num_pred_labels": int(y_pred[i].sum()),
            "exact_match": int((y_true[i] == y_pred[i]).all()),
            "jaccard": 1.0 if union == 0 else inter / union,
        }
        for j, slug in enumerate(slugs):
            row[f"true__{slug}"] = int(y_true[i, j])
            row[f"pred__{slug}"] = int(y_pred[i, j])
            row[f"prob__{slug}"] = float(y_prob[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_sweep_df(y_true, y_prob):
    thresholds = np.arange(0.1, 0.91, 0.05)
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int32)
        rows.append(
            {
                "threshold": float(thr),
                "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
                "mean_pred_labels": float(y_pred.sum(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows)


def maybe_run_r_plotting(plot_script, csv_path, history_path, per_label_path, pred_path, sweep_path, outdir):
    rscript = shutil.which("Rscript")
    if not rscript:
        print("[Warn] Rscript not found; skipping plot generation.")
        return False
    cmd = [
        rscript,
        str(plot_script),
        "--csv",
        str(csv_path),
        "--train-history",
        str(history_path),
        "--per-label",
        str(per_label_path),
        "--predictions",
        str(pred_path),
        "--threshold-sweep",
        str(sweep_path),
        "--outdir",
        str(outdir),
    ]
    subprocess.run(cmd, check=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/ligase_subcellular_auto.csv")
    ap.add_argument("--checkpoint", default="outputs/ligase_subcellular_v1/best_ligase_subcellular.pt")
    ap.add_argument("--config", default="outputs/ligase_subcellular_v1/config.json")
    ap.add_argument("--train-history", default="outputs/ligase_subcellular_v1/train_history.csv")
    ap.add_argument("--outdir", default="reports/subcellular_eval")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--seq-col", default="")
    ap.add_argument("--label-col", default="")
    ap.add_argument("--sep", default="")
    ap.add_argument("--valid-size", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--plot-backend", choices=["auto", "python", "r", "none"], default="auto")
    ap.add_argument("--plot-script", default="scripts/plot_ligase_subcellular_visuals.R")
    args = ap.parse_args()

    cfg = load_json(args.config)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    seq_col = args.seq_col or cfg.get("seq_col", "sequence")
    label_col = args.label_col or cfg.get("label_col", "subcellular_labels")
    sep = args.sep or cfg.get("sep", ";")
    valid_size = float(args.valid_size if args.valid_size is not None else cfg.get("valid_size", 0.2))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    label_map = ckpt["label_maps"]["subcellular_to_idx"]
    id2label = [x for x, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    model_cfg = ckpt.get("config", {})
    threshold = float(
        args.threshold
        if args.threshold is not None
        else model_cfg.get("threshold", cfg.get("threshold", 0.5))
    )
    max_length = int(args.max_length if args.max_length is not None else model_cfg.get("max_length", 512))
    model_name = model_cfg.get("model_name", cfg.get("model_name", "facebook/esm2_t6_8M_UR50D"))

    df = pd.read_csv(args.csv)
    if seq_col not in df.columns:
        raise KeyError(f"Sequence column not found: {seq_col}")
    if label_col not in df.columns:
        raise KeyError(f"Label column not found: {label_col}")
    if args.id_col not in df.columns:
        df[args.id_col] = [f"row_{i}" for i in range(len(df))]

    df = df.copy()
    df[seq_col] = df[seq_col].astype(str).map(clean_sequence)
    df = df[df[seq_col].str.len() > 0].reset_index(drop=True)
    train_df, val_df = split_df(df, valid_size=valid_size, seed=seed, label_col=label_col)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = EvalDataset(
        df=val_df,
        label_map=label_map,
        seq_col=seq_col,
        label_col=label_col,
        id_col=args.id_col,
        sep=sep,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate(tokenizer, max_length=max_length),
    )

    device = get_device(args.device)
    model = LigaseSubcellularModel(
        model_name=model_name,
        num_subcellular=len(id2label),
        dropout=float(model_cfg.get("dropout", 0.2)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        unfreeze_last_n_layers=int(model_cfg.get("unfreeze_last_n_layers", 0)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    all_prob, all_true, all_mask, records = infer_predictions(model, loader, device)
    valid_rows = all_mask.sum(axis=1) > 0
    y_true = all_true[valid_rows].astype(np.int32)
    y_prob = all_prob[valid_rows].astype(np.float32)
    valid_records = [records[i] for i in range(len(records)) if valid_rows[i]]

    overall_metrics, y_pred = compute_metrics(y_true, y_prob, threshold=threshold)
    overall_metrics.update(
        {
            "n_total_rows": int(len(df)),
            "n_validation_rows": int(len(val_df)),
            "n_validation_sequences_used": int(len(dataset)),
            "n_validation_labeled_rows": int(valid_rows.sum()),
            "num_labels": int(len(id2label)),
            "labels": id2label,
        }
    )

    per_label_df = build_per_label_metrics(y_true, y_prob, y_pred, id2label)
    pred_df = build_prediction_table(valid_records, y_true, y_prob, y_pred, id2label)

    with open(outdir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, ensure_ascii=False, indent=2)
    per_label_df.to_csv(outdir / "per_label_metrics.csv", index=False)
    pred_df.to_csv(outdir / "val_predictions_subcellular.csv", index=False)
    sweep_df = build_threshold_sweep_df(y_true, y_prob)
    sweep_df.to_csv(outdir / "threshold_sweep.csv", index=False)

    plot_backend = args.plot_backend
    if plot_backend == "auto":
        plot_backend = "python" if plt is not None else "r"

    if plot_backend == "python":
        plot_label_frequency(df=df, label_map=label_map, label_col=label_col, sep=sep, out_png=outdir / "fig_label_frequency.png")
        plot_training_curves(history_path=args.train_history, out_png=outdir / "fig_training_curves.png")
        plot_per_label_f1_support(per_label_df=per_label_df, out_png=outdir / "fig_per_label_f1_support.png")
        plot_ovr_confusion_grid(per_label_df=per_label_df, out_png=outdir / "fig_ovr_confusion_grid.png")
        plot_precision_recall_curves(y_true=y_true, y_prob=y_prob, id2label=id2label, out_png=outdir / "fig_precision_recall_curves.png")
        plot_threshold_sweep(y_true=y_true, y_prob=y_prob, out_png=outdir / "fig_threshold_sweep.png")
        plot_label_cooccurrence(y_true=y_true, y_pred=y_pred, id2label=id2label, out_png=outdir / "fig_label_cooccurrence.png")
    elif plot_backend == "r":
        maybe_run_r_plotting(
            plot_script=Path(args.plot_script).resolve(),
            csv_path=Path(args.csv).resolve(),
            history_path=Path(args.train_history).resolve(),
            per_label_path=outdir / "per_label_metrics.csv",
            pred_path=outdir / "val_predictions_subcellular.csv",
            sweep_path=outdir / "threshold_sweep.csv",
            outdir=outdir,
        )

    print(f"[Done] Visual evaluation saved to: {outdir}")
    print(f"[Info] Validation labeled rows used: {valid_rows.sum()}")
    print(f"[Info] threshold={threshold:.2f} micro_f1={overall_metrics['micro_f1']:.4f} macro_f1={overall_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
