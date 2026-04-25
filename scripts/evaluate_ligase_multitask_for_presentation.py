#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Presentation-ready evaluation for ligase multitask model.

Outputs:
- overall_metrics.json / overall_metrics.csv
- ec_classification_report.csv
- ec_stratified_report.csv
- ec_confusion_counts.csv
- ec_confusion_row_norm.csv
- fig_ec_confusion_counts.png
- fig_ec_confusion_row_norm.png
- fig_ec_per_class_f1.png
- presentation_notes.md
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ligase_multitask import (
    LigaseMultiTaskModel,
    clean_sequence,
    get_device,
    is_explicit_none_label,
    multilabel_micro_f1,
    parse_multilabel_cell,
    unpack_multitask_outputs,
)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def split_df(df, valid_size, seed, ec_col):
    ec_vals = df[ec_col].fillna("__MISSING__").astype(str)
    class_counts = ec_vals.value_counts()
    can_stratify = bool((class_counts >= 2).all()) and class_counts.shape[0] > 1
    stratify = ec_vals if can_stratify else None
    try:
        return train_test_split(df, test_size=valid_size, random_state=seed, shuffle=True, stratify=stratify)
    except Exception:
        return train_test_split(df, test_size=valid_size, random_state=seed, shuffle=True, stratify=None)


def parse_multilabel_target(raw, label_map, sep=";"):
    tokens = parse_multilabel_cell(raw, sep=sep)
    if len(tokens) == 0:
        return None
    vec = np.zeros(len(label_map), dtype=np.int32)
    if is_explicit_none_label(tokens):
        return vec
    mapped = 0
    for t in tokens:
        if t in label_map:
            vec[label_map[t]] = 1
            mapped += 1
    if mapped == 0:
        return None
    return vec


def parse_ec_target(raw, ec_map):
    v = str(raw).strip()
    if not v or v.lower() in {"nan", "none", "null"}:
        return None
    if v not in ec_map:
        return None
    return int(ec_map[v])


def pretty_metric(v, digits=4):
    try:
        if np.isnan(v):
            return "nan"
    except Exception:
        pass
    return f"{float(v):.{digits}f}"


def apply_ec_logit_adjustment(ec_logits, ec_log_prior=None, tau=0.0):
    if ec_log_prior is None or float(tau) <= 0.0:
        return ec_logits
    lp = torch.tensor(ec_log_prior, dtype=ec_logits.dtype, device=ec_logits.device).view(1, -1)
    return ec_logits - float(tau) * lp


def plot_confusion(cm, labels, out_png, title, normalize=False):
    n = len(labels)
    size = max(8.0, min(20.0, 0.75 * n + 4.0))
    fig, ax = plt.subplots(figsize=(size, size * 0.86))

    mat = cm.astype(np.float64)
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)

    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted EC subclass", fontsize=11)
    ax.set_ylabel("True EC subclass", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    show_text = n <= 18
    if show_text:
        vmax = mat.max() if mat.size else 0.0
        for i in range(n):
            for j in range(n):
                txt = f"{mat[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                color = "white" if mat[i, j] > (0.55 * vmax if vmax > 0 else 0) else "#1f2937"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(class_report_df, out_png):
    d = class_report_df.copy()
    d = d.sort_values("f1-score", ascending=False)
    labels = d["ec_subclass"].tolist()
    f1s = d["f1-score"].astype(float).values

    w = max(9.0, min(20.0, 0.45 * len(labels) + 5.0))
    fig, ax = plt.subplots(figsize=(w, 5.3))
    bars = ax.bar(np.arange(len(labels)), f1s, color="#0ea5a6", edgecolor="#0f766e", alpha=0.9)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_title("EC Subclass Per-class F1 (Validation)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    if len(labels) <= 24:
        for b, v in zip(bars, f1s):
            ax.text(b.get_x() + b.get_width() / 2, min(0.98, v + 0.015), f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def infer_eval_rows(
    model,
    tokenizer,
    df,
    device,
    max_length,
    ec_to_idx,
    sub_to_idx,
    metal_to_idx,
    seq_col,
    ec_col,
    substrate_col,
    metal_col,
    sep,
    batch_size,
    substrate_threshold,
    metal_threshold,
    metal_presence_threshold,
    substrate_thresholds=None,
    metal_thresholds=None,
    ec_log_prior=None,
    ec_logit_adjust_tau=0.0,
):
    id2ec = [x for x, _ in sorted(ec_to_idx.items(), key=lambda kv: kv[1])]
    n = len(df)

    rows = []
    indices = list(range(n))
    for i in range(0, n, batch_size):
        idx_batch = indices[i : i + batch_size]
        seqs = [df.iloc[j][seq_col] for j in idx_batch]

        enc = tokenizer(seqs, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            ec_logits, sub_logits, metal_logits, metal_presence_logits = unpack_multitask_outputs(outputs)
            ec_logits = apply_ec_logit_adjustment(ec_logits, ec_log_prior=ec_log_prior, tau=ec_logit_adjust_tau)

        ec_prob = torch.softmax(ec_logits, dim=1).detach().cpu().numpy()
        sub_prob = torch.sigmoid(sub_logits).detach().cpu().numpy()
        metal_prob = torch.sigmoid(metal_logits).detach().cpu().numpy()
        metal_presence_prob = None
        if metal_presence_logits is not None:
            metal_presence_prob = torch.sigmoid(metal_presence_logits).detach().cpu().numpy()

        for bi, row_idx in enumerate(idx_batch):
            src = df.iloc[row_idx]

            true_ec = parse_ec_target(src.get(ec_col, ""), ec_to_idx)
            true_sub = parse_multilabel_target(src.get(substrate_col, ""), sub_to_idx, sep=sep)
            true_metal = parse_multilabel_target(src.get(metal_col, ""), metal_to_idx, sep=sep)

            ec_pred_idx = int(np.argmax(ec_prob[bi]))
            ec_pred_label = id2ec[ec_pred_idx]
            ec_pred_prob = float(ec_prob[bi, ec_pred_idx])

            if len(sub_to_idx) > 0:
                if substrate_thresholds is not None and len(substrate_thresholds) == len(sub_to_idx):
                    sub_pred = (sub_prob[bi] >= np.asarray(substrate_thresholds, dtype=np.float32)).astype(np.int32)
                else:
                    sub_pred = (sub_prob[bi] >= substrate_threshold).astype(np.int32)
            else:
                sub_pred = np.zeros((0,), dtype=np.int32)

            if len(metal_to_idx) > 0:
                if metal_presence_prob is not None and float(metal_presence_prob[bi]) < metal_presence_threshold:
                    metal_pred = np.zeros(len(metal_to_idx), dtype=np.int32)
                else:
                    if metal_thresholds is not None and len(metal_thresholds) == len(metal_to_idx):
                        metal_pred = (metal_prob[bi] >= np.asarray(metal_thresholds, dtype=np.float32)).astype(np.int32)
                    else:
                        metal_pred = (metal_prob[bi] >= metal_threshold).astype(np.int32)
            else:
                metal_pred = np.zeros((0,), dtype=np.int32)

            rows.append(
                {
                    "row_idx": int(row_idx),
                    "ec_true_idx": true_ec,
                    "ec_pred_idx": ec_pred_idx,
                    "ec_pred_label": ec_pred_label,
                    "ec_pred_prob": ec_pred_prob,
                    "sub_true": true_sub,
                    "sub_pred": sub_pred,
                    "metal_true": true_metal,
                    "metal_pred": metal_pred,
                    "metal_presence_prob": float(metal_presence_prob[bi]) if metal_presence_prob is not None else float("nan"),
                }
            )

    return rows


def compute_overall_metrics(rows, id2ec, metal_two_stage=False, metal_presence_threshold=0.5):
    ec_true = [r["ec_true_idx"] for r in rows if r["ec_true_idx"] is not None]
    ec_pred = [r["ec_pred_idx"] for r in rows if r["ec_true_idx"] is not None]

    sub_true = [r["sub_true"] for r in rows if r["sub_true"] is not None]
    sub_pred = [r["sub_pred"] for r in rows if r["sub_true"] is not None]

    metal_true = [r["metal_true"] for r in rows if r["metal_true"] is not None]
    metal_pred = [r["metal_pred"] for r in rows if r["metal_true"] is not None]

    metrics = {
        "n_total": int(len(rows)),
        "n_ec_labeled": int(len(ec_true)),
        "n_substrate_labeled": int(len(sub_true)),
        "n_metal_labeled": int(len(metal_true)),
    }

    if len(ec_true) > 0:
        yt = np.asarray(ec_true, dtype=np.int32)
        yp = np.asarray(ec_pred, dtype=np.int32)
        metrics["ec_acc"] = float((yt == yp).mean())
        metrics["ec_macro_f1"] = float(f1_score(yt, yp, average="macro", zero_division=0))
        metrics["ec_weighted_f1"] = float(f1_score(yt, yp, average="weighted", zero_division=0))
    else:
        metrics["ec_acc"] = float("nan")
        metrics["ec_macro_f1"] = float("nan")
        metrics["ec_weighted_f1"] = float("nan")

    if len(sub_true) > 0:
        yt = np.stack(sub_true, axis=0).astype(np.int32)
        yp = np.stack(sub_pred, axis=0).astype(np.int32)
        metrics["substrate_micro_f1"] = float(multilabel_micro_f1(yt, yp))
    else:
        metrics["substrate_micro_f1"] = float("nan")

    if len(metal_true) > 0:
        yt = np.stack(metal_true, axis=0).astype(np.int32)
        yp = np.stack(metal_pred, axis=0).astype(np.int32)
        metrics["metal_micro_f1"] = float(multilabel_micro_f1(yt, yp))

        if metal_two_stage:
            presence_true = (yt.sum(axis=1) > 0).astype(np.int32)
            presence_pred = (np.asarray([r["metal_presence_prob"] for r in rows if r["metal_true"] is not None]) >= metal_presence_threshold).astype(np.int32)
            metrics["metal_presence_acc"] = float((presence_true == presence_pred).mean())
        else:
            metrics["metal_presence_acc"] = float("nan")
    else:
        metrics["metal_micro_f1"] = float("nan")
        metrics["metal_presence_acc"] = float("nan")

    return metrics


def build_ec_reports(rows, id2ec):
    valid = [r for r in rows if r["ec_true_idx"] is not None]
    if len(valid) == 0:
        return None, None, None, None

    y_true = np.asarray([r["ec_true_idx"] for r in valid], dtype=np.int32)
    y_pred = np.asarray([r["ec_pred_idx"] for r in valid], dtype=np.int32)

    labels_idx = list(range(len(id2ec)))
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels_idx)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=id2ec,
        output_dict=True,
        zero_division=0,
    )

    cls_rows = []
    for name in id2ec:
        rec = report_dict.get(name, {})
        cls_rows.append(
            {
                "ec_subclass": name,
                "precision": float(rec.get("precision", 0.0)),
                "recall": float(rec.get("recall", 0.0)),
                "f1-score": float(rec.get("f1-score", 0.0)),
                "support": int(rec.get("support", 0)),
            }
        )
    cls_df = pd.DataFrame(cls_rows)

    strat_rows = []
    for c_idx, c_name in enumerate(id2ec):
        class_subset = [r for r in rows if r["ec_true_idx"] == c_idx]
        n_class = len(class_subset)

        tp = int(cm_counts[c_idx, c_idx])
        fn = int(cm_counts[c_idx, :].sum() - tp)
        fp = int(cm_counts[:, c_idx].sum() - tp)

        ec_precision = safe_div(tp, tp + fp)
        ec_recall = safe_div(tp, tp + fn)
        ec_f1 = safe_div(2 * ec_precision * ec_recall, ec_precision + ec_recall) if np.isfinite(ec_precision) and np.isfinite(ec_recall) and (ec_precision + ec_recall) > 0 else float("nan")

        class_sub_true = [r["sub_true"] for r in class_subset if r["sub_true"] is not None]
        class_sub_pred = [r["sub_pred"] for r in class_subset if r["sub_true"] is not None]
        if len(class_sub_true) > 0:
            yt = np.stack(class_sub_true, axis=0).astype(np.int32)
            yp = np.stack(class_sub_pred, axis=0).astype(np.int32)
            sub_f1 = float(multilabel_micro_f1(yt, yp))
        else:
            sub_f1 = float("nan")

        class_metal_true = [r["metal_true"] for r in class_subset if r["metal_true"] is not None]
        class_metal_pred = [r["metal_pred"] for r in class_subset if r["metal_true"] is not None]
        if len(class_metal_true) > 0:
            yt = np.stack(class_metal_true, axis=0).astype(np.int32)
            yp = np.stack(class_metal_pred, axis=0).astype(np.int32)
            metal_f1 = float(multilabel_micro_f1(yt, yp))
        else:
            metal_f1 = float("nan")

        row_counts = cm_counts[c_idx].copy()
        row_counts[c_idx] = 0
        confused_to = ""
        confused_n = 0
        if row_counts.sum() > 0:
            j = int(np.argmax(row_counts))
            confused_to = id2ec[j]
            confused_n = int(row_counts[j])

        strat_rows.append(
            {
                "ec_subclass": c_name,
                "n_samples": n_class,
                "ec_precision": ec_precision,
                "ec_recall": ec_recall,
                "ec_f1": ec_f1,
                "substrate_micro_f1_in_class": sub_f1,
                "metal_micro_f1_in_class": metal_f1,
                "top_confused_as": confused_to,
                "top_confused_count": confused_n,
            }
        )

    strat_df = pd.DataFrame(strat_rows).sort_values(["n_samples", "ec_f1"], ascending=[False, False]).reset_index(drop=True)

    cm_row_norm = cm_counts.astype(np.float64)
    row_sum = cm_row_norm.sum(axis=1, keepdims=True)
    cm_row_norm = np.divide(cm_row_norm, row_sum, out=np.zeros_like(cm_row_norm), where=row_sum > 0)

    return cls_df, strat_df, cm_counts, cm_row_norm


def write_notes(out_md, metrics, thresholds, split_name, selected_n, ec_cls_df):
    best_ec = ec_cls_df.sort_values("f1-score", ascending=False).head(3) if ec_cls_df is not None else pd.DataFrame()
    weak_ec = ec_cls_df.sort_values("f1-score", ascending=True).head(3) if ec_cls_df is not None else pd.DataFrame()

    lines = []
    lines.append("# Ligase Multi-task Evaluation Notes (Classroom)")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 评估集：`{split_name}`，样本数 `{selected_n}`")
    lines.append("")
    lines.append("## 1) Overall Metrics")
    lines.append(f"- EC 准确率：`{pretty_metric(metrics.get('ec_acc'))}`")
    lines.append(f"- EC Macro-F1：`{pretty_metric(metrics.get('ec_macro_f1'))}`")
    lines.append(f"- 底物谱 Micro-F1：`{pretty_metric(metrics.get('substrate_micro_f1'))}`")
    lines.append(f"- 金属依赖 Micro-F1：`{pretty_metric(metrics.get('metal_micro_f1'))}`")
    if np.isfinite(metrics.get("metal_presence_acc", np.nan)):
        lines.append(f"- 金属存在性准确率：`{pretty_metric(metrics.get('metal_presence_acc'))}`")
    lines.append("")
    lines.append("## 2) Decision Thresholds")
    lines.append(f"- substrate threshold: `{thresholds['substrate']:.2f}`")
    lines.append(f"- metal type threshold: `{thresholds['metal_type']:.2f}`")
    lines.append(f"- metal presence threshold: `{thresholds['metal_presence']:.2f}`")
    lines.append("")
    if not best_ec.empty:
        lines.append("## 3) Best EC Subclasses (Top-3 by F1)")
        for _, r in best_ec.iterrows():
            lines.append(
                f"- `{r['ec_subclass']}`: F1=`{r['f1-score']:.3f}`, "
                f"Precision=`{r['precision']:.3f}`, Recall=`{r['recall']:.3f}`, Support=`{int(r['support'])}`"
            )
        lines.append("")
    if not weak_ec.empty:
        lines.append("## 4) Weak EC Subclasses (Bottom-3 by F1)")
        for _, r in weak_ec.iterrows():
            lines.append(
                f"- `{r['ec_subclass']}`: F1=`{r['f1-score']:.3f}`, "
                f"Precision=`{r['precision']:.3f}`, Recall=`{r['recall']:.3f}`, Support=`{int(r['support'])}`"
            )
        lines.append("")

    lines.append("## 5) Presentation Guidance")
    lines.append("- 先展示 `fig_ec_confusion_row_norm.png`，强调哪些 EC 子类彼此最易混淆。")
    lines.append("- 再展示 `ec_stratified_report.csv`，说明不同 EC 子类上底物谱/金属依赖任务的性能差异。")
    lines.append("- 最后展示 `fig_ec_per_class_f1.png`，给出模型下一步改进优先级。")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="best_ligase_multitask.pt")
    ap.add_argument("--csv", required=True, help="training-ready CSV with sequence + labels")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--split", choices=["valid", "full"], default="valid")
    ap.add_argument("--valid-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--seq-col", default="")
    ap.add_argument("--ec-col", default="")
    ap.add_argument("--substrate-col", default="")
    ap.add_argument("--metal-col", default="")
    ap.add_argument("--sep", default=";")

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")

    ap.add_argument("--substrate-threshold", type=float, default=None)
    ap.add_argument("--metal-threshold", type=float, default=None)
    ap.add_argument("--metal-presence-threshold", type=float, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    maps = ckpt.get("label_maps", {})
    ec_to_idx = maps.get("ec_to_idx", {})
    sub_to_idx = maps.get("substrate_to_idx", {})
    metal_to_idx = maps.get("metal_to_idx", {})
    if len(ec_to_idx) == 0:
        raise ValueError("Invalid checkpoint: missing ec_to_idx")

    id2ec = [x for x, _ in sorted(ec_to_idx.items(), key=lambda kv: kv[1])]

    seq_col = args.seq_col or cfg.get("seq_col", "sequence")
    ec_col = args.ec_col or cfg.get("ec_col", "ec_subclass")
    substrate_col = args.substrate_col or cfg.get("substrate_col", "substrate_labels")
    metal_col = args.metal_col or cfg.get("metal_col", "metal_labels")
    sep = args.sep

    df = pd.read_csv(args.csv)
    for c in [seq_col, ec_col, substrate_col, metal_col]:
        if c not in df.columns:
            raise KeyError(f"Column not found in CSV: {c}")

    df = df.copy()
    df[seq_col] = df[seq_col].astype(str).map(clean_sequence)
    df = df[df[seq_col].str.len() > 0].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid sequences after cleaning.")

    if args.split == "valid":
        _, eval_df = split_df(df, valid_size=args.valid_size, seed=args.seed, ec_col=ec_col)
    else:
        eval_df = df

    thresholds = ckpt.get("decision_thresholds", {})
    substrate_threshold = float(args.substrate_threshold) if args.substrate_threshold is not None else float(thresholds.get("substrate", 0.5))
    metal_threshold = float(args.metal_threshold) if args.metal_threshold is not None else float(thresholds.get("metal_type", thresholds.get("metal", 0.5)))
    metal_presence_threshold = float(args.metal_presence_threshold) if args.metal_presence_threshold is not None else float(thresholds.get("metal_presence", 0.5))
    substrate_thresholds = None
    metal_thresholds = None
    if args.substrate_threshold is None:
        sub_thr_map = thresholds.get("substrate_per_label", {}) or {}
        if len(sub_thr_map) > 0:
            id2sub = [x for x, _ in sorted(sub_to_idx.items(), key=lambda kv: kv[1])]
            substrate_thresholds = [float(sub_thr_map.get(lbl, substrate_threshold)) for lbl in id2sub]
    if args.metal_threshold is None:
        metal_thr_map = thresholds.get("metal_type_per_label", {}) or {}
        if len(metal_thr_map) > 0:
            id2metal = [x for x, _ in sorted(metal_to_idx.items(), key=lambda kv: kv[1])]
            metal_thresholds = [float(metal_thr_map.get(lbl, metal_threshold)) for lbl in id2metal]
    ec_log_prior = ckpt.get("ec_log_prior", [])
    ec_logit_adjust_tau = float(cfg.get("ec_logit_adjust_tau", 0.0))
    if not bool(cfg.get("ec_logit_adjust", False)):
        ec_logit_adjust_tau = 0.0

    model_name = cfg.get("model_name", "facebook/esm2_t6_8M_UR50D")
    device = get_device(args.device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = LigaseMultiTaskModel(
        model_name=model_name,
        num_ec=len(ec_to_idx),
        num_substrate=len(sub_to_idx),
        num_metal=len(metal_to_idx),
        dropout=float(cfg.get("dropout", 0.2)),
        freeze_backbone=True,
        metal_two_stage=bool(cfg.get("metal_two_stage", False)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    eval_rows = infer_eval_rows(
        model=model,
        tokenizer=tokenizer,
        df=eval_df,
        device=device,
        max_length=int(cfg.get("max_length", 512)),
        ec_to_idx=ec_to_idx,
        sub_to_idx=sub_to_idx,
        metal_to_idx=metal_to_idx,
        seq_col=seq_col,
        ec_col=ec_col,
        substrate_col=substrate_col,
        metal_col=metal_col,
        sep=sep,
        batch_size=args.batch_size,
        substrate_threshold=substrate_threshold,
        metal_threshold=metal_threshold,
        metal_presence_threshold=metal_presence_threshold,
        substrate_thresholds=substrate_thresholds,
        metal_thresholds=metal_thresholds,
        ec_log_prior=ec_log_prior if ec_log_prior else None,
        ec_logit_adjust_tau=ec_logit_adjust_tau,
    )

    overall = compute_overall_metrics(
        rows=eval_rows,
        id2ec=id2ec,
        metal_two_stage=bool(cfg.get("metal_two_stage", False)),
        metal_presence_threshold=metal_presence_threshold,
    )

    ec_cls_df, ec_strat_df, cm_counts, cm_row_norm = build_ec_reports(eval_rows, id2ec)

    with open(outdir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_metrics": overall,
                "thresholds": {
                    "substrate": substrate_threshold,
                    "metal_type": metal_threshold,
                    "metal_presence": metal_presence_threshold,
                    "substrate_per_label_enabled": bool(substrate_thresholds is not None),
                    "metal_type_per_label_enabled": bool(metal_thresholds is not None),
                    "ec_logit_adjust_tau": float(ec_logit_adjust_tau),
                },
                "split": args.split,
                "n_eval": int(len(eval_df)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame([overall]).to_csv(outdir / "overall_metrics.csv", index=False)

    if ec_cls_df is not None:
        ec_cls_df.to_csv(outdir / "ec_classification_report.csv", index=False)
    if ec_strat_df is not None:
        ec_strat_df.to_csv(outdir / "ec_stratified_report.csv", index=False)

    if cm_counts is not None:
        cm_df = pd.DataFrame(cm_counts, index=id2ec, columns=id2ec)
        cm_df.to_csv(outdir / "ec_confusion_counts.csv")
        cmn_df = pd.DataFrame(cm_row_norm, index=id2ec, columns=id2ec)
        cmn_df.to_csv(outdir / "ec_confusion_row_norm.csv")

        plot_confusion(
            cm=cm_counts,
            labels=id2ec,
            out_png=outdir / "fig_ec_confusion_counts.png",
            title="EC Subclass Confusion Matrix (Counts)",
            normalize=False,
        )
        plot_confusion(
            cm=cm_counts,
            labels=id2ec,
            out_png=outdir / "fig_ec_confusion_row_norm.png",
            title="EC Subclass Confusion Matrix (Row-normalized)",
            normalize=True,
        )

    if ec_cls_df is not None:
        plot_per_class_f1(ec_cls_df, outdir / "fig_ec_per_class_f1.png")

    write_notes(
        out_md=outdir / "presentation_notes.md",
        metrics=overall,
        thresholds={
            "substrate": substrate_threshold,
            "metal_type": metal_threshold,
            "metal_presence": metal_presence_threshold,
        },
        split_name=args.split,
        selected_n=int(len(eval_df)),
        ec_cls_df=ec_cls_df,
    )

    print(f"[Done] Saved outputs to: {outdir}")
    print(
        "[Summary] "
        f"ec_acc={pretty_metric(overall.get('ec_acc'))}, "
        f"ec_macro_f1={pretty_metric(overall.get('ec_macro_f1'))}, "
        f"substrate_f1={pretty_metric(overall.get('substrate_micro_f1'))}, "
        f"metal_f1={pretty_metric(overall.get('metal_micro_f1'))}"
    )


if __name__ == "__main__":
    main()
