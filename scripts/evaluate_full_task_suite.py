#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt  # noqa: E402


AA_ALLOWED = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_seq(seq: str) -> str:
    s = str(seq).strip().upper().replace(" ", "")
    if not s:
        return ""
    bad = [c for c in s if c not in AA_ALLOWED]
    if bad:
        return ""
    return s


def parse_fasta(path):
    records = []
    name = None
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            if t.startswith(">"):
                if name is not None:
                    seq = clean_seq("".join(chunks))
                    if seq:
                        records.append((name, seq))
                name = t[1:].strip() or f"seq_{len(records)+1}"
                chunks = []
            else:
                chunks.append(t)
    if name is not None:
        seq = clean_seq("".join(chunks))
        if seq:
            records.append((name, seq))
    return records


def get_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SeqClsDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        lab = int(self.labels[idx])
        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(lab, dtype=torch.long),
        }


class LigaseClassifier(nn.Module):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", num_classes=2):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name, local_files_only=True)
        self.classifier = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


class MeanPoolClassifier(nn.Module):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", num_classes=2):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name, local_files_only=True)
        self.classifier = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hs.size()).float()
        s = torch.sum(hs * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = s / denom
        return self.classifier(pooled)


@dataclass
class BinaryEvalResult:
    task: str
    n_total: int
    n_val: int
    n_val_used: int
    sampled: bool
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    pre = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def plot_confusion_binary(tn, fp, fn, tp, out_png, title, labels=("Negative", "Positive")):
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {labels[0]}", f"Pred {labels[1]}"])
    ax.set_yticklabels([f"True {labels[0]}", f"True {labels[1]}"])
    ax.set_title(title, fontsize=13, fontweight="bold")

    vmax = cm.max() if cm.size > 0 else 0
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            color = "white" if vmax > 0 and val > 0.55 * vmax else "#111827"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_binary_task(
    task_name,
    sequences,
    labels,
    model,
    tokenizer,
    device,
    outdir,
    batch_size=16,
    seed=42,
    val_ratio=0.2,
    max_val_samples=3000,
    threshold=0.5,
    cm_labels=("Negative", "Positive"),
):
    X_tr, X_va, y_tr, y_va = train_test_split(
        sequences,
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )

    sampled = False
    n_val_raw = len(X_va)
    if max_val_samples is not None and n_val_raw > max_val_samples:
        sampled = True
        rng = np.random.default_rng(seed)
        idx_all = np.arange(n_val_raw)
        y_va_np = np.asarray(y_va)

        pos_idx = idx_all[y_va_np == 1]
        neg_idx = idx_all[y_va_np == 0]
        n_pos = max(1, int(round(max_val_samples * (len(pos_idx) / n_val_raw))))
        n_neg = max_val_samples - n_pos
        n_pos = min(n_pos, len(pos_idx))
        n_neg = min(n_neg, len(neg_idx))

        keep = np.concatenate([
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ])
        rng.shuffle(keep)

        X_va = [X_va[i] for i in keep]
        y_va = [y_va[i] for i in keep]

    ds = SeqClsDataset(X_va, y_va, tokenizer=tokenizer, max_length=512)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    all_prob = []
    all_true = []

    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_prob.append(probs.detach().cpu().numpy())
            all_true.append(labels_t.detach().cpu().numpy())

    y_prob = np.concatenate(all_prob).astype(np.float64)
    y_true = np.concatenate(all_true).astype(np.int32)

    m = binary_metrics(y_true, y_prob, threshold=threshold)

    task_dir = Path(outdir) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_binary(
        m["tn"],
        m["fp"],
        m["fn"],
        m["tp"],
        out_png=task_dir / "confusion_matrix.png",
        title=f"{task_name} Confusion Matrix",
        labels=cm_labels,
    )

    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_prob_pos": y_prob,
            "y_pred": (y_prob >= threshold).astype(int),
        }
    )
    pred_df.to_csv(task_dir / "val_predictions.csv", index=False)

    result = BinaryEvalResult(
        task=task_name,
        n_total=len(sequences),
        n_val=n_val_raw,
        n_val_used=len(y_true),
        sampled=sampled,
        accuracy=m["accuracy"],
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        roc_auc=m["roc_auc"],
        tn=m["tn"],
        fp=m["fp"],
        fn=m["fn"],
        tp=m["tp"],
    )

    with open(task_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, ensure_ascii=False, indent=2)

    return result


def pearson_np(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman_np(a, b):
    a = pd.Series(np.asarray(a).reshape(-1))
    b = pd.Series(np.asarray(b).reshape(-1))
    return float(a.corr(b, method="spearman"))


def evaluate_kcat(oof_csv, outdir):
    df = pd.read_csv(oof_csv)
    if "y_pred_blend" in df.columns:
        y_pred_log = df["y_pred_blend"].values.astype(np.float64)
    elif "y_pred" in df.columns:
        y_pred_log = df["y_pred"].values.astype(np.float64)
    else:
        raise KeyError("Cannot find y_pred_blend or y_pred in kcat OOF CSV")

    y_true_log = df["y_true"].values.astype(np.float64)

    log_metrics = {
        "n": int(len(df)),
        "pearson_r": pearson_np(y_true_log, y_pred_log),
        "spearman_r": spearman_np(y_true_log, y_pred_log),
        "rmse": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2": float(r2_score(y_true_log, y_pred_log)),
    }

    y_true = np.power(10.0, y_true_log)
    y_pred = np.power(10.0, y_pred_log)

    eps = 1e-12
    kcat_metrics = {
        "pearson_r": pearson_np(y_true, y_pred),
        "spearman_r": spearman_np(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps)))),
    }

    out_task = Path(outdir) / "kcat"
    out_task.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(y_true_log, y_pred_log, s=18, alpha=0.75, color="#0ea5a6", edgecolor="none")
    mn = min(y_true_log.min(), y_pred_log.min())
    mx = max(y_true_log.max(), y_pred_log.max())
    ax.plot([mn, mx], [mn, mx], "--", color="#ef4444", linewidth=1.5)
    ax.set_xlabel("True log_kcat")
    ax.set_ylabel("Predicted log_kcat")
    ax.set_title("kcat Regression (log scale)", fontsize=13, fontweight="bold")
    txt = f"R={log_metrics['pearson_r']:.4f}\nRMSE={log_metrics['rmse']:.4f}\nR2={log_metrics['r2']:.4f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", bbox=dict(facecolor="white", edgecolor="#d1d5db"))
    fig.tight_layout()
    fig.savefig(out_task / "scatter_log_kcat.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(y_true, y_pred, s=18, alpha=0.75, color="#2563eb", edgecolor="none")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "--", color="#ef4444", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True kcat")
    ax.set_ylabel("Predicted kcat")
    ax.set_title("kcat Regression (original scale, log-log plot)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_task / "scatter_kcat_loglog.png", dpi=220)
    plt.close(fig)

    with open(out_task / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"log_kcat": log_metrics, "kcat": kcat_metrics}, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        {
            "y_true_log_kcat": y_true_log,
            "y_pred_log_kcat": y_pred_log,
            "y_true_kcat": y_true,
            "y_pred_kcat": y_pred,
        }
    ).to_csv(out_task / "oof_predictions_with_kcat.csv", index=False)

    return {"task": "kcat", "log_kcat": log_metrics, "kcat": kcat_metrics}


def write_report(out_path, class_results, kcat_result):
    lines = []
    lines.append("# 多任务统一评估报告")
    lines.append("")
    lines.append("## 1) 分类任务总览")
    lines.append("| 任务 | n_total | n_val | n_val_used | sampled | Acc | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in class_results:
        lines.append(
            "| {task} | {n_total} | {n_val} | {n_val_used} | {sampled} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {roc_auc:.4f} |".format(
                task=r.task,
                n_total=r.n_total,
                n_val=r.n_val,
                n_val_used=r.n_val_used,
                sampled=str(bool(r.sampled)),
                accuracy=r.accuracy,
                precision=r.precision,
                recall=r.recall,
                f1=r.f1,
                roc_auc=r.roc_auc if np.isfinite(r.roc_auc) else float("nan"),
            )
        )

    lines.append("")
    lines.append("## 2) kcat 回归任务")
    lk = kcat_result["log_kcat"]
    kk = kcat_result["kcat"]
    lines.append(
        f"- log_kcat: n={lk['n']}, Pearson R={lk['pearson_r']:.4f}, Spearman R={lk['spearman_r']:.4f}, RMSE={lk['rmse']:.4f}, MAE={lk['mae']:.4f}, R2={lk['r2']:.4f}"
    )
    lines.append(
        f"- kcat: Pearson R={kk['pearson_r']:.4f}, Spearman R={kk['spearman_r']:.4f}, RMSE={kk['rmse']:.4e}, MAE={kk['mae']:.4e}, R2={kk['r2']:.4f}, MAPE={kk['mape']:.4f}"
    )

    lines.append("")
    lines.append("## 3) 结果文件")
    lines.append("- 每个分类任务目录含 `metrics.json`、`val_predictions.csv`、`confusion_matrix.png`")
    lines.append("- kcat 目录含 `metrics.json`、`oof_predictions_with_kcat.csv`、两张散点图")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def first_existing_path(*paths: Path):
    for p in paths:
        if p and p.exists():
            return str(p.resolve())
    return str(paths[0].resolve()) if paths else ""


def main():
    cwd = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        default=first_existing_path(
            cwd / "outputs" / "full_task_eval_v1",
            cwd / "outputs" / "full_task_eval",
        ),
    )
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-val-samples", type=int, default=3000, help="None or <=0 means full validation set")
    ap.add_argument("--val-ratio", type=float, default=0.2)

    ap.add_argument("--ligase-model", default=str((cwd / "models" / "checkpoints" / "best_ligase_model.pth").resolve()))
    ap.add_argument("--ligase-pos", default=str((cwd / "data" / "raw" / "ligase_positive.fasta").resolve()))
    ap.add_argument("--ligase-neg", default=str((cwd / "data" / "raw" / "negative_balanced.fasta").resolve()))

    ap.add_argument(
        "--cofactor-model",
        default=str((cwd / "legacy" / "experiments" / "atp_nad" / "best_cofactor_model.pth").resolve()),
    )
    ap.add_argument(
        "--cofactor-atp",
        default=str((cwd / "legacy" / "experiments" / "atp_nad" / "atp_ligase.fasta").resolve()),
    )
    ap.add_argument(
        "--cofactor-nad",
        default=str((cwd / "legacy" / "experiments" / "atp_nad" / "nad_ligase.fasta").resolve()),
    )

    ap.add_argument(
        "--sol-model",
        default=str((cwd / "legacy" / "experiments" / "solubility" / "best_solubility_model.pth").resolve()),
    )
    ap.add_argument(
        "--sol-csv",
        default=str((cwd / "legacy" / "experiments" / "solubility" / "solubility_data.csv").resolve()),
    )

    ap.add_argument(
        "--kcat-oof",
        default=first_existing_path(
            cwd / "outputs" / "kcat_blend" / "blend" / "oof_predictions.csv",
        ),
    )

    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)

    max_val_samples = None if args.max_val_samples is None or args.max_val_samples <= 0 else int(args.max_val_samples)

    model_name = "facebook/esm2_t6_8M_UR50D"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            "Cannot load ESM tokenizer from local cache. Please ensure model is cached locally."
        ) from e

    # 1) Ligase
    lig_pos = parse_fasta(args.ligase_pos)
    lig_neg = parse_fasta(args.ligase_neg)
    lig_seqs = [s for _, s in lig_pos] + [s for _, s in lig_neg]
    lig_labels = [1] * len(lig_pos) + [0] * len(lig_neg)

    lig_model = LigaseClassifier(model_name=model_name)
    lig_model.load_state_dict(torch.load(args.ligase_model, map_location="cpu"), strict=False)

    lig_res = evaluate_binary_task(
        task_name="ligase_identification",
        sequences=lig_seqs,
        labels=lig_labels,
        model=lig_model,
        tokenizer=tokenizer,
        device=device,
        outdir=outdir,
        batch_size=args.batch_size,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_val_samples=max_val_samples,
        threshold=0.5,
        cm_labels=("Non-ligase", "Ligase"),
    )

    # 2) Cofactor ATP/NAD
    atp = parse_fasta(args.cofactor_atp)
    nad = parse_fasta(args.cofactor_nad)
    cof_seqs = [s for _, s in atp] + [s for _, s in nad]
    cof_labels = [1] * len(atp) + [0] * len(nad)

    cof_model = MeanPoolClassifier(model_name=model_name)
    cof_model.load_state_dict(torch.load(args.cofactor_model, map_location="cpu"), strict=False)

    cof_res = evaluate_binary_task(
        task_name="cofactor_atp_vs_nad",
        sequences=cof_seqs,
        labels=cof_labels,
        model=cof_model,
        tokenizer=tokenizer,
        device=device,
        outdir=outdir,
        batch_size=args.batch_size,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_val_samples=max_val_samples,
        threshold=0.5,
        cm_labels=("NAD", "ATP"),
    )

    # 3) Solubility
    sol_df = pd.read_csv(args.sol_csv)
    if "Sequence" not in sol_df.columns or "Label" not in sol_df.columns:
        raise KeyError("solubility CSV must have Sequence and Label columns")
    sol_df = sol_df.copy()
    sol_df["Sequence"] = sol_df["Sequence"].astype(str).map(clean_seq)
    sol_df = sol_df[sol_df["Sequence"].str.len() > 0].reset_index(drop=True)

    sol_seqs = sol_df["Sequence"].tolist()
    sol_labels = sol_df["Label"].astype(int).tolist()

    sol_model = MeanPoolClassifier(model_name=model_name)
    sol_model.load_state_dict(torch.load(args.sol_model, map_location="cpu"), strict=False)

    sol_res = evaluate_binary_task(
        task_name="solubility",
        sequences=sol_seqs,
        labels=sol_labels,
        model=sol_model,
        tokenizer=tokenizer,
        device=device,
        outdir=outdir,
        batch_size=args.batch_size,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_val_samples=max_val_samples,
        threshold=0.5,
        cm_labels=("Insoluble", "Soluble"),
    )

    # 4) kcat
    kcat_res = evaluate_kcat(args.kcat_oof, outdir=outdir)

    class_results = [lig_res, cof_res, sol_res]

    with open(outdir / "all_tasks_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "classification": [r.__dict__ for r in class_results],
                "kcat": kcat_res,
                "settings": {
                    "seed": args.seed,
                    "device": str(device),
                    "batch_size": args.batch_size,
                    "val_ratio": args.val_ratio,
                    "max_val_samples": max_val_samples,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    cls_df = pd.DataFrame([r.__dict__ for r in class_results])
    cls_df.to_csv(outdir / "classification_metrics.csv", index=False)

    write_report(outdir / "FULL_TASK_EVALUATION_REPORT.md", class_results, kcat_res)

    print(f"[Done] outputs: {outdir}")
    print(cls_df[["task", "accuracy", "f1", "roc_auc", "n_val_used", "sampled"]])
    print("[Done] report:", outdir / "FULL_TASK_EVALUATION_REPORT.md")


if __name__ == "__main__":
    main()
