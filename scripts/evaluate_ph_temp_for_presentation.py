#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create presentation-ready evaluation figures and report for optimal pH and
temperature prediction.

Two evaluation modes are supported:
1) full:
   Evaluate on the full input CSV. This is useful for quick diagnostics but is
   optimistically biased if the same data was used during training.
2) recreated-val:
   Recreate the random validation split used by train_ligase_ph_temp.py using
   train_test_split(test_size=valid_size, random_state=seed). This is closer to
   the original validation metrics, assuming the same input CSV and split args.

Outputs:
- predictions.csv
- metrics.json
- ph_metrics.csv
- temp_metrics.csv
- fig1_ph_scatter.png
- fig2_temp_scatter.png
- fig3_ph_residuals.png
- fig4_temp_residuals.png
- fig5_tolerance_hits.png
- fig6_error_by_range.png
- presentation_notes.md
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.stats import pearsonr, spearmanr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ligase_ph_temp import LigasePhTempModel, clean_sequence, get_device

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def nice_style():
    plt.rcParams.update(
        {
            "figure.figsize": (8.6, 6.0),
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#f8fafc",
            "axes.grid": True,
            "grid.color": "#d1d5db",
            "grid.alpha": 0.35,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "font.size": 11,
        }
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="outputs/ligase_ph_temp_finetuned/best_ligase_ph_temp.pt",
        help="Path to pH/temp checkpoint",
    )
    ap.add_argument(
        "--csv",
        default="data/processed/ligase_ph_temp_auto.csv",
        help="Input CSV with sequence and true labels",
    )
    ap.add_argument(
        "--outdir",
        default="outputs/ligase_ph_temp_finetuned/eval_presentation",
        help="Output directory",
    )
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--ph-col", default="opt_ph")
    ap.add_argument("--temp-col", default="opt_temp")
    ap.add_argument(
        "--eval-mode",
        choices=["full", "recreated-val"],
        default="recreated-val",
        help="Use full dataset or recreate the training validation split",
    )
    ap.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Validation size used during training; only used in recreated-val mode",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for recreated-val split; -1 means load from config.json/checkpoint if available",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap repeats for confidence intervals; 0 disables bootstrap",
    )
    return ap.parse_args()


def load_model_and_tokenizer(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})

    model = LigasePhTempModel(
        model_name=config.get("model_name", "facebook/esm2_t6_8M_UR50D"),
        dropout=float(config.get("dropout", 0.2)),
        freeze_backbone=True,
        unfreeze_last_n_layers=0,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_ph_bins=int(config.get("ph_num_bins", 0)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    model_name = config.get("model_name", "facebook/esm2_t6_8M_UR50D")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, config, ckpt


def load_training_seed(checkpoint_path: Path, config: dict):
    if "seed" in config:
        return int(config["seed"])
    cfg_json = checkpoint_path.parent / "config.json"
    if cfg_json.exists():
        try:
            obj = json.loads(cfg_json.read_text(encoding="utf-8"))
            if "seed" in obj:
                return int(obj["seed"])
        except Exception:
            pass
    return 42


def safe_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def load_eval_dataframe(csv_path: Path, seq_col: str, id_col: str, ph_col: str, temp_col: str, max_samples: int):
    df = pd.read_csv(csv_path).copy()
    if seq_col not in df.columns:
        raise KeyError(f"Sequence column not found: {seq_col}")

    if id_col not in df.columns:
        df[id_col] = [f"row_{i}" for i in range(len(df))]

    df[seq_col] = df[seq_col].astype(str).map(clean_sequence)
    df = df[df[seq_col].str.len() > 0].reset_index(drop=True)
    df[ph_col] = df[ph_col].map(safe_float) if ph_col in df.columns else np.nan
    df[temp_col] = df[temp_col].map(safe_float) if temp_col in df.columns else np.nan

    has_ph = df[ph_col].between(0, 14, inclusive="both") if ph_col in df.columns else pd.Series(False, index=df.index)
    has_temp = df[temp_col].between(0, 120, inclusive="both") if temp_col in df.columns else pd.Series(False, index=df.index)
    df = df[has_ph | has_temp].reset_index(drop=True)

    if max_samples and max_samples > 0:
        df = df.iloc[:max_samples].copy().reset_index(drop=True)

    return df


def select_eval_split(df: pd.DataFrame, eval_mode: str, valid_size: float, seed: int):
    if eval_mode == "full":
        out = df.copy()
        out["_eval_split"] = "full"
        return out

    _, val_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=seed,
        shuffle=True,
    )
    out = val_df.copy().reset_index(drop=True)
    out["_eval_split"] = "recreated_val"
    return out


@torch.no_grad()
def predict_dataframe(df: pd.DataFrame, model, tokenizer, device, max_length: int, batch_size: int, seq_col: str):
    ph_preds = []
    temp_preds = []
    seqs = df[seq_col].tolist()

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        ph_pred, temp_pred = model(input_ids, attention_mask)
        ph_preds.extend(ph_pred.detach().cpu().numpy().astype(np.float64).tolist())
        temp_preds.extend(temp_pred.detach().cpu().numpy().astype(np.float64).tolist())

    out = df.copy()
    out["pred_opt_ph"] = np.asarray(ph_preds, dtype=np.float64)
    out["pred_opt_temp"] = np.asarray(temp_preds, dtype=np.float64)
    out["ph_abs_err"] = np.abs(out["pred_opt_ph"] - out["opt_ph"])
    out["temp_abs_err"] = np.abs(out["pred_opt_temp"] - out["opt_temp"])
    return out


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "mbe": float("nan"),
        }

    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    p = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    s = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": p,
        "spearman_r": s,
        "mbe": float(np.mean(residual)),
    }


def add_tolerance_metrics(metrics: dict, y_true, y_pred, tolerances):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    abs_err = np.abs(y_pred[mask] - y_true[mask])
    for tol in tolerances:
        metrics[f"acc_within_{str(tol).replace('.', 'p')}"] = float(np.mean(abs_err <= tol)) if len(abs_err) else float("nan")
    return metrics


def bootstrap_ci(y_true, y_pred, tolerances, n_boot=1000, seed=42):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0 or n_boot <= 0:
        return {}

    rng = np.random.default_rng(seed)
    keys = ["mae", "rmse", "r2", "pearson_r", "spearman_r"]
    keys.extend([f"acc_within_{str(t).replace('.', 'p')}" for t in tolerances])
    stats = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt = y_true[idx]
        yp = y_pred[idx]
        m = compute_metrics(yt, yp)
        m = add_tolerance_metrics(m, yt, yp, tolerances)
        for k in keys:
            stats[k].append(m.get(k, np.nan))

    ci = {}
    for k, arr in stats.items():
        arr = np.asarray(arr, dtype=np.float64)
        ci[k] = (
            float(np.nanpercentile(arr, 2.5)),
            float(np.nanpercentile(arr, 97.5)),
        )
    return ci


def plot_scatter(y_true, y_pred, out_png, title, xlabel, ylabel, stat_text):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.3, 6.2))
    ax.scatter(y_true, y_pred, s=28, alpha=0.78, color="#0ea5a6", edgecolor="none")
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    pad = 0.06 * (mx - mn + 1e-8)
    lo, hi = mn - pad, mx + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color="#ef4444", label="Ideal: y=x")

    coef = np.polyfit(y_true, y_pred, deg=1)
    xx = np.linspace(lo, hi, 100)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, color="#1d4ed8", linewidth=2.0, label=f"Fit: y={coef[0]:.2f}x+{coef[1]:.2f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.text(
        0.02,
        0.98,
        stat_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.35"),
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_residuals(y_true, y_pred, out_png, title, xlabel):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    ax = axes[0]
    ax.scatter(y_true, residual, s=24, alpha=0.72, color="#2563eb", edgecolor="none")
    ax.axhline(0.0, linestyle="--", color="#ef4444", linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (pred - true)")
    ax.set_title("Residual vs True", fontsize=12, fontweight="bold")

    ax2 = axes[1]
    ax2.hist(residual, bins=24, color="#f59e0b", alpha=0.88, edgecolor="#b45309")
    ax2.axvline(float(np.mean(residual)), color="#1d4ed8", linewidth=1.8, label="Mean residual")
    ax2.axvline(0.0, color="#ef4444", linestyle="--", linewidth=1.4, label="Zero error")
    ax2.set_xlabel("Residual (pred - true)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tolerance_hits(ph_metrics, temp_metrics, out_png):
    labels = ["pH<=0.5", "pH<=1.0", "Temp<=5C", "Temp<=10C"]
    values = [
        ph_metrics.get("acc_within_0p5", np.nan),
        ph_metrics.get("acc_within_1p0", np.nan),
        temp_metrics.get("acc_within_5", np.nan),
        temp_metrics.get("acc_within_10", np.nan),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = ["#f59e0b", "#f97316", "#38bdf8", "#2563eb"]
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy within tolerance")
    ax.set_title("Practical Tolerance Accuracy", fontsize=15, fontweight="bold")
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _binned_mae(y_true, y_pred, edges):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    abs_err = np.abs(y_pred - y_true)
    vals = []
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        vals.append(float(abs_err[mask].mean()) if np.any(mask) else float("nan"))
        labels.append(f"{lo:g}-{hi:g}")
    return labels, vals


def plot_error_by_range(ph_true, ph_pred, temp_true, temp_pred, out_png):
    ph_labels, ph_vals = _binned_mae(ph_true, ph_pred, [0, 5.5, 7.5, 9.0, 14.1])
    temp_labels, temp_vals = _binned_mae(temp_true, temp_pred, [0, 25, 40, 60, 120.1])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax = axes[0]
    ax.bar(ph_labels, ph_vals, color=["#fde68a", "#fbbf24", "#f59e0b", "#d97706"])
    ax.set_ylabel("MAE")
    ax.set_xlabel("True pH range")
    ax.set_title("pH Error by Target Range", fontsize=12, fontweight="bold")
    for i, v in enumerate(ph_vals):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    ax2.bar(temp_labels, temp_vals, color=["#bae6fd", "#7dd3fc", "#38bdf8", "#2563eb"])
    ax2.set_ylabel("MAE (°C)")
    ax2.set_xlabel("True temperature range (°C)")
    ax2.set_title("Temperature Error by Target Range", fontsize=12, fontweight="bold")
    for i, v in enumerate(temp_vals):
        if np.isfinite(v):
            ax2.text(i, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Where the Model Makes Larger Errors", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def metrics_to_csv_row(task_name, metrics, ci):
    row = {"task": task_name}
    for k, v in metrics.items():
        row[k] = v
        if k in ci:
            row[f"{k}_ci_low"] = ci[k][0]
            row[f"{k}_ci_high"] = ci[k][1]
    return row


def write_notes(out_md: Path, summary: dict):
    ph = summary["ph_metrics"]
    temp = summary["temp_metrics"]
    lines = []
    lines.append("# pH / Temperature Model Evaluation Notes")
    lines.append("")
    lines.append("## 1) Evaluation Setup")
    lines.append(f"- Evaluation mode: `{summary['eval_mode']}`")
    lines.append(f"- Data file: `{summary['csv']}`")
    lines.append(f"- Checkpoint: `{summary['checkpoint']}`")
    lines.append(f"- Number of evaluated sequences: `n={summary['n_eval']}`")
    lines.append("")
    lines.append("## 2) Core Results")
    lines.append(
        f"- pH: n={ph['n']}, MAE={ph['mae']:.3f}, RMSE={ph['rmse']:.3f}, R2={ph['r2']:.3f}, Pearson R={ph['pearson_r']:.3f}"
    )
    lines.append(
        f"- Temperature: n={temp['n']}, MAE={temp['mae']:.3f} °C, RMSE={temp['rmse']:.3f} °C, R2={temp['r2']:.3f}, Pearson R={temp['pearson_r']:.3f}"
    )
    if np.isfinite(ph.get("acc_within_1p0", np.nan)):
        lines.append(f"- pH tolerance accuracy: <=0.5 = {ph.get('acc_within_0p5', np.nan):.3f}, <=1.0 = {ph.get('acc_within_1p0', np.nan):.3f}")
    if np.isfinite(temp.get("acc_within_10", np.nan)):
        lines.append(f"- Temperature tolerance accuracy: <=5C = {temp.get('acc_within_5', np.nan):.3f}, <=10C = {temp.get('acc_within_10', np.nan):.3f}")
    lines.append("")
    lines.append("## 3) How to Explain the Figures")
    lines.append("- `fig1_ph_scatter.png`: pH true-vs-pred scatter. Points closer to `y=x` indicate better calibration.")
    lines.append("- `fig2_temp_scatter.png`: temperature true-vs-pred scatter. This often shows whether the model captures the overall thermal trend.")
    lines.append("- `fig3_ph_residuals.png` and `fig4_temp_residuals.png`: used to check systematic overestimation or underestimation.")
    lines.append("- `fig5_tolerance_hits.png`: easiest practical figure for class presentation, because it directly answers how often the model is close enough.")
    lines.append("- `fig6_error_by_range.png`: shows whether extreme acidic/alkaline or high-temperature samples are harder to predict.")
    lines.append("")
    lines.append("## 4) Interpretation Standard")
    lines.append("- For pH, MAE within about 0.5-1.0 is usually practically meaningful.")
    lines.append("- For temperature, MAE within about 5-10 °C is usually usable for rough experimental guidance.")
    lines.append("- If `R2 < 0`, the model is worse than predicting the dataset mean and should not be presented as reliable.")
    lines.append("")
    lines.append("## 5) Important Limitation")
    if summary["eval_mode"] == "full":
        lines.append("- This report uses the full dataset for apparent-fit diagnostics, so the numbers are optimistic and should not be claimed as strict generalization performance.")
    else:
        lines.append("- This report recreates the original random validation split. It is closer to the training-time result, but still not a homology-aware evaluation.")
    lines.append("- For a stricter bioinformatics evaluation, the next step should be sequence-similarity-grouped splitting instead of plain random split.")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    csv_path = Path(args.csv).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    device = get_device(args.device)
    model, tokenizer, config, _ = load_model_and_tokenizer(checkpoint_path, device)
    seed = args.seed if args.seed >= 0 else load_training_seed(checkpoint_path, config)
    max_length = int(config.get("max_length", 512))
    batch_size = int(args.batch_size or config.get("batch_size", 8))

    df = load_eval_dataframe(
        csv_path=csv_path,
        seq_col=args.seq_col,
        id_col=args.id_col,
        ph_col=args.ph_col,
        temp_col=args.temp_col,
        max_samples=args.max_samples,
    )
    eval_df = select_eval_split(df, args.eval_mode, args.valid_size, seed)
    eval_df = predict_dataframe(
        eval_df.rename(columns={args.ph_col: "opt_ph", args.temp_col: "opt_temp"}),
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        seq_col=args.seq_col,
    )

    ph_mask = np.isfinite(eval_df["opt_ph"].to_numpy())
    temp_mask = np.isfinite(eval_df["opt_temp"].to_numpy())
    ph_true = eval_df.loc[ph_mask, "opt_ph"].to_numpy(dtype=np.float64)
    ph_pred = eval_df.loc[ph_mask, "pred_opt_ph"].to_numpy(dtype=np.float64)
    temp_true = eval_df.loc[temp_mask, "opt_temp"].to_numpy(dtype=np.float64)
    temp_pred = eval_df.loc[temp_mask, "pred_opt_temp"].to_numpy(dtype=np.float64)

    ph_metrics = add_tolerance_metrics(compute_metrics(ph_true, ph_pred), ph_true, ph_pred, [0.5, 1.0])
    temp_metrics = add_tolerance_metrics(compute_metrics(temp_true, temp_pred), temp_true, temp_pred, [5, 10])

    ph_ci = bootstrap_ci(ph_true, ph_pred, [0.5, 1.0], n_boot=args.bootstrap, seed=seed)
    temp_ci = bootstrap_ci(temp_true, temp_pred, [5, 10], n_boot=args.bootstrap, seed=seed + 1)

    summary = {
        "checkpoint": str(checkpoint_path),
        "csv": str(csv_path),
        "eval_mode": args.eval_mode,
        "seed": seed,
        "valid_size": args.valid_size,
        "n_eval": int(len(eval_df)),
        "ph_metrics": ph_metrics,
        "temp_metrics": temp_metrics,
    }

    summary_path = outdir / "metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    keep_cols = [args.id_col, args.seq_col, "_eval_split", "opt_ph", "pred_opt_ph", "ph_abs_err", "opt_temp", "pred_opt_temp", "temp_abs_err"]
    keep_cols = [c for c in keep_cols if c in eval_df.columns]
    eval_df[keep_cols].to_csv(outdir / "predictions.csv", index=False)

    pd.DataFrame([metrics_to_csv_row("opt_ph", ph_metrics, ph_ci)]).to_csv(outdir / "ph_metrics.csv", index=False)
    pd.DataFrame([metrics_to_csv_row("opt_temp", temp_metrics, temp_ci)]).to_csv(outdir / "temp_metrics.csv", index=False)

    nice_style()
    plot_scatter(
        ph_true,
        ph_pred,
        outdir / "fig1_ph_scatter.png",
        title="Optimal pH: True vs Predicted",
        xlabel="True optimal pH",
        ylabel="Predicted optimal pH",
        stat_text=f"n={ph_metrics['n']}\nPearson R={ph_metrics['pearson_r']:.3f}\nMAE={ph_metrics['mae']:.3f}\nR2={ph_metrics['r2']:.3f}",
    )
    plot_scatter(
        temp_true,
        temp_pred,
        outdir / "fig2_temp_scatter.png",
        title="Optimal Temperature: True vs Predicted",
        xlabel="True optimal temperature (°C)",
        ylabel="Predicted optimal temperature (°C)",
        stat_text=f"n={temp_metrics['n']}\nPearson R={temp_metrics['pearson_r']:.3f}\nMAE={temp_metrics['mae']:.3f} °C\nR2={temp_metrics['r2']:.3f}",
    )
    plot_residuals(
        ph_true,
        ph_pred,
        outdir / "fig3_ph_residuals.png",
        title="pH Residual Diagnostics",
        xlabel="True optimal pH",
    )
    plot_residuals(
        temp_true,
        temp_pred,
        outdir / "fig4_temp_residuals.png",
        title="Temperature Residual Diagnostics",
        xlabel="True optimal temperature (°C)",
    )
    plot_tolerance_hits(ph_metrics, temp_metrics, outdir / "fig5_tolerance_hits.png")
    plot_error_by_range(ph_true, ph_pred, temp_true, temp_pred, outdir / "fig6_error_by_range.png")
    write_notes(outdir / "presentation_notes.md", summary)

    print(f"[Done] Evaluation report generated in: {outdir}")
    for name in [
        "metrics.json",
        "predictions.csv",
        "ph_metrics.csv",
        "temp_metrics.csv",
        "fig1_ph_scatter.png",
        "fig2_temp_scatter.png",
        "fig3_ph_residuals.png",
        "fig4_temp_residuals.png",
        "fig5_tolerance_hits.png",
        "fig6_error_by_range.png",
        "presentation_notes.md",
    ]:
        print(f" - {outdir / name}")


if __name__ == "__main__":
    main()
