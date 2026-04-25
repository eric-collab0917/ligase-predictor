#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create presentation-ready evaluation figures and report for kcat regression.

Inputs:
- summary.json from train_kcat_baseline.py (blend or non-blend)

Outputs (in outdir):
- metrics_overview.csv
- metrics_with_ci.csv
- fig1_scatter_blend.png
- fig2_residuals_blend.png
- fig3_model_comparison.png
- fig4_fold_stability.png
- fig5_error_by_target_bin.png
- presentation_notes.md
"""

import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.stats import pearsonr, spearmanr


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    p = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    s = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    return {
        "pearson_r": p,
        "spearman_r": s,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mbe": float(np.mean(residual)),  # mean bias error
    }


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = len(y_true)
    stats = {
        "pearson_r": [],
        "spearman_r": [],
        "rmse": [],
        "mae": [],
        "r2": [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        m = compute_metrics(yt, yp)
        for k in stats:
            stats[k].append(m[k])
    ci = {}
    for k, arr in stats.items():
        arr = np.asarray(arr, dtype=np.float64)
        ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci


def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_entry_map(summary):
    return {m["model_name"]: m for m in summary.get("models", [])}


def load_oof(path):
    df = pd.read_csv(path)
    return df


def _relocate_old_outputs_path(path_str: str, project_root: Path):
    p = Path(path_str)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.startswith("outputs_"):
            sub = part[len("outputs_") :]
            trailing = parts[i + 1 :]
            cand = project_root / "outputs" / sub
            for t in trailing:
                cand = cand / t
            if cand.exists():
                return cand
            break
    return None


def resolve_oof_path(path_str: str, base_dir: Path):
    p = Path(path_str)
    if p.is_absolute():
        if p.exists():
            return p
        relocated = _relocate_old_outputs_path(path_str, project_root=base_dir.parent)
        return relocated
    cand = (base_dir / p).resolve()
    return cand if cand.exists() else None


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


def plot_scatter(y_true, y_pred, out_png, title, subtitle=""):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.scatter(y_true, y_pred, s=26, alpha=0.75, color="#0ea5a6", edgecolor="none")
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (mx - mn + 1e-8)
    lo, hi = mn - pad, mx + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="#ef4444", label="Ideal: y=x")

    coef = np.polyfit(y_true, y_pred, deg=1)
    xx = np.linspace(lo, hi, 120)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, color="#1d4ed8", linewidth=2.0, label=f"Fit: y={coef[0]:.2f}x+{coef[1]:.2f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True log_kcat")
    ax.set_ylabel("Predicted log_kcat")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color="#334155")
    text = f"n={len(y_true)}\nPearson R={m['pearson_r']:.4f}\nRMSE={m['rmse']:.4f}\nR2={m['r2']:.4f}"
    ax.text(
        0.02,
        0.98,
        text,
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


def plot_residuals(y_true, y_pred, out_png, title):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    ax = axes[0]
    ax.scatter(y_pred, residual, s=24, alpha=0.7, color="#2563eb", edgecolor="none")
    ax.axhline(0.0, linestyle="--", color="#ef4444", linewidth=1.5)
    ax.set_xlabel("Predicted log_kcat")
    ax.set_ylabel("Residual (pred - true)")
    ax.set_title("Residual vs Prediction", fontsize=12, fontweight="bold")

    ax2 = axes[1]
    ax2.hist(residual, bins=28, color="#10b981", alpha=0.86, edgecolor="#0f766e")
    ax2.axvline(float(np.mean(residual)), color="#1e3a8a", linestyle="-", linewidth=1.8, label="Mean residual")
    ax2.axvline(0.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Zero error")
    ax2.set_xlabel("Residual (pred - true)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics_df, out_png):
    order = metrics_df.sort_values("pearson_r", ascending=False)["model"].tolist()
    d = metrics_df.set_index("model").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    ax = axes[0]
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d["pearson_r"].values, width=w, color="#0ea5a6", label="Pearson R")
    ax.bar(x + w / 2, d["r2"].values, width=w, color="#2563eb", label="R2")
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"].values)
    ax.set_ylim(0, 1.0)
    ax.set_title("Correlation Metrics (Higher is better)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)

    ax2 = axes[1]
    ax2.bar(x - w / 2, d["rmse"].values, width=w, color="#f59e0b", label="RMSE")
    ax2.bar(x + w / 2, d["mae"].values, width=w, color="#ef4444", label="MAE")
    ax2.set_xticks(x)
    ax2.set_xticklabels(d["model"].values)
    ax2.set_title("Error Metrics (Lower is better)", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)

    fig.suptitle("Model Comparison on OOF Predictions", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fold_stability(summary, out_png):
    rows = []
    for m in summary.get("models", []):
        model_name = m.get("model_name")
        folds = m.get("folds", [])
        for f in folds:
            rows.append(
                {
                    "model": model_name,
                    "fold": f["fold"],
                    "pearson_r": f["pearson_r"],
                    "rmse": f["rmse"],
                    "r2": f["r2"],
                }
            )
    if not rows:
        return False

    df = pd.DataFrame(rows)
    models = sorted(df["model"].unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    data_r = [df[df["model"] == m]["pearson_r"].values for m in models]
    bp = ax.boxplot(data_r, tick_labels=models, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], ["#14b8a6", "#60a5fa", "#f59e0b"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_title("Fold-wise Pearson R", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pearson R")

    ax2 = axes[1]
    data_rmse = [df[df["model"] == m]["rmse"].values for m in models]
    bp2 = ax2.boxplot(data_rmse, tick_labels=models, patch_artist=True, widths=0.6)
    for patch, c in zip(bp2["boxes"], ["#14b8a6", "#60a5fa", "#f59e0b"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax2.set_title("Fold-wise RMSE", fontsize=12, fontweight="bold")
    ax2.set_ylabel("RMSE")

    fig.suptitle("Cross-validation Stability", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_error_by_target_bin(y_true, y_pred, out_png):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    abs_err = np.abs(y_pred - y_true)

    q = np.quantile(y_true, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    q[0] -= 1e-9
    q[-1] += 1e-9
    bins = np.digitize(y_true, q[1:-1], right=True)

    mae_by_bin = []
    labels = []
    for i in range(5):
        mask = bins == i
        mae_by_bin.append(float(abs_err[mask].mean()) if np.any(mask) else float("nan"))
        labels.append(f"Q{i+1}")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, mae_by_bin, color=["#0ea5a6", "#22c55e", "#f59e0b", "#f97316", "#ef4444"])
    ax.set_ylabel("MAE")
    ax.set_xlabel("True log_kcat quantile bin")
    ax.set_title("Error by Target Range (Blend OOF)", fontsize=14, fontweight="bold")
    for i, v in enumerate(mae_by_bin):
        if np.isfinite(v):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def to_table(metrics_map):
    rows = []
    for name, m in metrics_map.items():
        rows.append({"model": name, **m})
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)


def write_notes(out_md, metrics_df, ci_map, best_model):
    top = metrics_df.iloc[0].to_dict()
    lines = []
    lines.append("# kcat Model Evaluation Notes (Class Presentation)")
    lines.append("")
    lines.append("## 1) Task & Setup")
    lines.append("- 任务：预测酶催化活性 `log_kcat`（回归）。")
    lines.append(f"- 数据规模：`n={int(top.get('n_samples', 0))}`（OOF 评估）。")
    lines.append("- 评估方式：Repeated 5-fold CV 的 OOF（避免单次划分偶然性）。")
    lines.append("")
    lines.append("## 2) Core Result")
    lines.append(
        f"- 最优模型：`{best_model}`，Pearson R={top['pearson_r']:.4f}，RMSE={top['rmse']:.4f}，R2={top['r2']:.4f}。"
    )
    if best_model in ci_map:
        ci = ci_map[best_model]
        lines.append(
            f"- 95% Bootstrap CI: Pearson R [{ci['pearson_r'][0]:.4f}, {ci['pearson_r'][1]:.4f}], "
            f"RMSE [{ci['rmse'][0]:.4f}, {ci['rmse'][1]:.4f}]。"
        )
    lines.append("")
    lines.append("## 3) How to Explain Figures")
    lines.append("- `fig1_scatter_blend.png`: 点越贴近 `y=x` 越好，直观看整体拟合能力。")
    lines.append("- `fig2_residuals_blend.png`: 观察误差是否偏置、是否存在系统性高估/低估。")
    lines.append("- `fig3_model_comparison.png`: 同时比较相关性指标（R/R2）和误差指标（RMSE/MAE）。")
    lines.append("- `fig4_fold_stability.png`: 展示不同折之间稳定性，说明结果不是偶然。")
    lines.append("- `fig5_error_by_target_bin.png`: 解释模型在低/中/高活性区间的误差差异。")
    lines.append("")
    lines.append("## 4) Suggested Talking Points")
    lines.append("- 小样本场景下，轻量模型与特征工程优于复杂重型 GNN。")
    lines.append("- 融合模型（lgbm+xgb）相对单模型带来稳定增益。")
    lines.append("- 后续可通过更严格同源分组验证进一步检验泛化能力。")
    lines.append("")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to summary.json")
    ap.add_argument("--outdir", default="", help="Output directory for figures/report")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap repeats for CI")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = load_summary(str(summary_path))
    base_dir = summary_path.parent
    outdir = Path(args.outdir).resolve() if args.outdir else (base_dir / "evaluation_report")
    outdir.mkdir(parents=True, exist_ok=True)

    nice_style()

    entry = model_entry_map(summary)

    # Load OOF predictions if paths are available.
    oof_map = {}
    for name, e in entry.items():
        p = e.get("oof_path")
        if p:
            pth = resolve_oof_path(p, base_dir=base_dir)
            if pth is not None and pth.exists():
                oof_map[name] = load_oof(str(pth))

    if "blend" not in oof_map:
        raise FileNotFoundError("Blend OOF not found in summary; expected model 'blend' with oof_path.")

    blend_df = oof_map["blend"]
    y_true = blend_df["y_true"].to_numpy(dtype=np.float64)

    # Metrics from OOF files (preferred consistency).
    metrics_map = {}
    if "blend" in oof_map:
        metrics_map["blend"] = compute_metrics(
            oof_map["blend"]["y_true"].to_numpy(),
            oof_map["blend"]["y_pred_blend"].to_numpy(),
        )
    if "lgbm" in oof_map:
        col = "y_pred" if "y_pred" in oof_map["lgbm"].columns else "pred"
        metrics_map["lgbm"] = compute_metrics(oof_map["lgbm"]["y_true"].to_numpy(), oof_map["lgbm"][col].to_numpy())
    if "xgb" in oof_map:
        col = "y_pred" if "y_pred" in oof_map["xgb"].columns else "pred"
        metrics_map["xgb"] = compute_metrics(oof_map["xgb"]["y_true"].to_numpy(), oof_map["xgb"][col].to_numpy())

    metrics_df = to_table(metrics_map)
    metrics_df.insert(1, "n_samples", len(y_true))
    metrics_df.to_csv(outdir / "metrics_overview.csv", index=False)

    # Bootstrap CI for each model in metrics_map
    ci_map = {}
    for model_name, m in metrics_map.items():
        if model_name == "blend":
            yp = blend_df["y_pred_blend"].to_numpy()
            yt = blend_df["y_true"].to_numpy()
        else:
            df = oof_map[model_name]
            col = "y_pred" if "y_pred" in df.columns else "pred"
            yp = df[col].to_numpy()
            yt = df["y_true"].to_numpy()
        ci_map[model_name] = bootstrap_ci(yt, yp, n_boot=args.bootstrap, seed=args.seed)

    ci_rows = []
    for model_name, m in metrics_map.items():
        row = {"model": model_name}
        for k, v in m.items():
            row[k] = v
            c = ci_map[model_name][k] if k in ci_map[model_name] else (np.nan, np.nan)
            row[f"{k}_ci_low"] = c[0]
            row[f"{k}_ci_high"] = c[1]
        ci_rows.append(row)
    pd.DataFrame(ci_rows).sort_values("pearson_r", ascending=False).to_csv(
        outdir / "metrics_with_ci.csv", index=False
    )

    # Figures
    plot_scatter(
        blend_df["y_true"].to_numpy(),
        blend_df["y_pred_blend"].to_numpy(),
        outdir / "fig1_scatter_blend.png",
        title="Blend Model: True vs Predicted log_kcat (OOF)",
        subtitle="OOF predictions across repeated 5-fold CV",
    )
    plot_residuals(
        blend_df["y_true"].to_numpy(),
        blend_df["y_pred_blend"].to_numpy(),
        outdir / "fig2_residuals_blend.png",
        title="Blend Model Residual Diagnostics",
    )
    plot_model_comparison(metrics_df, outdir / "fig3_model_comparison.png")
    plot_fold_stability(summary, outdir / "fig4_fold_stability.png")
    plot_error_by_target_bin(
        blend_df["y_true"].to_numpy(),
        blend_df["y_pred_blend"].to_numpy(),
        outdir / "fig5_error_by_target_bin.png",
    )

    best_model = metrics_df.iloc[0]["model"]
    write_notes(outdir / "presentation_notes.md", metrics_df, ci_map, best_model=best_model)

    print(f"[Done] Evaluation report generated in: {outdir}")
    print("[Done] Key files:")
    for name in [
        "metrics_overview.csv",
        "metrics_with_ci.csv",
        "fig1_scatter_blend.png",
        "fig2_residuals_blend.png",
        "fig3_model_comparison.png",
        "fig4_fold_stability.png",
        "fig5_error_by_target_bin.png",
        "presentation_notes.md",
    ]:
        p = outdir / name
        print(f" - {p}")


if __name__ == "__main__":
    main()
