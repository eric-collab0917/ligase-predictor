#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402


def fmt(x, d=4):
    try:
        if np.isnan(x):
            return "nan"
    except Exception:
        pass
    return f"{float(x):.{d}f}"


def load_img(path):
    p = Path(path)
    if p.exists():
        return mpimg.imread(str(p))
    return None


def draw_image_panel(ax, img, title):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    if img is None:
        ax.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=10)
        return
    ax.imshow(img)


def translate_task(task: str, lang: str):
    if lang != "zh":
        return task
    mp = {
        "ligase_identification": "连接酶鉴定",
        "cofactor_atp_vs_nad": "ATP / NAD 偏好",
        "solubility": "水溶性预测",
    }
    return mp.get(task, task)


def apply_plot_style(lang: str):
    if lang == "zh":
        plt.rcParams["font.sans-serif"] = [
            "PingFang SC",
            "Hiragino Sans GB",
            "Songti SC",
            "Heiti SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False


def build_dashboard(eval_dir: Path, out_png: Path, out_pdf: Path, lang: str = "en"):
    cls_csv = eval_dir / "classification_metrics.csv"
    kcat_metrics_json = eval_dir / "kcat" / "metrics.json"
    report_md = eval_dir / "FULL_TASK_EVALUATION_REPORT.md"

    if not cls_csv.exists():
        raise FileNotFoundError(f"Missing file: {cls_csv}")
    if not kcat_metrics_json.exists():
        raise FileNotFoundError(f"Missing file: {kcat_metrics_json}")

    cls = pd.read_csv(cls_csv)
    with open(kcat_metrics_json, "r", encoding="utf-8") as f:
        kcat_m = json.load(f)
    text_family = "monospace" if lang != "zh" else "sans-serif"

    # Images
    img_lig = load_img(eval_dir / "ligase_identification" / "confusion_matrix.png")
    img_cof = load_img(eval_dir / "cofactor_atp_vs_nad" / "confusion_matrix.png")
    img_sol = load_img(eval_dir / "solubility" / "confusion_matrix.png")
    img_log = load_img(eval_dir / "kcat" / "scatter_log_kcat.png")
    img_kcat = load_img(eval_dir / "kcat" / "scatter_kcat_loglog.png")

    apply_plot_style(lang)
    fig = plt.figure(figsize=(19, 12), facecolor="#f8fafc")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.1, 1.0], width_ratios=[1.2, 1.0, 1.0, 1.0])

    # Title
    if lang == "zh":
        title = "酶 AI 多任务评估课堂总览"
        subtitle = f"报告文件: {report_md.name}"
    else:
        title = "Enzyme AI Multi-task Evaluation Dashboard"
        subtitle = f"Report: {report_md.name}"

    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.958,
        subtitle,
        ha="center",
        fontsize=10,
        color="#374151",
    )

    # Panel A: classification bars
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(cls))
    w = 0.24
    ax0.bar(x - w, cls["accuracy"].values, width=w, label="Accuracy", color="#0f766e")
    ax0.bar(x, cls["f1"].values, width=w, label="F1", color="#1d4ed8")
    ax0.bar(x + w, cls["roc_auc"].values, width=w, label="ROC-AUC", color="#ea580c")
    ax0.set_xticks(x)
    ax0.set_xticklabels([translate_task(t, lang) for t in cls["task"].tolist()], rotation=15, ha="right")
    ax0.set_ylim(0, 1.02)
    ax0.set_ylabel("Score" if lang != "zh" else "指标值")
    ax0.set_title("Classification Task Performance" if lang != "zh" else "分类任务性能对比", fontsize=12, fontweight="bold")
    ax0.grid(axis="y", linestyle="--", alpha=0.28)
    ax0.legend(frameon=False, fontsize=9, loc="lower right")

    # Panel B: key metrics text
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.axis("off")

    lk = kcat_m["log_kcat"]
    kk = kcat_m["kcat"]

    lines = []
    lines.append("Key Results" if lang != "zh" else "核心结果")
    lines.append("")
    for _, r in cls.iterrows():
        task_name = translate_task(r["task"], lang)
        lines.append(
            f"- {task_name}: Acc={fmt(r['accuracy'])}, F1={fmt(r['f1'])}, AUC={fmt(r['roc_auc'])}, "
            f"n_val_used={int(r['n_val_used'])}, sampled={bool(r['sampled'])}"
        )
    lines.append("")
    lines.append(
        f"- log_kcat: Pearson={fmt(lk['pearson_r'])}, Spearman={fmt(lk['spearman_r'])}, RMSE={fmt(lk['rmse'])}, R2={fmt(lk['r2'])}"
    )
    lines.append(
        f"- kcat: Pearson={fmt(kk['pearson_r'])}, Spearman={fmt(kk['spearman_r'])}, RMSE={fmt(kk['rmse'],2)}, "
        f"R2={fmt(kk['r2'])}, MAPE={fmt(kk['mape'])}"
    )
    lines.append("")
    if lang == "zh":
        lines.append("课堂讲解要点")
        lines.append("1) 分类任务整体表现优秀，ATP/NAD 偏好几乎达到饱和。")
        lines.append("2) log_kcat 回归更稳定，原始 kcat 受极值影响更大。")
        lines.append("3) 下一步优先补齐长尾样本并做阈值校准。")
    else:
        lines.append("Classroom Talking Points")
        lines.append("1) Classification tasks are strong (especially ATP/NAD).")
        lines.append("2) log_kcat is reliable; raw kcat is more sensitive to extreme values.")
        lines.append("3) Improve long-tail data and calibrate thresholds for next iteration.")

    ax1.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        fontsize=11,
        family=text_family,
        bbox=dict(facecolor="white", edgecolor="#d1d5db", boxstyle="round,pad=0.6"),
    )

    # Panel C/D: regression scatters
    ax2 = fig.add_subplot(gs[1, 0:2])
    draw_image_panel(ax2, img_log, "log_kcat: True vs Pred" if lang != "zh" else "log_kcat：真实值 vs 预测值")

    ax3 = fig.add_subplot(gs[1, 2:4])
    draw_image_panel(ax3, img_kcat, "kcat: True vs Pred (log-log)" if lang != "zh" else "kcat：真实值 vs 预测值（log-log）")

    # Panel E/F/G: confusion matrices
    ax4 = fig.add_subplot(gs[2, 0])
    draw_image_panel(ax4, img_lig, "Ligase Identification Confusion Matrix" if lang != "zh" else "连接酶鉴定混淆矩阵")

    ax5 = fig.add_subplot(gs[2, 1])
    draw_image_panel(ax5, img_cof, "ATP/NAD Cofactor Confusion Matrix" if lang != "zh" else "ATP/NAD 偏好混淆矩阵")

    ax6 = fig.add_subplot(gs[2, 2])
    draw_image_panel(ax6, img_sol, "Solubility Confusion Matrix" if lang != "zh" else "水溶性混淆矩阵")

    # Panel H: task score radar-like table
    ax7 = fig.add_subplot(gs[2, 3])
    ax7.axis("off")
    table_lines = ["Task Summary Table", ""] if lang != "zh" else ["任务摘要表", ""]
    for _, r in cls.iterrows():
        task_name = translate_task(r["task"], lang)
        table_lines.append(
            f"{task_name[:22]:<22} | Acc {fmt(r['accuracy'],3)} | F1 {fmt(r['f1'],3)}"
        )
    table_lines += [
        "",
        f"log_kcat R    : {fmt(lk['pearson_r'],3)}",
        f"log_kcat RMSE : {fmt(lk['rmse'],3)}",
        f"kcat MAPE     : {fmt(kk['mape'],3)}",
    ]
    ax7.text(
        0.02,
        0.98,
        "\n".join(table_lines),
        va="top",
        fontsize=11,
        family=text_family,
        bbox=dict(facecolor="white", edgecolor="#d1d5db", boxstyle="round,pad=0.6"),
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    cwd = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    default_eval = cwd / "outputs" / "full_task_eval_v1"
    ap.add_argument("--eval-dir", default=str(default_eval.resolve()))
    ap.add_argument("--out-png", default="")
    ap.add_argument("--out-pdf", default="")
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    if args.lang == "zh":
        default_png = "CLASSROOM_DASHBOARD_CN.png"
        default_pdf = "CLASSROOM_DASHBOARD_CN.pdf"
    else:
        default_png = "CLASSROOM_DASHBOARD.png"
        default_pdf = "CLASSROOM_DASHBOARD.pdf"

    out_png = Path(args.out_png).resolve() if args.out_png else eval_dir / default_png
    out_pdf = Path(args.out_pdf).resolve() if args.out_pdf else eval_dir / default_pdf

    out_png.parent.mkdir(parents=True, exist_ok=True)
    build_dashboard(eval_dir, out_png, out_pdf, lang=args.lang)

    print(f"[Done] PNG: {out_png}")
    print(f"[Done] PDF: {out_pdf}")


if __name__ == "__main__":
    main()
