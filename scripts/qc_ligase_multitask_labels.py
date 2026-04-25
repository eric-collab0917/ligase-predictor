#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quality control for ligase multitask dataset labels.

Checks:
- missingness and coverage
- sequence length stats
- duplicate sequences / duplicate IDs
- EC format validity and class distribution
- substrate/metal multi-label distribution
- inconsistent label spellings (case/space variants)
- conflicting labels for same sequence
- trainability risk warnings (rare classes, too many missing labels)

Outputs in outdir:
- qc_summary.json
- ec_counts.csv
- substrate_counts.csv
- metal_counts.csv
- suspicious_variants.csv
- sequence_conflicts.csv
- row_coverage.csv
- recommendations.md
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def clean_sequence(seq: str) -> str:
    s = re.sub(r"\s+", "", str(seq)).upper()
    if not s:
        return ""
    if re.search(r"[^ACDEFGHIKLMNPQRSTVWYBXZUOJ]", s):
        return ""
    return s


def parse_multilabel(v: str, sep=";") -> List[str]:
    if v is None:
        return []
    t = str(v).strip()
    if not t or t.lower() in {"nan", "none", "null"}:
        return []
    return [x.strip() for x in t.split(sep) if x.strip()]


def canonical_token(tok: str) -> str:
    t = str(tok).strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9+_.-]", "", t)
    return t


def ec_subclass_ok(x: str) -> bool:
    return bool(re.fullmatch(r"\d+\.\d+", str(x).strip()))


def count_tokens(series: pd.Series, sep=";") -> Dict[str, int]:
    c = {}
    for v in series.fillna("").astype(str):
        for t in parse_multilabel(v, sep=sep):
            c[t] = c.get(t, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def build_variant_table(tokens: List[str]) -> pd.DataFrame:
    raw = [x for x in tokens if str(x).strip()]
    if not raw:
        return pd.DataFrame(columns=["canonical", "variants", "n_variants", "total_count"])

    # count raw tokens
    raw_count = {}
    for t in raw:
        raw_count[t] = raw_count.get(t, 0) + 1

    # group by canonical key
    g = {}
    for t, n in raw_count.items():
        key = canonical_token(t)
        g.setdefault(key, []).append((t, n))

    rows = []
    for key, arr in g.items():
        if len(arr) <= 1:
            continue
        arr = sorted(arr, key=lambda x: (-x[1], x[0]))
        rows.append(
            {
                "canonical": key,
                "variants": " | ".join([f"{t} ({n})" for t, n in arr]),
                "n_variants": len(arr),
                "total_count": int(sum(n for _, n in arr)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["canonical", "variants", "n_variants", "total_count"])
    return pd.DataFrame(rows).sort_values(["total_count", "n_variants"], ascending=[False, False])


def conflict_records(df: pd.DataFrame, seq_col: str, ec_col: str, sub_col: str, metal_col: str, sep=";") -> pd.DataFrame:
    rows = []
    g = df.groupby(seq_col, dropna=False)
    for seq, d in g:
        if not isinstance(seq, str) or not seq:
            continue
        ec_set = set([str(x).strip() for x in d[ec_col].fillna("").tolist() if str(x).strip()])
        sub_set = set()
        metal_set = set()
        for v in d[sub_col].fillna("").astype(str):
            sub_set.update(parse_multilabel(v, sep=sep))
        for v in d[metal_col].fillna("").astype(str):
            metal_set.update(parse_multilabel(v, sep=sep))

        ec_conflict = len(ec_set) > 1
        # substrate/metal conflict: same sequence has both NONE and positive labels
        sub_conflict = ("NONE" in sub_set and len(sub_set - {"NONE"}) > 0)
        metal_conflict = ("NONE" in metal_set and len(metal_set - {"NONE"}) > 0)

        if ec_conflict or sub_conflict or metal_conflict:
            rows.append(
                {
                    "sequence_len": len(seq),
                    "n_rows": len(d),
                    "ec_values": ";".join(sorted(ec_set)),
                    "substrate_values": ";".join(sorted(sub_set)),
                    "metal_values": ";".join(sorted(metal_set)),
                    "ec_conflict": ec_conflict,
                    "substrate_conflict": sub_conflict,
                    "metal_conflict": metal_conflict,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "sequence_len",
                "n_rows",
                "ec_values",
                "substrate_values",
                "metal_values",
                "ec_conflict",
                "substrate_conflict",
                "metal_conflict",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["ec_conflict", "substrate_conflict", "metal_conflict", "n_rows"], ascending=False
    )


def recommendations(summary: Dict, ec_counts: Dict[str, int], sub_counts: Dict[str, int], metal_counts: Dict[str, int], min_count: int) -> List[str]:
    rec = []
    n = summary["n_rows"]
    if n < 200:
        rec.append(f"样本总数仅 {n}，建议优先扩充数据或冻结骨干模型，减少过拟合风险。")

    if summary["missing_rate"]["ec_subclass"] > 0.3:
        rec.append("EC 子类缺失率较高，建议优先补齐 EC 标签，否则主任务监督不足。")
    if summary["missing_rate"]["substrate_labels"] > 0.5:
        rec.append("底物标签缺失率过高，建议增加文献或数据库补注，或先降级为粗粒度底物类别。")
    if summary["missing_rate"]["metal_labels"] > 0.5:
        rec.append("金属依赖标签缺失率过高，建议优先补注“NONE/明确金属依赖”。")

    rare_ec = [k for k, v in ec_counts.items() if v < min_count]
    rare_sub = [k for k, v in sub_counts.items() if v < min_count]
    rare_metal = [k for k, v in metal_counts.items() if v < min_count]

    if rare_ec:
        rec.append(f"EC 小类样本不足（<{min_count}）：{', '.join(rare_ec[:8])}，建议合并近邻类别或继续采样。")
    if rare_sub:
        rec.append(f"底物长尾标签较多（<{min_count}）：{', '.join(rare_sub[:8])}，建议先做 top-K 标签。")
    if rare_metal:
        rec.append(f"金属标签长尾较多（<{min_count}）：{', '.join(rare_metal[:8])}，建议保留主要金属种类。")

    if summary["n_conflicts"] > 0:
        rec.append(f"存在 {summary['n_conflicts']} 条同序列冲突标签，训练前应人工仲裁。")
    if summary["n_suspicious_variants"] > 0:
        rec.append(f"检测到 {summary['n_suspicious_variants']} 组标签拼写变体，建议统一命名后再训练。")

    if not rec:
        rec.append("数据质量整体可用，可直接进入训练并在验证集观察各任务指标。")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input multitask CSV")
    ap.add_argument("--outdir", default="./ligase_label_qc")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--ec-col", default="ec_subclass")
    ap.add_argument("--substrate-col", default="substrate_labels")
    ap.add_argument("--metal-col", default="metal_labels")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--sep", default=";")
    ap.add_argument("--min-class-count", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype=str).fillna("")
    for c in [args.seq_col, args.ec_col, args.substrate_col, args.metal_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # normalize basic fields
    df = df.copy()
    df[args.seq_col] = df[args.seq_col].map(clean_sequence)
    df = df[df[args.seq_col].str.len() > 0].reset_index(drop=True)

    n = len(df)
    seq_len = df[args.seq_col].str.len()

    # missingness
    miss_ec = (df[args.ec_col].str.strip() == "").mean()
    miss_sub = (df[args.substrate_col].str.strip() == "").mean()
    miss_metal = (df[args.metal_col].str.strip() == "").mean()

    # EC validity
    ec_nonempty = df[args.ec_col].str.strip() != ""
    ec_invalid = (~df.loc[ec_nonempty, args.ec_col].map(ec_subclass_ok)).sum()
    ec_counts_series = df.loc[ec_nonempty, args.ec_col].value_counts()
    ec_counts = ec_counts_series.to_dict()

    # multi-label counts
    sub_counts = count_tokens(df[args.substrate_col], sep=args.sep)
    metal_counts = count_tokens(df[args.metal_col], sep=args.sep)

    # variant detection
    all_sub_tokens = []
    all_metal_tokens = []
    for v in df[args.substrate_col].tolist():
        all_sub_tokens.extend(parse_multilabel(v, sep=args.sep))
    for v in df[args.metal_col].tolist():
        all_metal_tokens.extend(parse_multilabel(v, sep=args.sep))
    sub_var = build_variant_table(all_sub_tokens)
    metal_var = build_variant_table(all_metal_tokens)
    suspicious_variants = pd.concat(
        [
            sub_var.assign(field="substrate"),
            metal_var.assign(field="metal"),
        ],
        ignore_index=True,
    ) if (len(sub_var) + len(metal_var)) > 0 else pd.DataFrame(columns=["field", "canonical", "variants", "n_variants", "total_count"])

    # duplicates / conflicts
    n_dup_seq = int(df.duplicated(subset=[args.seq_col]).sum())
    n_dup_id = 0
    if args.id_col in df.columns:
        n_dup_id = int(df[args.id_col].str.strip().replace("", np.nan).dropna().duplicated().sum())
    conflicts = conflict_records(df, args.seq_col, args.ec_col, args.substrate_col, args.metal_col, sep=args.sep)

    # row coverage
    coverage = pd.DataFrame(
        {
            "has_ec": (df[args.ec_col].str.strip() != "").astype(int),
            "has_substrate": (df[args.substrate_col].str.strip() != "").astype(int),
            "has_metal": (df[args.metal_col].str.strip() != "").astype(int),
        }
    )
    coverage["n_tasks_labeled"] = coverage.sum(axis=1)

    summary = {
        "n_rows": int(n),
        "sequence_length": {
            "min": int(seq_len.min()) if n else 0,
            "p25": float(seq_len.quantile(0.25)) if n else 0.0,
            "median": float(seq_len.median()) if n else 0.0,
            "p75": float(seq_len.quantile(0.75)) if n else 0.0,
            "max": int(seq_len.max()) if n else 0,
            "mean": float(seq_len.mean()) if n else 0.0,
        },
        "missing_rate": {
            "ec_subclass": float(miss_ec),
            "substrate_labels": float(miss_sub),
            "metal_labels": float(miss_metal),
        },
        "ec_invalid_format_count": int(ec_invalid),
        "n_ec_classes": int(len(ec_counts)),
        "n_substrate_classes": int(len(sub_counts)),
        "n_metal_classes": int(len(metal_counts)),
        "n_duplicate_sequences": int(n_dup_seq),
        "n_duplicate_ids": int(n_dup_id),
        "n_conflicts": int(len(conflicts)),
        "n_suspicious_variants": int(len(suspicious_variants)),
    }

    recs = recommendations(summary, ec_counts, sub_counts, metal_counts, min_count=args.min_class_count)

    # save outputs
    with open(outdir / "qc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame([{"label": k, "count": v} for k, v in ec_counts.items()]).to_csv(outdir / "ec_counts.csv", index=False)
    pd.DataFrame([{"label": k, "count": v} for k, v in sub_counts.items()]).to_csv(outdir / "substrate_counts.csv", index=False)
    pd.DataFrame([{"label": k, "count": v} for k, v in metal_counts.items()]).to_csv(outdir / "metal_counts.csv", index=False)

    suspicious_variants.to_csv(outdir / "suspicious_variants.csv", index=False)
    conflicts.to_csv(outdir / "sequence_conflicts.csv", index=False)
    coverage.to_csv(outdir / "row_coverage.csv", index=False)

    lines = []
    lines.append("# Label QC Recommendations")
    lines.append("")
    lines.append(f"- Dataset rows: **{summary['n_rows']}**")
    lines.append(f"- EC classes: **{summary['n_ec_classes']}** | Substrate classes: **{summary['n_substrate_classes']}** | Metal classes: **{summary['n_metal_classes']}**")
    lines.append(f"- Missing rates: EC={summary['missing_rate']['ec_subclass']:.2%}, Substrate={summary['missing_rate']['substrate_labels']:.2%}, Metal={summary['missing_rate']['metal_labels']:.2%}")
    lines.append("")
    lines.append("## Action Items")
    for r in recs:
        lines.append(f"- {r}")
    (outdir / "recommendations.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[Done] QC report saved to: {outdir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
