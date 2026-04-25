#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare train-ready multitask CSV by normalizing/merging long-tail labels.

Main features:
- Keep top-K frequent labels (or labels above min count)
- Merge rare labels into OTHER (optional)
- Optionally fill empty multi-label cells with NONE
- Keep columns compatible with train_ligase_multitask.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_labels(v: str, sep=";") -> List[str]:
    if v is None:
        return []
    t = str(v).strip()
    if not t or t.lower() in {"nan", "null"}:
        return []
    return [x.strip() for x in t.split(sep) if x.strip()]


def count_labels(series: pd.Series, sep=";") -> Dict[str, int]:
    c = {}
    for v in series.fillna("").astype(str):
        for t in parse_labels(v, sep=sep):
            c[t] = c.get(t, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def choose_kept_labels(counts: Dict[str, int], top_k: int, min_count: int, keep_explicit: List[str]) -> List[str]:
    items = list(counts.items())
    kept = [k for k, v in items if v >= min_count]
    if top_k > 0:
        kept = list(dict.fromkeys(kept + [k for k, _ in items[:top_k]]))
    kept = list(dict.fromkeys(keep_explicit + kept))
    return kept


def remap_multilabel_cell(v: str, keep: List[str], sep=";", rare_to_other=True, other_label="OTHER", fill_none=False):
    tokens = parse_labels(v, sep=sep)
    if not tokens:
        return "NONE" if fill_none else ""

    out = []
    rare_seen = False
    for t in tokens:
        if t in keep:
            out.append(t)
        else:
            rare_seen = True
    if rare_to_other and rare_seen:
        out.append(other_label)
    out = sorted(set(out))
    if not out and fill_none:
        return "NONE"
    return sep.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--report-json", default="")

    ap.add_argument("--substrate-col", default="substrate_labels")
    ap.add_argument("--metal-col", default="metal_labels")
    ap.add_argument("--sep", default=";")

    ap.add_argument("--substrate-top-k", type=int, default=0, help="0 means disabled")
    ap.add_argument("--substrate-min-count", type=int, default=8)
    ap.add_argument("--metal-top-k", type=int, default=4)
    ap.add_argument("--metal-min-count", type=int, default=8)

    ap.add_argument("--substrate-rare-to-other", action="store_true", default=True)
    ap.add_argument("--metal-rare-to-other", action="store_true", default=True)
    ap.add_argument("--no-substrate-rare-to-other", dest="substrate_rare_to_other", action="store_false")
    ap.add_argument("--no-metal-rare-to-other", dest="metal_rare_to_other", action="store_false")

    ap.add_argument("--fill-empty-substrate-none", action="store_true", default=False)
    ap.add_argument("--fill-empty-metal-none", action="store_true", default=False)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, dtype=str).fillna("")
    for col in [args.substrate_col, args.metal_col]:
        if col not in df.columns:
            raise KeyError(f"Column not found: {col}")

    sub_counts_before = count_labels(df[args.substrate_col], sep=args.sep)
    metal_counts_before = count_labels(df[args.metal_col], sep=args.sep)

    keep_sub = choose_kept_labels(
        sub_counts_before,
        top_k=args.substrate_top_k,
        min_count=args.substrate_min_count,
        keep_explicit=["NONE"],
    )
    keep_metal = choose_kept_labels(
        metal_counts_before,
        top_k=args.metal_top_k,
        min_count=args.metal_min_count,
        keep_explicit=["NONE"],
    )

    out = df.copy()
    out[args.substrate_col] = out[args.substrate_col].map(
        lambda x: remap_multilabel_cell(
            x,
            keep=keep_sub,
            sep=args.sep,
            rare_to_other=args.substrate_rare_to_other,
            other_label="OTHER",
            fill_none=args.fill_empty_substrate_none,
        )
    )
    out[args.metal_col] = out[args.metal_col].map(
        lambda x: remap_multilabel_cell(
            x,
            keep=keep_metal,
            sep=args.sep,
            rare_to_other=args.metal_rare_to_other,
            other_label="OTHER",
            fill_none=args.fill_empty_metal_none,
        )
    )

    sub_counts_after = count_labels(out[args.substrate_col], sep=args.sep)
    metal_counts_after = count_labels(out[args.metal_col], sep=args.sep)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    report = {
        "n_rows": int(len(out)),
        "substrate": {
            "kept_labels": keep_sub,
            "before_counts": sub_counts_before,
            "after_counts": sub_counts_after,
            "fill_empty_none": args.fill_empty_substrate_none,
            "rare_to_other": args.substrate_rare_to_other,
        },
        "metal": {
            "kept_labels": keep_metal,
            "before_counts": metal_counts_before,
            "after_counts": metal_counts_after,
            "fill_empty_none": args.fill_empty_metal_none,
            "rare_to_other": args.metal_rare_to_other,
        },
    }

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved train-ready CSV: {args.out_csv}")
    if args.report_json:
        print(f"[Done] Saved report: {args.report_json}")
    print(json.dumps(report["metal"]["after_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

