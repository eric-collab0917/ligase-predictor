#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build training CSV for ligase optimal pH & temperature prediction.

Data sources:
1) UniProt EC 6.* entries — extract pH/temperature from catalytic activity
   and function annotations via regex heuristics.
2) Optional: local BRENDA export with explicit pH_optimum / temperature_optimum.
3) Optional: manual overrides CSV.

Output columns:
  id, sequence, opt_ph, opt_temp, ec_subclass, source, accession, organism, notes

Usage:
  python scripts/build_ligase_ph_temp_dataset.py \
      --fetch-uniprot \
      --out-csv data/processed/ligase_ph_temp_auto.csv \
      --report-json data/processed/ligase_ph_temp_auto_report.json
"""

import argparse
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# UniProt API
# ---------------------------------------------------------------------------
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
DEFAULT_QUERY = "(reviewed:true) AND (ec:6.*) AND (cc_biophysicochemical_properties:*)"
DEFAULT_FIELDS = [
    "accession",
    "id",
    "protein_name",
    "organism_name",
    "ec",
    "cc_catalytic_activity",
    "cc_cofactor",
    "cc_function",
    "ft_act_site",
    "temp_dependence",
    "ph_dependence",
    "sequence",
]


def clean_sequence(seq: str) -> str:
    s = re.sub(r"\s+", "", str(seq)).upper()
    if not s:
        return ""
    if re.search(r"[^ACDEFGHIKLMNPQRSTVWYBXZUOJ]", s):
        return ""
    return s


def _parse_next_link(headers) -> Optional[str]:
    link = headers.get("Link", "")
    if not link:
        return None
    m = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
    return m.group(1) if m else None


def fetch_uniprot(
    query: str,
    fields: List[str],
    timeout: int = 180,
    method: str = "search",
    page_size: int = 500,
    max_rows: int = 5000,
) -> pd.DataFrame:
    if method == "stream":
        params = {"query": query, "format": "tsv", "fields": ",".join(fields)}
        r = requests.get(UNIPROT_STREAM, params=params, timeout=timeout)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text), sep="\t")

    params = {
        "query": query,
        "format": "tsv",
        "fields": ",".join(fields),
        "size": int(page_size),
    }
    url = UNIPROT_SEARCH
    chunks = []
    total = 0
    while url:
        r = requests.get(
            url,
            params=params if url == UNIPROT_SEARCH else None,
            timeout=timeout,
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="\t")
        chunks.append(df)
        total += len(df)
        print(f"[Info] UniProt fetched rows: {total}")
        if max_rows and total >= max_rows:
            break
        url = _parse_next_link(r.headers)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    if max_rows and len(out) > max_rows:
        out = out.iloc[:max_rows].copy()
    return out


# ---------------------------------------------------------------------------
# pH / Temperature extraction heuristics
# ---------------------------------------------------------------------------

# Patterns for extracting numeric pH values from text
PH_PATTERNS = [
    # "Optimum pH is 7.5", "Optimum pH is 7.0-8.0"
    r"optim(?:um|al)\s+ph\s+(?:is\s+)?(\d+\.?\d*)\s*[-–-]\s*(\d+\.?\d*)",
    r"optim(?:um|al)\s+ph\s+(?:is\s+)?(\d+\.?\d*)",
    # "pH optimum of 7.5", "pH optimum: 7.5"
    r"ph\s+optim(?:um|al)\s+(?:of\s+|:\s*|is\s+)?(\d+\.?\d*)",
    # "pH optimum at 7.5"
    r"ph\s+optim(?:um|al)\s+at\s+(\d+\.?\d*)",
    # "maximum activity at pH 7.5"
    r"maxim(?:um|al)\s+activity\s+at\s+ph\s+(\d+\.?\d*)",
    # "most active at pH 7.5"
    r"most\s+active\s+at\s+ph\s+(\d+\.?\d*)",
    # "pH 7.0-8.0" → take midpoint
    r"ph\s+(\d+\.?\d*)\s*[-–-]\s*(\d+\.?\d*)",
    # "at pH 7.5"  (weaker, used as fallback)
    r"at\s+ph\s+(\d+\.?\d*)",
]

TEMP_PATTERNS = [
    # "Optimum temperature is 37 degrees Celsius"
    r"optim(?:um|al)\s+temperature\s+(?:is\s+)?(\d+\.?\d*)\s*(?:degrees?\s*)?(?:celsius|c\b)",
    # "optimum temperature 37°C", "optimal temperature of 37 °C"
    r"optim(?:um|al)\s+temperature\s+(?:of\s+)?(\d+\.?\d*)\s*°?\s*C",
    # "temperature optimum of 37°C"
    r"temperature\s+optim(?:um|al)\s+(?:of\s+|:\s*|is\s+)?(\d+\.?\d*)\s*(?:degrees?\s*)?(?:celsius|°?\s*C)",
    # "temperature optimum at 37°C"
    r"temperature\s+optim(?:um|al)\s+at\s+(\d+\.?\d*)\s*(?:degrees?\s*)?(?:celsius|°?\s*C)",
    # "maximum activity at 37°C"
    r"maxim(?:um|al)\s+activity\s+at\s+(\d+\.?\d*)\s*(?:degrees?\s*)?(?:celsius|°?\s*C)",
    # "most active at 37°C"
    r"most\s+active\s+at\s+(\d+\.?\d*)\s*(?:degrees?\s*)?(?:celsius|°?\s*C)",
    # "37-45 degrees Celsius" → take midpoint
    r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(?:degrees?\s*)?(?:celsius|°?\s*C)",
    # "37 degrees Celsius"
    r"(\d+\.?\d*)\s*(?:degrees?\s*)?(?:celsius|°?\s*C)",
]


def _extract_ph(text: str) -> Optional[float]:
    """Extract optimal pH from annotation text."""
    if not text:
        return None
    t = str(text).lower()
    for pat in PH_PATTERNS:
        m = re.search(pat, t)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                # range → midpoint
                lo, hi = float(groups[0]), float(groups[1])
                if 0 <= lo <= 14 and 0 <= hi <= 14:
                    return round((lo + hi) / 2, 2)
            else:
                val = float(groups[0])
                if 0 <= val <= 14:
                    return round(val, 2)
    return None


def _extract_temp(text: str) -> Optional[float]:
    """Extract optimal temperature (°C) from annotation text."""
    if not text:
        return None
    t = str(text).lower()
    for pat in TEMP_PATTERNS:
        m = re.search(pat, t)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                lo, hi = float(groups[0]), float(groups[1])
                if 0 <= lo <= 120 and 0 <= hi <= 120:
                    return round((lo + hi) / 2, 1)
            else:
                val = float(groups[0])
                if 0 <= val <= 120:
                    return round(val, 1)
    return None


def normalize_ec_subclass(ec_text: str) -> str:
    t = str(ec_text or "")
    ecs = re.findall(
        r"\b(\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+\.-|\d+\.\d+\.\-.\-|\d+\.\d+)\b", t
    )
    for ec in ecs:
        parts = ec.split(".")
        if len(parts) >= 2 and parts[0] == "6":
            return f"{parts[0]}.{parts[1]}"
    return ""


def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in low:
            return low[x.lower()]
    return None


def coalesce_text(*vals) -> str:
    out = []
    for v in vals:
        if v is None:
            continue
        t = str(v).strip()
        if t and t.lower() not in {"nan", "none", "null"}:
            out.append(t)
    return " | ".join(out)


# ---------------------------------------------------------------------------
# Parse UniProt dataframe
# ---------------------------------------------------------------------------
def parse_uniprot_df(df: pd.DataFrame) -> pd.DataFrame:
    col_acc = first_col(df, ["accession", "entry"])
    col_id = first_col(df, ["id", "entry name"])
    col_org = first_col(df, ["organism_name", "organism"])
    col_seq = first_col(df, ["sequence"])
    col_ec = first_col(df, ["ec", "ec number"])
    col_pn = first_col(df, ["protein_name", "protein names"])
    col_cat = first_col(df, ["cc_catalytic_activity", "catalytic activity"])
    col_cof = first_col(df, ["cc_cofactor", "cofactor"])
    col_fun = first_col(df, ["cc_function", "function [cc]"])
    col_temp_dep = first_col(df, ["temp_dependence", "temperature dependence"])
    col_ph_dep = first_col(df, ["ph_dependence", "ph dependence"])

    if col_seq is None:
        raise ValueError("UniProt dataframe does not contain sequence column.")

    rows = []
    for _, r in df.iterrows():
        seq = clean_sequence(r.get(col_seq, ""))
        if not seq:
            continue
        ec_sub = normalize_ec_subclass(r.get(col_ec, ""))

        # Combine all text fields for pH/temp extraction
        ph_text = coalesce_text(
            r.get(col_ph_dep, ""),
            r.get(col_cat, ""),
            r.get(col_fun, ""),
        )
        temp_text = coalesce_text(
            r.get(col_temp_dep, ""),
            r.get(col_cat, ""),
            r.get(col_fun, ""),
        )

        opt_ph = _extract_ph(ph_text)
        opt_temp = _extract_temp(temp_text)

        # Skip entries with neither pH nor temperature
        if opt_ph is None and opt_temp is None:
            continue

        rows.append(
            {
                "id": str(
                    r.get(col_id, r.get(col_acc, ""))
                ).strip()
                or str(r.get(col_acc, "")).strip(),
                "sequence": seq,
                "opt_ph": opt_ph if opt_ph is not None else "",
                "opt_temp": opt_temp if opt_temp is not None else "",
                "ec_subclass": ec_sub,
                "source": "UniProt",
                "accession": str(r.get(col_acc, "")).strip(),
                "organism": str(r.get(col_org, "")).strip(),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parse BRENDA export
# ---------------------------------------------------------------------------
def parse_brenda_export(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(p, sep="\t", dtype=str)
    except Exception:
        df = pd.read_csv(p, dtype=str)

    col_seq = first_col(df, ["sequence", "aa sequence", "protein sequence"])
    col_id = first_col(
        df, ["id", "entry", "uniprot", "uniprot id", "uniprot accession"]
    )
    col_acc = first_col(
        df, ["accession", "uniprot", "uniprot id", "uniprot accession"]
    )
    col_org = first_col(df, ["organism", "organism name"])
    col_ec = first_col(df, ["ec", "ec number"])
    col_ph = first_col(df, ["ph_optimum", "ph optimum", "optimal ph", "opt_ph"])
    col_temp = first_col(
        df,
        [
            "temperature_optimum",
            "temperature optimum",
            "optimal temperature",
            "opt_temp",
        ],
    )
    col_comment = first_col(df, ["comment", "comments", "description"])

    rows = []
    for _, r in df.iterrows():
        seq = clean_sequence(r.get(col_seq, "")) if col_seq else ""
        if not seq:
            continue
        ec_sub = normalize_ec_subclass(r.get(col_ec, ""))

        # Try explicit columns first
        opt_ph = None
        opt_temp = None
        if col_ph:
            try:
                val = float(r.get(col_ph, ""))
                if 0 <= val <= 14:
                    opt_ph = round(val, 2)
            except (ValueError, TypeError):
                pass
        if col_temp:
            try:
                val = float(r.get(col_temp, ""))
                if 0 <= val <= 120:
                    opt_temp = round(val, 1)
            except (ValueError, TypeError):
                pass

        # Fallback: extract from comment text
        if opt_ph is None and col_comment:
            opt_ph = _extract_ph(str(r.get(col_comment, "")))
        if opt_temp is None and col_comment:
            opt_temp = _extract_temp(str(r.get(col_comment, "")))

        if opt_ph is None and opt_temp is None:
            continue

        rid = ""
        if col_id:
            rid = str(r.get(col_id, "")).strip()
        if not rid and col_acc:
            rid = str(r.get(col_acc, "")).strip()
        if not rid:
            rid = f"BRENDA_{len(rows) + 1}"

        rows.append(
            {
                "id": rid,
                "sequence": seq,
                "opt_ph": opt_ph if opt_ph is not None else "",
                "opt_temp": opt_temp if opt_temp is not None else "",
                "ec_subclass": ec_sub,
                "source": "BRENDA",
                "accession": str(r.get(col_acc, "")).strip() if col_acc else "",
                "organism": str(r.get(col_org, "")).strip() if col_org else "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Merge & dedup
# ---------------------------------------------------------------------------
def merge_and_dedup(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "id", "sequence", "opt_ph", "opt_temp",
                "ec_subclass", "source", "accession", "organism", "notes",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    for c in ["id", "sequence", "ec_subclass", "source", "accession", "organism", "notes"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    for c in ["opt_ph", "opt_temp"]:
        if c not in df.columns:
            df[c] = ""

    # Dedup by sequence: keep first non-empty value for numeric fields
    def agg_numeric_first(vals):
        for v in vals:
            try:
                fv = float(v)
                if not np.isnan(fv):
                    return fv
            except (ValueError, TypeError):
                continue
        return ""

    def agg_first_nonempty(vals):
        for v in vals:
            t = str(v).strip()
            if t:
                return t
        return ""

    def agg_labels(vals):
        all_labels = set()
        for v in vals:
            for x in str(v).split(";"):
                x = x.strip()
                if x:
                    all_labels.add(x)
        return ";".join(sorted(all_labels))

    df["sequence"] = df["sequence"].map(clean_sequence)
    df = df[df["sequence"].str.len() > 0].copy()

    grouped = (
        df.groupby("sequence", as_index=False)
        .agg(
            {
                "id": agg_first_nonempty,
                "opt_ph": agg_numeric_first,
                "opt_temp": agg_numeric_first,
                "ec_subclass": agg_first_nonempty,
                "source": agg_labels,
                "accession": agg_first_nonempty,
                "organism": agg_first_nonempty,
                "notes": agg_labels,
            }
        )
        .reset_index(drop=True)
    )
    return grouped


# ---------------------------------------------------------------------------
# Manual overrides
# ---------------------------------------------------------------------------
def apply_manual_overrides(df: pd.DataFrame, manual_csv: str) -> pd.DataFrame:
    md = pd.read_csv(manual_csv, dtype=str).fillna("")
    for c in ["id", "accession", "sequence", "opt_ph", "opt_temp", "ec_subclass", "notes"]:
        if c not in md.columns:
            md[c] = ""
    md["sequence"] = md["sequence"].map(clean_sequence)

    idx_by_acc = {str(v): i for i, v in enumerate(df["accession"].tolist()) if str(v).strip()}
    idx_by_seq = {str(v): i for i, v in enumerate(df["sequence"].tolist()) if str(v).strip()}

    appended = []
    for _, r in md.iterrows():
        racc = str(r["accession"]).strip()
        rseq = str(r["sequence"]).strip()
        candidate_idx = None
        if racc and racc in idx_by_acc:
            candidate_idx = idx_by_acc[racc]
        elif rseq and rseq in idx_by_seq:
            candidate_idx = idx_by_seq[rseq]

        if candidate_idx is None:
            if not rseq:
                continue
            appended.append(
                {
                    "id": str(r["id"]).strip() or f"manual_{len(df) + len(appended) + 1}",
                    "sequence": rseq,
                    "opt_ph": str(r["opt_ph"]).strip(),
                    "opt_temp": str(r["opt_temp"]).strip(),
                    "ec_subclass": str(r["ec_subclass"]).strip(),
                    "source": "manual",
                    "accession": racc,
                    "organism": "",
                    "notes": str(r["notes"]).strip(),
                }
            )
        else:
            for col in ["opt_ph", "opt_temp", "ec_subclass", "notes"]:
                v = str(r[col]).strip()
                if v:
                    df.at[candidate_idx, col] = v
            src = str(df.at[candidate_idx, "source"]).strip()
            df.at[candidate_idx, "source"] = f"{src};manual" if src else "manual"

    if appended:
        df = pd.concat([df, pd.DataFrame(appended)], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> Dict:
    def safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    ph_vals = [safe_float(v) for v in df["opt_ph"] if safe_float(v) is not None]
    temp_vals = [safe_float(v) for v in df["opt_temp"] if safe_float(v) is not None]

    return {
        "n_rows": int(len(df)),
        "n_with_ph": len(ph_vals),
        "n_with_temp": len(temp_vals),
        "n_with_both": int(
            sum(
                1
                for _, r in df.iterrows()
                if safe_float(r["opt_ph"]) is not None
                and safe_float(r["opt_temp"]) is not None
            )
        ),
        "ph_stats": {
            "mean": round(float(np.mean(ph_vals)), 2) if ph_vals else None,
            "std": round(float(np.std(ph_vals)), 2) if ph_vals else None,
            "min": round(float(np.min(ph_vals)), 2) if ph_vals else None,
            "max": round(float(np.max(ph_vals)), 2) if ph_vals else None,
        },
        "temp_stats": {
            "mean": round(float(np.mean(temp_vals)), 1) if temp_vals else None,
            "std": round(float(np.std(temp_vals)), 1) if temp_vals else None,
            "min": round(float(np.min(temp_vals)), 1) if temp_vals else None,
            "max": round(float(np.max(temp_vals)), 1) if temp_vals else None,
        },
        "ec_distribution": (
            df["ec_subclass"]
            .replace("", np.nan)
            .dropna()
            .value_counts()
            .to_dict()
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build ligase optimal pH & temperature dataset"
    )
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--report-json", default="", help="Optional summary report JSON")

    ap.add_argument("--fetch-uniprot", action="store_true")
    ap.add_argument("--uniprot-query", default=DEFAULT_QUERY)
    ap.add_argument("--uniprot-fields", default=",".join(DEFAULT_FIELDS))
    ap.add_argument(
        "--uniprot-method", choices=["search", "stream"], default="search"
    )
    ap.add_argument("--uniprot-page-size", type=int, default=500)
    ap.add_argument(
        "--uniprot-max-rows",
        type=int,
        default=5000,
        help="0 means no limit",
    )
    ap.add_argument(
        "--save-uniprot-raw", default="", help="Optional path to save raw UniProt TSV"
    )

    ap.add_argument("--brenda-export", default="", help="Local BRENDA export TSV/CSV")
    ap.add_argument("--manual-overrides", default="", help="Manual override CSV")
    args = ap.parse_args()

    frames = []

    if args.fetch_uniprot:
        fields = [x.strip() for x in args.uniprot_fields.split(",") if x.strip()]
        print(f"[Info] Fetching UniProt with query: {args.uniprot_query}")
        udf = fetch_uniprot(
            args.uniprot_query,
            fields,
            method=args.uniprot_method,
            page_size=args.uniprot_page_size,
            max_rows=args.uniprot_max_rows,
        )
        if args.save_uniprot_raw:
            Path(args.save_uniprot_raw).parent.mkdir(parents=True, exist_ok=True)
            udf.to_csv(args.save_uniprot_raw, sep="\t", index=False)
            print(f"[Info] Saved UniProt raw to: {args.save_uniprot_raw}")
        parsed = parse_uniprot_df(udf)
        frames.append(parsed)
        print(f"[Info] Parsed UniProt records with pH/temp: {len(parsed)}")

    if args.brenda_export:
        bdf = parse_brenda_export(args.brenda_export)
        frames.append(bdf)
        print(f"[Info] Parsed BRENDA records with pH/temp: {len(bdf)}")

    if not frames:
        raise ValueError(
            "No input data. Use --fetch-uniprot and/or --brenda-export"
        )

    merged = merge_and_dedup(frames)
    print(f"[Info] After merge+dedup: {len(merged)}")

    if args.manual_overrides:
        merged = apply_manual_overrides(merged, args.manual_overrides)
        merged = merge_and_dedup([merged])
        print(f"[Info] After manual overrides: {len(merged)}")

    out_cols = [
        "id", "sequence", "opt_ph", "opt_temp",
        "ec_subclass", "source", "accession", "organism", "notes",
    ]
    for c in out_cols:
        if c not in merged.columns:
            merged[c] = ""
    merged = merged[out_cols].copy()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[Done] Saved dataset to: {args.out_csv}")

    rep = summarize(merged)
    print("[Done] Summary:")
    print(json.dumps(rep, ensure_ascii=False, indent=2))

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[Done] Saved report to: {args.report_json}")


if __name__ == "__main__":
    main()
