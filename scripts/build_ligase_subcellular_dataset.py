#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build training CSV for ligase subcellular localization prediction.

Data sources:
1) UniProt EC 6.* entries — extract subcellular location from
   cc_subcellular_location annotations via regex heuristics.
2) Optional: manual overrides CSV.

Output columns:
  id, sequence, subcellular_labels, ec_subclass, source, accession, organism, notes

Usage:
  python scripts/build_ligase_subcellular_dataset.py \
      --fetch-uniprot \
      --out-csv data/processed/ligase_subcellular_auto.csv \
      --report-json data/processed/ligase_subcellular_auto_report.json
"""

import argparse
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# UniProt API
# ---------------------------------------------------------------------------
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
DEFAULT_QUERY = "(reviewed:true) AND (ec:6.*)"
DEFAULT_FIELDS = [
    "accession",
    "id",
    "protein_name",
    "organism_name",
    "ec",
    "cc_subcellular_location",
    "sequence",
]

# ---------------------------------------------------------------------------
# Subcellular location heuristic rules
# Each rule: (regex_pattern, label_name)
# Order matters: first-match wins for ambiguous text
# ---------------------------------------------------------------------------
SUBCELLULAR_RULES = [
    (r"\bcytoplasm\b", "Cytoplasm"),
    (r"\bnucleus\b|\bnuclear\b|\bnucleoplasm\b|\bnucleolus\b", "Nucleus"),
    (r"\bmitochondr|\bmitochondrial\b", "Mitochondrion"),
    (r"\bendoplasmic reticulum\b|\ber\b|\bsarcoplasmic reticulum\b", "Endoplasmic reticulum"),
    (r"\bgolgi\b|\bgolgi apparatus\b", "Golgi apparatus"),
    (r"\bperoxisom\b|\bglyoxysom\b", "Peroxisome"),
    (r"\blysosom\b|\bvacuol\b", "Lysosome/Vacuole"),
    (r"\bcell membrane\b|\bplasma membrane\b|\bmembrane\b|\bcytoplasmic membrane\b", "Cell membrane"),
    (r"\bsecreted\b|\bextracellular\b|\bextracell\b|\bexcreted\b|\bsecretory\b", "Extracellular/Secreted"),
    (r"\bchloroplast\b|\bplastid\b", "Chloroplast"),
    (r"\bcytoskeleton\b|\bmicrotubul\b|\bactin\b|\bfilament\b", "Cytoskeleton"),
    (r"\bcell wall\b|\bcell surface\b", "Cell wall"),
    (r"\bchromosome\b|\bchromatin\b|\bcentromer\b", "Chromosome"),
    (r"\bribosome\b", "Ribosome"),
    (r"\bintegral component of membrane\b|\btransmembrane\b", "Integral membrane"),
]


def clean_sequence(seq: str) -> str:
    s = re.sub(r"\s+", "", str(seq)).upper()
    if not s:
        return ""
    if re.search(r"[^ACDEFGHIKLMNPQRSTVWYBXZUOJ]", s):
        return ""
    return s


def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in low:
            return low[x.lower()]
    return None


def normalize_ec_subclass(ec_text: str) -> str:
    t = str(ec_text or "")
    ecs = re.findall(r"\b(\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+\.-|\d+\.\d+\.\-.\-|\d+\.\d+)\b", t)
    for ec in ecs:
        parts = ec.split(".")
        if len(parts) >= 2 and parts[0] == "6":
            return f"{parts[0]}.{parts[1]}"
    return ""


def extract_subcellular_labels(text: str) -> List[str]:
    """Apply regex rules to extract subcellular location labels from text."""
    if not text or str(text).strip().lower() in {"", "nan", "none", "null"}:
        return []
    tt = str(text).lower()
    labels: Set[str] = set()
    for pattern, label in SUBCELLULAR_RULES:
        if re.search(pattern, tt):
            labels.add(label)
    return sorted(labels)


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
        r = requests.get(url, params=params if url == UNIPROT_SEARCH else None, timeout=timeout)
        r.raise_for_status()
        txt = r.text
        df = pd.read_csv(io.StringIO(txt), sep="\t")
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


def parse_uniprot_df(df: pd.DataFrame) -> pd.DataFrame:
    col_acc = first_col(df, ["accession", "entry"])
    col_id = first_col(df, ["id", "entry name"])
    col_org = first_col(df, ["organism_name", "organism"])
    col_seq = first_col(df, ["sequence"])
    col_ec = first_col(df, ["ec", "ec number"])
    col_subcell = first_col(df, ["cc_subcellular_location", "subcellular location [cc]"])

    if col_seq is None:
        raise ValueError("UniProt dataframe does not contain sequence column.")

    rows = []
    for _, r in df.iterrows():
        seq = clean_sequence(r.get(col_seq, ""))
        if not seq:
            continue
        ec_sub = normalize_ec_subclass(r.get(col_ec, ""))
        subcell_text = str(r.get(col_subcell, "")) if col_subcell else ""
        subcell = extract_subcellular_labels(subcell_text)

        rows.append(
            {
                "id": str(r.get(col_id, r.get(col_acc, ""))).strip()
                or str(r.get(col_acc, "")).strip(),
                "sequence": seq,
                "subcellular_labels": ";".join(subcell),
                "ec_subclass": ec_sub,
                "source": "UniProt",
                "accession": str(r.get(col_acc, "")).strip(),
                "organism": str(r.get(col_org, "")).strip(),
                "notes": subcell_text[:200] if subcell_text else "",
            }
        )
    return pd.DataFrame(rows)


def merge_and_dedup(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "id", "sequence", "subcellular_labels", "ec_subclass",
                "source", "accession", "organism", "notes",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["accession"], keep="first").reset_index(drop=True)
    df = df.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    df = df[df["sequence"].notna() & (df["sequence"] != "")]
    return df


def apply_manual_overrides(df: pd.DataFrame, overrides_path: Optional[str]) -> pd.DataFrame:
    if not overrides_path:
        return df
    p = Path(overrides_path)
    if not p.exists():
        print(f"[Warning] Manual overrides file not found: {overrides_path}")
        return df
    ov = pd.read_csv(p, dtype=str)
    ov_cols = {c.lower(): c for c in ov.columns}
    # find matching columns
    id_cols = [c for c in ["id", "accession", "sequence"] if c in ov_cols]
    sub_col = next((c for c in ov.columns if "subcellular" in c.lower()), None)
    if not id_cols or not sub_col:
        print("[Warning] Override CSV missing required columns (id/accession/sequence + subcellular_labels)")
        return df
    for _, orow in ov.iterrows():
        for idc in id_cols:
            mask = None
            if idc == "id":
                mask = df["id"] == str(orow[idc])
            elif idc == "accession":
                mask = df["accession"] == str(orow[idc])
            elif idc == "sequence":
                seq = clean_sequence(str(orow[idc]))
                if seq:
                    mask = df["sequence"] == seq
            if mask is not None and mask.any():
                new_val = str(orow[sub_col]).strip()
                if new_val and new_val.lower() not in {"nan", "none", "null"}:
                    df.loc[mask, "subcellular_labels"] = new_val
                break
    return df


def generate_report(df: pd.DataFrame, out_json: Optional[str]) -> dict:
    report = {
        "total_entries": len(df),
        "total_with_subcellular": int((df["subcellular_labels"] != "").sum()),
        "total_with_ec": int((df["ec_subclass"] != "").sum()),
    }
    # per-label counts
    label_counts = {}
    for v in df["subcellular_labels"]:
        if not v:
            continue
        for lbl in v.split(";"):
            lbl = lbl.strip()
            if lbl:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
    report["label_counts"] = dict(sorted(label_counts.items(), key=lambda x: -x[1]))

    # per-EC counts
    ec_counts = df["ec_subclass"].value_counts().to_dict()
    report["ec_subclass_counts"] = {str(k): int(v) for k, v in ec_counts.items() if k}

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description="Build ligase subcellular localization dataset")
    parser.add_argument("--fetch-uniprot", action="store_true", help="Fetch from UniProt API")
    parser.add_argument("--uniprot-query", default=DEFAULT_QUERY)
    parser.add_argument("--uniprot-fields", nargs="*", default=DEFAULT_FIELDS)
    parser.add_argument("--uniprot-max-rows", type=int, default=5000)
    parser.add_argument("--brenda-export", help="Optional BRENDA export CSV/TSV")
    parser.add_argument("--manual-overrides", help="Optional manual overrides CSV")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument("--report-json", help="Optional report JSON path")
    args = parser.parse_args()

    frames: List[pd.DataFrame] = []

    if args.fetch_uniprot:
        print("[Step 1] Fetching from UniProt...")
        raw = fetch_uniprot(
            query=args.uniprot_query,
            fields=args.uniprot_fields,
            max_rows=args.uniprot_max_rows,
        )
        if raw.empty:
            print("[Warning] UniProt returned empty results")
        else:
            print(f"[Info] UniProt returned {len(raw)} raw entries")
            parsed = parse_uniprot_df(raw)
            print(f"[Info] Parsed {len(parsed)} valid entries with sequences")
            frames.append(parsed)

    if args.brenda_export:
        print("[Step 2] Processing BRENDA export...")
        bp = Path(args.brenda_export)
        if bp.exists():
            try:
                brenda = pd.read_csv(bp, sep="\t" if bp.suffix == ".tsv" else ",", dtype=str)
            except Exception:
                brenda = pd.read_csv(bp, dtype=str)
            col_seq = first_col(brenda, ["sequence", "aa sequence", "protein sequence"])
            col_subcell = first_col(brenda, [
                "subcellular_location", "subcellular location",
                "localization", "comment",
            ])
            if col_seq and col_subcell:
                rows = []
                for _, r in brenda.iterrows():
                    seq = clean_sequence(r.get(col_seq, ""))
                    if not seq:
                        continue
                    subcell = extract_subcellular_labels(str(r.get(col_subcell, "")))
                    rows.append({
                        "id": f"BRENDA_{len(rows)+1}",
                        "sequence": seq,
                        "subcellular_labels": ";".join(subcell),
                        "ec_subclass": "",
                        "source": "BRENDA",
                        "accession": "",
                        "organism": "",
                        "notes": str(r.get(col_subcell, ""))[:200],
                    })
                if rows:
                    frames.append(pd.DataFrame(rows))
                    print(f"[Info] BRENDA added {len(rows)} entries")
        else:
            print(f"[Warning] BRENDA file not found: {args.brenda_export}")

    if not frames:
        print("[Error] No data sources provided. Use --fetch-uniprot and/or --brenda-export.")
        return

    print("[Step 3] Merging and deduplicating...")
    df = merge_and_dedup(frames)
    print(f"[Info] After dedup: {len(df)} entries")

    if args.manual_overrides:
        print("[Step 4] Applying manual overrides...")
        df = apply_manual_overrides(df, args.manual_overrides)

    print("[Step 5] Generating report...")
    report = generate_report(df, args.report_json)

    print("[Step 6] Saving CSV...")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[Done] Saved {len(df)} entries to {args.out_csv}")
    print(f"  - {report['total_with_subcellular']} have subcellular labels")
    print(f"  - Label counts: {report['label_counts']}")


if __name__ == "__main__":
    main()
