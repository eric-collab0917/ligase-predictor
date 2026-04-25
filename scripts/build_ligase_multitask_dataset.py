#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build training CSV for ligase multitask learning:
- EC subclass (single-label)
- substrate type/spectrum (multi-label)
- metal-ion dependency (multi-label)

Pipeline:
1) Optional: fetch UniProt records via REST API.
2) Optional: merge local exports from BRENDA/SABIO-RK.
3) Heuristic label extraction from text fields.
4) Optional: apply manual overrides.
5) Export CSV ready for train_ligase_multitask.py.

Output columns:
id,sequence,ec_subclass,substrate_labels,metal_labels,source,accession,organism,notes
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


UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
DEFAULT_UNIPROT_QUERY = "(reviewed:true) AND (ec:6.*)"
DEFAULT_UNIPROT_FIELDS = [
    "accession",
    "id",
    "protein_name",
    "organism_name",
    "ec",
    "cc_catalytic_activity",
    "cc_cofactor",
    "cc_function",
    "sequence",
]


SUBSTRATE_RULES = [
    (r"\baminoacyl[- ]?trna\b", "aminoacyl_tRNA"),
    (r"\bdna\b", "DNA_ligation"),
    (r"\brna\b", "RNA_ligation"),
    (r"\bpeptid(e|yl)\b|\bprotein\b", "peptide_or_protein_ligation"),
    (r"\bubiquitin|ubiquityl|sumo\b", "ubiquitin_like_ligation"),
    (r"\bfatty acid|acyl[- ]coa|lipid\b", "lipid_or_fatty_acid"),
    (r"\bpolysaccharide|glycan|glyco\b", "glycan_or_polysaccharide"),
    (r"\bphospholipid|cardiolipin\b", "phospholipid_related"),
    (r"\bglutathione\b", "glutathione_related"),
    (r"\bcofactor|coenzyme\b", "cofactor_attachment"),
]


METAL_RULES = [
    (r"\bmg2\+|\bmagnesium\b", "Mg2+"),
    (r"\bmn2\+|\bmanganese\b", "Mn2+"),
    (r"\bzn2\+|\bzinc\b", "Zn2+"),
    (r"\bca2\+|\bcalcium\b", "Ca2+"),
    (r"\bfe2\+|\biron\(ii\)|\bferrous\b", "Fe2+"),
    (r"\bfe3\+|\biron\(iii\)|\bferric\b", "Fe3+"),
    (r"\bco2\+|\bcobalt\b", "Co2+"),
    (r"\bni2\+|\bnickel\b", "Ni2+"),
    (r"\bcu2\+|\bcopper\b", "Cu2+"),
    (r"\bk\+|\bpotassium\b", "K+"),
    (r"\bna\+|\bsodium\b", "Na+"),
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


def coalesce_text(*vals) -> str:
    out = []
    for v in vals:
        if v is None:
            continue
        t = str(v).strip()
        if t and t.lower() not in {"nan", "none", "null"}:
            out.append(t)
    return " | ".join(out)


def normalize_ec_subclass(ec_text: str) -> str:
    t = str(ec_text or "")
    ecs = re.findall(r"\b(\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+\.-|\d+\.\d+\.\-.\-|\d+\.\d+)\b", t)
    for ec in ecs:
        parts = ec.split(".")
        if len(parts) >= 2 and parts[0] == "6":
            return f"{parts[0]}.{parts[1]}"
    return ""


def extract_multi_labels(text: str, rules: List[Tuple[str, str]]) -> List[str]:
    tt = str(text or "").lower()
    labels: Set[str] = set()
    for pattern, label in rules:
        if re.search(pattern, tt):
            labels.add(label)
    return sorted(labels)


def detect_explicit_none(text: str) -> bool:
    tt = str(text or "").lower()
    return bool(
        re.search(
            r"(not metal dependent|metal independent|no metal required|without metal ions|does not require metal)",
            tt,
        )
    )


def _parse_next_link(headers) -> Optional[str]:
    link = headers.get("Link", "")
    if not link:
        return None
    # format: <url>; rel="next"
    m = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
    if m:
        return m.group(1)
    return None


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

    # paginated search mode (faster feedback + controllable size)
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
    col_pn = first_col(df, ["protein_name", "protein names"])
    col_cat = first_col(df, ["cc_catalytic_activity", "catalytic activity"])
    col_cof = first_col(df, ["cc_cofactor", "cofactor"])
    col_fun = first_col(df, ["cc_function", "function [cc]"])

    if col_seq is None:
        raise ValueError("UniProt dataframe does not contain sequence column.")

    rows = []
    for _, r in df.iterrows():
        seq = clean_sequence(r.get(col_seq, ""))
        if not seq:
            continue
        ec_sub = normalize_ec_subclass(r.get(col_ec, ""))
        info_text = coalesce_text(r.get(col_pn, ""), r.get(col_cat, ""), r.get(col_cof, ""), r.get(col_fun, ""))
        substrate = extract_multi_labels(info_text, SUBSTRATE_RULES)
        metal = extract_multi_labels(info_text, METAL_RULES)
        if len(metal) == 0 and detect_explicit_none(info_text):
            metal = ["NONE"]
        rows.append(
            {
                "id": str(r.get(col_id, r.get(col_acc, ""))).strip() or str(r.get(col_acc, "")).strip(),
                "sequence": seq,
                "ec_subclass": ec_sub,
                "substrate_labels": ";".join(substrate),
                "metal_labels": ";".join(metal),
                "source": "UniProt",
                "accession": str(r.get(col_acc, "")).strip(),
                "organism": str(r.get(col_org, "")).strip(),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def parse_brenda_or_sabio_table(path: str, source_name: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    # try tsv first, fallback csv
    try:
        df = pd.read_csv(p, sep="\t", dtype=str)
    except Exception:
        df = pd.read_csv(p, dtype=str)

    col_seq = first_col(df, ["sequence", "aa sequence", "protein sequence"])
    col_id = first_col(df, ["id", "entry", "uniprot", "uniprot id", "uniprot accession"])
    col_acc = first_col(df, ["accession", "uniprot", "uniprot id", "uniprot accession"])
    col_org = first_col(df, ["organism", "organism name"])
    col_ec = first_col(df, ["ec", "ec number"])
    col_sub = first_col(df, ["substrate", "substrates", "reaction", "reaction equation"])
    col_metal = first_col(df, ["metal", "metals/ions", "cofactor", "modifiers"])
    col_comment = first_col(df, ["comment", "comments", "description"])

    rows = []
    for _, r in df.iterrows():
        seq = clean_sequence(r.get(col_seq, "")) if col_seq else ""
        ec_sub = normalize_ec_subclass(r.get(col_ec, ""))
        text_sub = coalesce_text(r.get(col_sub, ""), r.get(col_comment, ""))
        text_metal = coalesce_text(r.get(col_metal, ""), r.get(col_comment, ""))
        substrate = extract_multi_labels(text_sub, SUBSTRATE_RULES)
        metal = extract_multi_labels(text_metal, METAL_RULES)
        if len(metal) == 0 and detect_explicit_none(text_metal):
            metal = ["NONE"]

        rid = ""
        if col_id:
            rid = str(r.get(col_id, "")).strip()
        if not rid and col_acc:
            rid = str(r.get(col_acc, "")).strip()
        if not rid:
            rid = f"{source_name}_{len(rows)+1}"

        rows.append(
            {
                "id": rid,
                "sequence": seq,
                "ec_subclass": ec_sub,
                "substrate_labels": ";".join(substrate),
                "metal_labels": ";".join(metal),
                "source": source_name,
                "accession": str(r.get(col_acc, "")).strip() if col_acc else "",
                "organism": str(r.get(col_org, "")).strip() if col_org else "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def merge_and_dedup(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "id",
                "sequence",
                "ec_subclass",
                "substrate_labels",
                "metal_labels",
                "source",
                "accession",
                "organism",
                "notes",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    for c in ["id", "sequence", "ec_subclass", "substrate_labels", "metal_labels", "source", "accession", "organism", "notes"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    # Merge duplicates by accession first, then by sequence.
    def agg_labels(vals):
        all_labels = set()
        for v in vals:
            for x in str(v).split(";"):
                x = x.strip()
                if x:
                    all_labels.add(x)
        return ";".join(sorted(all_labels))

    def agg_first_nonempty(vals):
        for v in vals:
            t = str(v).strip()
            if t:
                return t
        return ""

    grouped = []
    key_acc = df["accession"].str.len() > 0
    if key_acc.any():
        g = (
            df[key_acc]
            .groupby("accession", as_index=False)
            .agg(
                {
                    "id": agg_first_nonempty,
                    "sequence": agg_first_nonempty,
                    "ec_subclass": agg_first_nonempty,
                    "substrate_labels": agg_labels,
                    "metal_labels": agg_labels,
                    "source": agg_labels,
                    "organism": agg_first_nonempty,
                    "notes": agg_labels,
                }
            )
        )
        g["accession"] = g["accession"].astype(str)
        grouped.append(g)

    rest = df[~key_acc].copy()
    rest = rest[rest["sequence"].str.len() > 0]
    if len(rest):
        g2 = (
            rest.groupby("sequence", as_index=False)
            .agg(
                {
                    "id": agg_first_nonempty,
                    "ec_subclass": agg_first_nonempty,
                    "substrate_labels": agg_labels,
                    "metal_labels": agg_labels,
                    "source": agg_labels,
                    "accession": agg_first_nonempty,
                    "organism": agg_first_nonempty,
                    "notes": agg_labels,
                }
            )
            .reset_index(drop=True)
        )
        grouped.append(g2)

    if grouped:
        out = pd.concat(grouped, ignore_index=True)
    else:
        out = df.copy()

    # ensure cleaned sequence
    out["sequence"] = out["sequence"].map(clean_sequence)
    out = out[out["sequence"].str.len() > 0].copy()
    out = out.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    return out


def apply_manual_overrides(df: pd.DataFrame, manual_csv: str) -> pd.DataFrame:
    md = pd.read_csv(manual_csv, dtype=str).fillna("")
    for c in ["id", "accession", "sequence", "ec_subclass", "substrate_labels", "metal_labels", "notes"]:
        if c not in md.columns:
            md[c] = ""

    md["sequence"] = md["sequence"].map(clean_sequence)
    md = md[(md["id"].str.len() > 0) | (md["accession"].str.len() > 0) | (md["sequence"].str.len() > 0)].copy()

    # index for fast replace
    idx_by_id = {str(v): i for i, v in enumerate(df["id"].tolist()) if str(v).strip()}
    idx_by_acc = {str(v): i for i, v in enumerate(df["accession"].tolist()) if str(v).strip()}
    idx_by_seq = {str(v): i for i, v in enumerate(df["sequence"].tolist()) if str(v).strip()}

    appended = []
    for _, r in md.iterrows():
        candidate_idx = None
        rid = str(r["id"]).strip()
        racc = str(r["accession"]).strip()
        rseq = str(r["sequence"]).strip()

        if rid and rid in idx_by_id:
            candidate_idx = idx_by_id[rid]
        elif racc and racc in idx_by_acc:
            candidate_idx = idx_by_acc[racc]
        elif rseq and rseq in idx_by_seq:
            candidate_idx = idx_by_seq[rseq]

        if candidate_idx is None:
            if not rseq:
                continue
            appended.append(
                {
                    "id": rid or f"manual_{len(df) + len(appended) + 1}",
                    "sequence": rseq,
                    "ec_subclass": str(r["ec_subclass"]).strip(),
                    "substrate_labels": str(r["substrate_labels"]).strip(),
                    "metal_labels": str(r["metal_labels"]).strip(),
                    "source": "manual",
                    "accession": racc,
                    "organism": "",
                    "notes": str(r["notes"]).strip(),
                }
            )
        else:
            for col in ["ec_subclass", "substrate_labels", "metal_labels", "notes"]:
                v = str(r[col]).strip()
                if v:
                    df.at[candidate_idx, col] = v
            if rid and not str(df.at[candidate_idx, "id"]).strip():
                df.at[candidate_idx, "id"] = rid
            if racc and not str(df.at[candidate_idx, "accession"]).strip():
                df.at[candidate_idx, "accession"] = racc
            src = str(df.at[candidate_idx, "source"]).strip()
            df.at[candidate_idx, "source"] = f"{src};manual" if src else "manual"

    if appended:
        df = pd.concat([df, pd.DataFrame(appended)], ignore_index=True)
    return df


def summarize(df: pd.DataFrame) -> Dict:
    ec_counts = df["ec_subclass"].replace("", np.nan).dropna().value_counts().to_dict()

    def count_multilabel(col):
        c = {}
        for v in df[col].fillna("").astype(str):
            for x in v.split(";"):
                x = x.strip()
                if not x:
                    continue
                c[x] = c.get(x, 0) + 1
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "n_rows": int(len(df)),
        "n_with_ec": int((df["ec_subclass"].str.len() > 0).sum()),
        "n_with_substrate": int((df["substrate_labels"].str.len() > 0).sum()),
        "n_with_metal": int((df["metal_labels"].str.len() > 0).sum()),
        "ec_distribution": ec_counts,
        "substrate_distribution": count_multilabel("substrate_labels"),
        "metal_distribution": count_multilabel("metal_labels"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True, help="Final dataset csv")
    ap.add_argument("--report-json", default="", help="Optional summary report json")

    ap.add_argument("--fetch-uniprot", action="store_true")
    ap.add_argument("--uniprot-query", default=DEFAULT_UNIPROT_QUERY)
    ap.add_argument("--uniprot-fields", default=",".join(DEFAULT_UNIPROT_FIELDS))
    ap.add_argument("--uniprot-method", choices=["search", "stream"], default="search")
    ap.add_argument("--uniprot-page-size", type=int, default=500)
    ap.add_argument("--uniprot-max-rows", type=int, default=5000, help="0 means no limit")
    ap.add_argument("--save-uniprot-raw", default="", help="Optional path to save raw uniprot tsv")

    ap.add_argument("--brenda-export", default="", help="Optional local BRENDA export tsv/csv")
    ap.add_argument("--sabio-export", default="", help="Optional local SABIO export tsv/csv")
    ap.add_argument("--manual-overrides", default="", help="Optional manual override csv")
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
        print(f"[Info] Parsed UniProt records: {len(parsed)}")

    if args.brenda_export:
        bdf = parse_brenda_or_sabio_table(args.brenda_export, source_name="BRENDA")
        frames.append(bdf)
        print(f"[Info] Parsed BRENDA records: {len(bdf)}")

    if args.sabio_export:
        sdf = parse_brenda_or_sabio_table(args.sabio_export, source_name="SABIO-RK")
        frames.append(sdf)
        print(f"[Info] Parsed SABIO-RK records: {len(sdf)}")

    if not frames:
        raise ValueError("No input data. Use at least one of --fetch-uniprot / --brenda-export / --sabio-export")

    merged = merge_and_dedup(frames)
    print(f"[Info] After merge+dedup: {len(merged)}")

    if args.manual_overrides:
        merged = apply_manual_overrides(merged, args.manual_overrides)
        merged = merge_and_dedup([merged])
        print(f"[Info] After manual overrides: {len(merged)}")

    out_cols = [
        "id",
        "sequence",
        "ec_subclass",
        "substrate_labels",
        "metal_labels",
        "source",
        "accession",
        "organism",
        "notes",
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
