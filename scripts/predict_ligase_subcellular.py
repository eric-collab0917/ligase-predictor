#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI prediction for ligase subcellular localization.

Usage:
  python scripts/predict_ligase_subcellular.py \
      --checkpoint outputs/ligase_subcellular_v1/best_ligase_subcellular.pt \
      --sequence "MVLSPADKTNVKAAWGKVGAHAGEY..."
  python scripts/predict_ligase_subcellular.py \
      --checkpoint outputs/ligase_subcellular_v1/best_ligase_subcellular.pt \
      --fasta input.fasta
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ligase_subcellular import LigaseSubcellularModel, clean_sequence, get_device


def parse_fasta(path):
    records = []
    name = None
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seq = clean_sequence("".join(chunks))
                    if seq:
                        records.append((name, seq))
                name = line[1:].strip() or f"seq_{len(records)+1}"
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        seq = clean_sequence("".join(chunks))
        if seq:
            records.append((name, seq))
    return records


@torch.no_grad()
def predict_batch(model, tokenizer, records, device, max_length,
                  id2label, threshold):
    seqs = [s for _, s in records]
    enc = tokenizer(seqs, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    logits = model(input_ids, attention_mask)
    probs = torch.sigmoid(logits).cpu().numpy()

    out = []
    for i, (name, seq) in enumerate(records):
        preds = []
        for j in range(len(id2label)):
            p = float(probs[i, j])
            if p >= threshold:
                preds.append((id2label[j], p))
        if len(preds) == 0 and len(id2label) > 0:
            j = int(np.argmax(probs[i]))
            preds = [(id2label[j], float(probs[i, j]))]

        out.append({
            "id": name,
            "length": len(seq),
            "subcellular_pred": preds,
        })
    return out


def write_csv(rows, out_csv):
    fields = ["id", "length", "subcellular_pred"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["subcellular_pred"] = json.dumps(rr["subcellular_pred"], ensure_ascii=False)
            w.writerow(rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="best_ligase_subcellular.pt")
    ap.add_argument("--sequence", default="")
    ap.add_argument("--fasta", default="")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold (default: use checkpoint value or 0.5)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    if bool(args.sequence) == bool(args.fasta):
        raise ValueError("Provide exactly one of --sequence or --fasta")

    if args.sequence:
        seq = clean_sequence(args.sequence)
        if not seq:
            raise ValueError("Invalid sequence")
        records = [("input_sequence", seq)]
    else:
        records = parse_fasta(args.fasta)
        if len(records) == 0:
            raise ValueError("No valid sequences in FASTA")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    maps = ckpt.get("label_maps", {})
    subcell_to_idx = maps.get("subcellular_to_idx", {})

    if len(subcell_to_idx) == 0:
        raise ValueError("Checkpoint has no subcellular label map")

    id2label = [x for x, _ in sorted(subcell_to_idx.items(), key=lambda kv: kv[1])]

    threshold = args.threshold if args.threshold is not None else float(cfg.get("threshold", 0.5))
    print(f"[Info] Subcellular classes ({len(id2label)}): {id2label}")
    print(f"[Info] Decision threshold: {threshold:.2f}")

    device = get_device(args.device)
    model = LigaseSubcellularModel(
        model_name=cfg.get("model_name", "facebook/esm2_t6_8M_UR50D"),
        num_subcellular=len(id2label),
        dropout=cfg.get("dropout", 0.2),
        freeze_backbone=True,
        unfreeze_last_n_layers=0,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    model_name = cfg.get("model_name", "facebook/esm2_t6_8M_UR50D")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = int(cfg.get("max_length", 512))

    all_rows = []
    for i in range(0, len(records), args.batch_size):
        batch = records[i : i + args.batch_size]
        pred_rows = predict_batch(
            model=model,
            tokenizer=tokenizer,
            records=batch,
            device=device,
            max_length=max_length,
            id2label=id2label,
            threshold=threshold,
        )
        all_rows.extend(pred_rows)

    for r in all_rows:
        pred_str = "; ".join([f"{k} ({v:.2f})" for k, v in r["subcellular_pred"]])
        print(f"[{r['id']}] subcellular={pred_str}")

    if args.out_csv:
        out_csv = str(Path(args.out_csv).resolve())
        write_csv(all_rows, out_csv)
        print(f"[Done] CSV saved to: {out_csv}")

    if args.out_json:
        out_json = str(Path(args.out_json).resolve())
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)
        print(f"[Done] JSON saved to: {out_json}")


if __name__ == "__main__":
    main()
