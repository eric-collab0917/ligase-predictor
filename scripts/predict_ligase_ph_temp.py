#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for ligase optimal pH and temperature prediction.

Supports:
- Single sequence prediction
- FASTA file batch prediction
- CSV file batch prediction
- PDB file (extracts sequence from ATOM records)

Usage:
  # Single sequence
  python scripts/predict_ligase_ph_temp.py \
      --checkpoint outputs/ligase_ph_temp_v1/best_ligase_ph_temp.pt \
      --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALMPASQARIATVPVSMVERMLQAAQPGETLVVTQSFNDFGFPGELRLCEAATGCATPGSLAPVTLGQGVVLASNGKPVTTGKHASLAGMA"

  # FASTA file
  python scripts/predict_ligase_ph_temp.py \
      --checkpoint outputs/ligase_ph_temp_v1/best_ligase_ph_temp.pt \
      --fasta data/raw/ligase_positive.fasta \
      --output predictions.csv \
      --max-seqs 100

  # CSV file
  python scripts/predict_ligase_ph_temp.py \
      --checkpoint outputs/ligase_ph_temp_v1/best_ligase_ph_temp.pt \
      --csv data/processed/ligase_ph_temp_auto.csv \
      --seq-col sequence \
      --output predictions.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ligase_ph_temp import LigasePhTempModel, clean_sequence, get_device

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_model(
    checkpoint_path: str, device: torch.device
) -> Tuple[LigasePhTempModel, dict]:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})

    model = LigasePhTempModel(
        model_name=config.get("model_name", "facebook/esm2_t6_8M_UR50D"),
        dropout=config.get("dropout", 0.2),
        freeze_backbone=True,  # Always freeze for inference
        unfreeze_last_n_layers=0,
        hidden_dim=config.get("hidden_dim", 128),
        num_ph_bins=int(config.get("ph_num_bins", 0)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model, config


def parse_fasta(fasta_path: str, max_seqs: int = 0) -> List[Tuple[str, str]]:
    """Parse FASTA file, return list of (id, sequence)."""
    sequences = []
    current_id = ""
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    seq = clean_sequence("".join(current_seq))
                    if seq:
                        sequences.append((current_id, seq))
                        if max_seqs > 0 and len(sequences) >= max_seqs:
                            return sequences
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        # Last sequence
        if current_id and current_seq:
            seq = clean_sequence("".join(current_seq))
            if seq:
                sequences.append((current_id, seq))

    return sequences


def parse_pdb_sequence(pdb_path: str) -> str:
    """Extract sequence from PDB ATOM records."""
    aa_map = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    residues = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_name = line[17:20].strip()
                chain = line[21]
                res_num = int(line[22:26].strip())
                if res_name in aa_map:
                    key = (chain, res_num)
                    if key not in residues:
                        residues[key] = aa_map[res_name]
    # Sort by chain and residue number
    sorted_keys = sorted(residues.keys())
    seq = "".join(residues[k] for k in sorted_keys)
    return clean_sequence(seq)


@torch.no_grad()
def predict_batch(
    model: LigasePhTempModel,
    tokenizer,
    sequences: List[str],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 8,
) -> List[Tuple[float, float]]:
    """Predict pH and temperature for a batch of sequences."""
    results = []
    model.eval()

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        enc = tokenizer(
            batch_seqs,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        ph_pred, temp_pred = model(input_ids, attention_mask)
        ph_pred = ph_pred.cpu().numpy()
        temp_pred = temp_pred.cpu().numpy()

        for j in range(len(batch_seqs)):
            results.append((float(ph_pred[j]), float(temp_pred[j])))

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Predict ligase optimal pH and temperature"
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    ap.add_argument("--sequence", default="", help="Single sequence to predict")
    ap.add_argument("--fasta", default="", help="FASTA file for batch prediction")
    ap.add_argument("--csv", default="", help="CSV file for batch prediction")
    ap.add_argument("--pdb", default="", help="PDB file to extract sequence from")
    ap.add_argument("--seq-col", default="sequence", help="Sequence column in CSV")
    ap.add_argument("--id-col", default="id", help="ID column in CSV (optional)")
    ap.add_argument("--output", default="", help="Output CSV path for batch predictions")
    ap.add_argument("--max-seqs", type=int, default=0, help="Max sequences to process (0=all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"[Info] Using device: {device}")

    # Load model
    print(f"[Info] Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    max_length = config.get("max_length", 512)
    model_name = config.get("model_name", "facebook/esm2_t6_8M_UR50D")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Collect sequences
    sequences = []
    ids = []

    if args.sequence:
        seq = clean_sequence(args.sequence)
        if seq:
            sequences.append(seq)
            ids.append("input")
        else:
            print("[Error] Invalid sequence provided.")
            return

    elif args.pdb:
        seq = parse_pdb_sequence(args.pdb)
        if seq:
            sequences.append(seq)
            ids.append(Path(args.pdb).stem)
        else:
            print(f"[Error] Could not extract sequence from PDB: {args.pdb}")
            return

    elif args.fasta:
        print(f"[Info] Parsing FASTA: {args.fasta}")
        parsed = parse_fasta(args.fasta, max_seqs=args.max_seqs)
        for fid, seq in parsed:
            sequences.append(seq)
            ids.append(fid)
        print(f"[Info] Loaded {len(sequences)} sequences from FASTA")

    elif args.csv:
        print(f"[Info] Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        if args.seq_col not in df.columns:
            print(f"[Error] Column '{args.seq_col}' not found in CSV")
            return
        for i, row in df.iterrows():
            if args.max_seqs > 0 and len(sequences) >= args.max_seqs:
                break
            seq = clean_sequence(str(row.get(args.seq_col, "")))
            if seq:
                sequences.append(seq)
                rid = str(row.get(args.id_col, f"row_{i}"))
                ids.append(rid)
        print(f"[Info] Loaded {len(sequences)} sequences from CSV")

    else:
        print("[Error] Provide --sequence, --fasta, --csv, or --pdb")
        return

    if not sequences:
        print("[Error] No valid sequences to predict.")
        return

    # Predict
    print(f"[Info] Predicting {len(sequences)} sequences...")
    results = predict_batch(
        model=model,
        tokenizer=tokenizer,
        sequences=sequences,
        device=device,
        max_length=max_length,
        batch_size=args.batch_size,
    )

    # Output
    if len(sequences) == 1 and not args.output:
        # Single sequence: print to console
        ph, temp = results[0]
        print(f"\n{'='*50}")
        print(f"Sequence ID: {ids[0]}")
        print(f"Sequence length: {len(sequences[0])} aa")
        print(f"Predicted optimal pH: {ph:.2f}")
        print(f"Predicted optimal temperature: {temp:.1f} °C")
        print(f"{'='*50}")
    else:
        # Batch: save to CSV
        out_path = args.output or "predictions_ph_temp.csv"
        out_df = pd.DataFrame(
            {
                "id": ids,
                "sequence": sequences,
                "predicted_opt_ph": [r[0] for r in results],
                "predicted_opt_temp": [r[1] for r in results],
            }
        )
        out_df.to_csv(out_path, index=False)
        print(f"[Done] Saved predictions to: {out_path}")

        # Print summary
        ph_vals = [r[0] for r in results]
        temp_vals = [r[1] for r in results]
        print(f"\n[Summary]")
        print(f"  Predicted pH:   mean={sum(ph_vals)/len(ph_vals):.2f}, "
              f"min={min(ph_vals):.2f}, max={max(ph_vals):.2f}")
        print(f"  Predicted Temp: mean={sum(temp_vals)/len(temp_vals):.1f} °C, "
              f"min={min(temp_vals):.1f} °C, max={max(temp_vals):.1f} °C")


if __name__ == "__main__":
    main()
