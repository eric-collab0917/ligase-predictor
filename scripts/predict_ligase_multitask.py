#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from src.ligase_multitask import LigaseMultiTaskModel, clean_sequence, get_device, unpack_multitask_outputs


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


def topk_probs(probs, labels, k=3):
    idx = np.argsort(-probs)[:k]
    return [(labels[i], float(probs[i])) for i in idx]


def apply_ec_logit_adjustment(ec_logits, ec_log_prior=None, tau=0.0):
    if ec_log_prior is None or float(tau) <= 0.0:
        return ec_logits
    lp = torch.tensor(ec_log_prior, dtype=ec_logits.dtype, device=ec_logits.device).view(1, -1)
    return ec_logits - float(tau) * lp


@torch.no_grad()
def predict_batch(
    model,
    tokenizer,
    records,
    device,
    max_length,
    id2ec,
    id2sub,
    id2metal,
    substrate_threshold,
    metal_threshold,
    metal_presence_threshold,
    substrate_thresholds=None,
    metal_thresholds=None,
    ec_log_prior=None,
    ec_logit_adjust_tau=0.0,
):
    seqs = [s for _, s in records]
    enc = tokenizer(seqs, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask)
    ec_logits, sub_logits, metal_logits, metal_presence_logits = unpack_multitask_outputs(outputs)
    ec_logits = apply_ec_logit_adjustment(ec_logits, ec_log_prior=ec_log_prior, tau=ec_logit_adjust_tau)

    ec_prob = torch.softmax(ec_logits, dim=1).cpu().numpy()
    sub_prob = torch.sigmoid(sub_logits).cpu().numpy() if len(id2sub) else np.zeros((len(records), 0), dtype=np.float32)
    metal_prob = torch.sigmoid(metal_logits).cpu().numpy() if len(id2metal) else np.zeros((len(records), 0), dtype=np.float32)
    metal_presence_prob = None
    if metal_presence_logits is not None:
        metal_presence_prob = torch.sigmoid(metal_presence_logits).cpu().numpy()

    sub_thr_arr = None
    if substrate_thresholds is not None and len(substrate_thresholds) == len(id2sub):
        sub_thr_arr = np.asarray(substrate_thresholds, dtype=np.float32)
    metal_thr_arr = None
    if metal_thresholds is not None and len(metal_thresholds) == len(id2metal):
        metal_thr_arr = np.asarray(metal_thresholds, dtype=np.float32)

    out = []
    for i, (name, seq) in enumerate(records):
        ec_top1_idx = int(np.argmax(ec_prob[i]))
        ec_top1_label = id2ec[ec_top1_idx]
        ec_top1_prob = float(ec_prob[i, ec_top1_idx])
        ec_top3 = topk_probs(ec_prob[i], id2ec, k=min(3, len(id2ec)))

        sub_pos = []
        for j in range(len(id2sub)):
            thr = float(sub_thr_arr[j]) if sub_thr_arr is not None else float(substrate_threshold)
            if sub_prob[i, j] >= thr:
                sub_pos.append((id2sub[j], float(sub_prob[i, j])))
        if len(sub_pos) == 0 and len(id2sub) > 0:
            j = int(np.argmax(sub_prob[i]))
            sub_pos = [(id2sub[j], float(sub_prob[i, j]))]

        if len(id2metal) == 0:
            metal_pos = []
            metal_presence = float("nan")
        else:
            metal_presence = float(metal_presence_prob[i]) if metal_presence_prob is not None else float("nan")
            if metal_presence_prob is not None and metal_presence_prob[i] < metal_presence_threshold:
                metal_pos = [("NONE", float(1.0 - metal_presence_prob[i]))]
            else:
                metal_pos = []
                for j in range(len(id2metal)):
                    thr = float(metal_thr_arr[j]) if metal_thr_arr is not None else float(metal_threshold)
                    if metal_prob[i, j] >= thr:
                        metal_pos.append((id2metal[j], float(metal_prob[i, j])))
                if len(metal_pos) == 0:
                    j = int(np.argmax(metal_prob[i]))
                    metal_pos = [(id2metal[j], float(metal_prob[i, j]))]

        out.append(
            {
                "id": name,
                "length": len(seq),
                "ec_top1": ec_top1_label,
                "ec_top1_prob": ec_top1_prob,
                "ec_top3": ec_top3,
                "substrate_pred": sub_pos,
                "metal_pred": metal_pos,
                "metal_presence_prob": metal_presence,
            }
        )
    return out


def write_csv(rows, out_csv):
    fields = [
        "id",
        "length",
        "ec_top1",
        "ec_top1_prob",
        "ec_top3",
        "substrate_pred",
        "metal_pred",
        "metal_presence_prob",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["ec_top3"] = json.dumps(rr["ec_top3"], ensure_ascii=False)
            rr["substrate_pred"] = json.dumps(rr["substrate_pred"], ensure_ascii=False)
            rr["metal_pred"] = json.dumps(rr["metal_pred"], ensure_ascii=False)
            w.writerow(rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="best_ligase_multitask.pt")
    ap.add_argument("--sequence", default="")
    ap.add_argument("--fasta", default="")
    ap.add_argument("--threshold", type=float, default=None, help="legacy override for both substrate+metal type")
    ap.add_argument("--substrate-threshold", type=float, default=None)
    ap.add_argument("--metal-threshold", type=float, default=None)
    ap.add_argument("--metal-presence-threshold", type=float, default=None)
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
    maps = ckpt["label_maps"]
    ec_to_idx = maps["ec_to_idx"]
    sub_to_idx = maps["substrate_to_idx"]
    metal_to_idx = maps["metal_to_idx"]
    thresholds = ckpt.get("decision_thresholds", {})

    id2ec = [x for x, _ in sorted(ec_to_idx.items(), key=lambda kv: kv[1])]
    id2sub = [x for x, _ in sorted(sub_to_idx.items(), key=lambda kv: kv[1])]
    id2metal = [x for x, _ in sorted(metal_to_idx.items(), key=lambda kv: kv[1])]

    default_sub_thr = float(thresholds.get("substrate", 0.5))
    default_metal_thr = float(thresholds.get("metal_type", thresholds.get("metal", 0.5)))
    default_metal_presence_thr = float(thresholds.get("metal_presence", 0.5))
    substrate_per_label = thresholds.get("substrate_per_label", {}) or {}
    metal_per_label = thresholds.get("metal_type_per_label", {}) or {}
    ec_log_prior = ckpt.get("ec_log_prior", [])
    ec_logit_adjust_tau = float(cfg.get("ec_logit_adjust_tau", 0.0))
    if not bool(cfg.get("ec_logit_adjust", False)):
        ec_logit_adjust_tau = 0.0

    substrate_threshold = default_sub_thr
    metal_threshold = default_metal_thr
    metal_presence_threshold = default_metal_presence_thr
    substrate_thresholds = [float(substrate_per_label.get(lbl, default_sub_thr)) for lbl in id2sub] if id2sub else []
    metal_thresholds = [float(metal_per_label.get(lbl, default_metal_thr)) for lbl in id2metal] if id2metal else []

    if args.threshold is not None:
        substrate_threshold = float(args.threshold)
        metal_threshold = float(args.threshold)
        substrate_thresholds = []
        metal_thresholds = []
    if args.substrate_threshold is not None:
        substrate_threshold = float(args.substrate_threshold)
        substrate_thresholds = []
    if args.metal_threshold is not None:
        metal_threshold = float(args.metal_threshold)
        metal_thresholds = []
    if args.metal_presence_threshold is not None:
        metal_presence_threshold = float(args.metal_presence_threshold)

    device = get_device(args.device)
    model = LigaseMultiTaskModel(
        model_name=cfg.get("model_name", "facebook/esm2_t6_8M_UR50D"),
        num_ec=len(id2ec),
        num_substrate=len(id2sub),
        num_metal=len(id2metal),
        dropout=cfg.get("dropout", 0.2),
        freeze_backbone=True,
        metal_two_stage=bool(cfg.get("metal_two_stage", False)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    model_name = cfg.get("model_name", "facebook/esm2_t6_8M_UR50D")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = int(cfg.get("max_length", 512))

    print(
        f"[Info] thresholds: substrate={substrate_threshold:.2f}, "
        f"metal_type={metal_threshold:.2f}, metal_presence={metal_presence_threshold:.2f}"
    )
    if substrate_thresholds:
        print("[Info] using per-label substrate thresholds from checkpoint")
    if metal_thresholds:
        print("[Info] using per-label metal thresholds from checkpoint")
    if ec_logit_adjust_tau > 0.0 and ec_log_prior:
        print(f"[Info] EC logit adjustment enabled at inference, tau={ec_logit_adjust_tau:.2f}")

    all_rows = []
    for i in range(0, len(records), args.batch_size):
        batch = records[i : i + args.batch_size]
        pred_rows = predict_batch(
            model=model,
            tokenizer=tokenizer,
            records=batch,
            device=device,
            max_length=max_length,
            id2ec=id2ec,
            id2sub=id2sub,
            id2metal=id2metal,
            substrate_threshold=substrate_threshold,
            metal_threshold=metal_threshold,
            metal_presence_threshold=metal_presence_threshold,
            substrate_thresholds=substrate_thresholds if substrate_thresholds else None,
            metal_thresholds=metal_thresholds if metal_thresholds else None,
            ec_log_prior=ec_log_prior if ec_log_prior else None,
            ec_logit_adjust_tau=ec_logit_adjust_tau,
        )
        all_rows.extend(pred_rows)

    for r in all_rows:
        presence_msg = ""
        if np.isfinite(r.get("metal_presence_prob", np.nan)):
            presence_msg = f" | metal_presence={r['metal_presence_prob']:.3f}"
        print(
            f"[{r['id']}] EC={r['ec_top1']} ({r['ec_top1_prob']:.3f}) | "
            f"substrate={r['substrate_pred']} | metal={r['metal_pred']}{presence_msg}"
        )

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
