#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_kcat_from_sequence.py

Input:
- single sequence (--sequence) or FASTA file (--fasta)
- model path: either
  1) full_model.joblib (from train_kcat_baseline.py for xgb/lgbm/rf), or
  2) blend_model.json (from --model blend output)

Output:
- predicted log_kcat for each sequence

Notes:
- This script uses ESM-2 (default: facebook/esm2_t12_35M_UR50D) to extract per-residue embeddings.
- Features are assembled to match training-time flattened graph feature layout:
  [mean(480), std(480), max(480), min(480), node_l2_mean, global_mean, global_std, deg_mean, deg_std, n, e, e/n]
- Since only sequence is provided (no 3D graph), graph-topology tail features are approximated:
  - prefer means from training feature cache (--feature-cache, last 5 dims)
  - fallback to simple bidirectional chain statistics.
"""

import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import joblib

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from transformers import AutoTokenizer, AutoModel


def get_device(device_arg: str):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_sequence(seq: str):
    seq = re.sub(r"\s+", "", seq).upper()
    if not seq:
        raise ValueError("Empty sequence after cleaning.")
    # Keep common amino-acid alphabet + X/B/Z/U/O/J
    if re.search(r"[^ACDEFGHIKLMNPQRSTVWYBXZUOJ]", seq):
        raise ValueError(f"Sequence contains invalid residue letters: {seq}")
    return seq


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XLE": "J",
}


def parse_pdb_to_sequence_and_topology(pdb_path: str, chain: str = "", cutoff: float = 10.0):
    residues = []
    seen = set()
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain_id = line[21].strip()
            if chain and chain_id != chain:
                continue
            resname = line[17:20].strip().upper()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            seen.add(key)

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue
            aa = AA3_TO_1.get(resname, "X")
            residues.append((aa, (x, y, z)))

    if not residues:
        raise ValueError(f"No CA residues parsed from PDB: {pdb_path}")

    seq = "".join([r[0] for r in residues])
    seq = clean_sequence(seq)
    coords = np.asarray([r[1] for r in residues], dtype=np.float64)
    topo5 = topology_from_coords(coords, cutoff=cutoff)
    return seq, topo5


def parse_fasta(path: str):
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
                    records.append((name, clean_sequence("".join(chunks))))
                name = line[1:].strip() or f"seq_{len(records)+1}"
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, clean_sequence("".join(chunks))))
    if not records:
        raise ValueError(f"No FASTA records found in: {path}")
    return records


def load_sequences(args):
    used = sum(bool(x) for x in [args.sequence, args.fasta, args.pdb])
    if used > 1:
        raise ValueError("Use only one of --sequence / --fasta / --pdb.")
    if used == 0:
        raise ValueError("Please provide --sequence, --fasta, or --pdb.")
    if args.sequence:
        return [{"id": "input_sequence", "seq": clean_sequence(args.sequence), "topo5": None}]
    if args.fasta:
        return [{"id": n, "seq": s, "topo5": None} for n, s in parse_fasta(args.fasta)]
    pdb_abs = str(Path(args.pdb).resolve())
    seq, topo5 = parse_pdb_to_sequence_and_topology(
        pdb_abs, chain=args.pdb_chain, cutoff=args.pdb_cutoff
    )
    pdb_id = Path(pdb_abs).stem
    if args.pdb_chain:
        pdb_id = f"{pdb_id}_chain_{args.pdb_chain}"
    return [{"id": pdb_id, "seq": seq, "topo5": topo5}]


def chain_topology_stats(seq_len: int):
    n = float(seq_len)
    if seq_len <= 1:
        e = 0.0
        deg_mean = 0.0
        deg_std = 0.0
    else:
        # Bidirectional chain edges: i<->i+1
        e = float(2 * (seq_len - 1))
        deg = np.array([1.0] + [2.0] * (seq_len - 2) + [1.0], dtype=np.float64)
        deg_mean = float(deg.mean())
        deg_std = float(deg.std())
    density = e / max(n, 1.0)
    return np.array([deg_mean, deg_std, n, e, density], dtype=np.float64)


def topology_from_coords(coords: np.ndarray, cutoff: float = 10.0):
    n = int(coords.shape[0])
    if n <= 1:
        return np.array([0.0, 0.0, float(n), 0.0, 0.0], dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    adj = (d2 <= float(cutoff) ** 2)
    np.fill_diagonal(adj, False)
    deg = adj.sum(axis=1).astype(np.float64)
    e = float(adj.sum())  # directed edges count
    density = e / max(float(n), 1.0)
    return np.array([float(deg.mean()), float(deg.std()), float(n), e, density], dtype=np.float64)


def load_topology_priors(feature_cache_path: str):
    if not feature_cache_path:
        return None
    if not os.path.exists(feature_cache_path):
        raise FileNotFoundError(f"--feature-cache not found: {feature_cache_path}")
    cache = np.load(feature_cache_path, allow_pickle=True)
    if "X" not in cache.files:
        raise KeyError(f"{feature_cache_path} does not contain key 'X'.")
    X = np.asarray(cache["X"], dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 5:
        raise ValueError(f"Invalid X shape in feature cache: {X.shape}")
    # training feature tail: [..., deg_mean, deg_std, n, e, e/n]
    return X[:, -5:].mean(axis=0)


def load_esm(model_name: str, device: torch.device):
    # Prefer local cache first; fall back to online fetch.
    local_err = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        local_err = e
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        except Exception as online_err:
            raise RuntimeError(
                f"Failed to load ESM model '{model_name}'. "
                "Network to Hugging Face may be unstable. "
                "Try pre-downloading model files or switch to a smaller model "
                "(e.g. facebook/esm2_t6_8M_UR50D). "
                f"local_error={local_err} online_error={online_err}"
            ) from online_err
    model.eval()
    model.to(device)
    return tokenizer, model


@torch.no_grad()
def residue_embeddings(seq: str, tokenizer, model, device: torch.device):
    # Use plain sequence string with special tokens handled by tokenizer.
    toks = tokenizer(seq, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model(**toks).last_hidden_state  # [1, T, H]

    attn = toks.get("attention_mask")
    if attn is None:
        # Fallback: assume all valid tokens
        valid = out[0]
    else:
        idx = torch.where(attn[0] > 0)[0]
        valid = out[0, idx, :]

    # Remove BOS/EOS if present
    if valid.shape[0] >= 3:
        valid = valid[1:-1, :]
    if valid.shape[0] == 0:
        raise RuntimeError("No residue embeddings extracted. Check sequence/tokenizer compatibility.")
    return valid.float()  # [L, H]


def sequence_to_feature(
    seq: str, tokenizer, esm_model, device: torch.device, topo_priors=None, topo5_override=None
):
    x = residue_embeddings(seq, tokenizer, esm_model, device=device)  # [L, H]
    x_np = x.detach().cpu().numpy().astype(np.float64)

    mean = x_np.mean(axis=0)
    std = x_np.std(axis=0)
    mx = x_np.max(axis=0)
    mn = x_np.min(axis=0)

    node_l2_mean = float(np.linalg.norm(x_np, axis=1).mean())
    global_mean = float(x_np.mean())
    global_std = float(x_np.std())

    if topo5_override is not None:
        topo5 = np.asarray(topo5_override, dtype=np.float64).reshape(-1)
        if topo5.shape[0] != 5:
            raise ValueError(f"topo5_override must have 5 dims, got {topo5.shape}")
    elif topo_priors is not None:
        topo5 = np.asarray(topo_priors, dtype=np.float64).reshape(-1)
        if topo5.shape[0] != 5:
            raise ValueError(f"topo_priors must have 5 dims, got {topo5.shape}")
    else:
        topo5 = chain_topology_stats(seq_len=x_np.shape[0])

    feature = np.concatenate(
        [
            mean,
            std,
            mx,
            mn,
            np.array([node_l2_mean, global_mean, global_std], dtype=np.float64),
            topo5,
        ],
        axis=0,
    )
    return feature.astype(np.float64)


def _relocate_old_outputs_path(path_str: str, project_root: Path):
    p = Path(path_str)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.startswith("outputs_"):
            sub = part[len("outputs_") :]
            trailing = parts[i + 1 :]
            candidate = project_root / "outputs" / sub
            for t in trailing:
                candidate = candidate / t
            if candidate.exists():
                return str(candidate.resolve())
            break
    return ""


def resolve_path(path_str: str, base_dir: str):
    p = Path(path_str)
    base = Path(base_dir).resolve()
    # `blend_model.json` may store component paths as:
    # 1) absolute paths
    # 2) relative to blend dir (e.g., ../lgbm/full_model.joblib)
    # 3) relative to repo root (e.g., outputs/kcat_blend/lgbm/full_model.joblib)
    # Try common anchors in order and pick the first existing path.
    project_root = Path.cwd().resolve()

    if p.is_absolute():
        if p.exists():
            return str(p.resolve())
        relocated = _relocate_old_outputs_path(path_str, project_root=project_root)
        if relocated:
            return relocated
        return str(p)

    candidates = [
        (base / p).resolve(),
        (base.parent / p).resolve(),
        (base.parent.parent / p).resolve(),
        (project_root / p).resolve(),
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # Fallback: keep previous behavior for debuggability.
    return str((base / p).resolve())


def predict_with_joblib(feature_vec: np.ndarray, artifact_path: str):
    art = joblib.load(artifact_path)
    if not isinstance(art, dict):
        raise ValueError(f"Unsupported joblib content in {artifact_path}; expected dict artifact.")
    if not all(k in art for k in ("scaler", "pca", "model")):
        raise KeyError(f"{artifact_path} must contain keys: scaler, pca, model")

    X = feature_vec.reshape(1, -1)
    Xs = art["scaler"].transform(X)
    Xp = art["pca"].transform(Xs)
    pred = float(np.asarray(art["model"].predict(Xp)).reshape(-1)[0])
    return pred


def predict_with_blend(feature_vec: np.ndarray, blend_json_path: str):
    with open(blend_json_path, "r", encoding="utf-8") as f:
        blend = json.load(f)

    weights = blend.get("weights", {})
    w_lgbm = float(weights.get("lgbm", 0.0))
    w_xgb = float(weights.get("xgb", 0.0))
    if abs((w_lgbm + w_xgb) - 1.0) > 1e-6 and (w_lgbm + w_xgb) > 0:
        s = w_lgbm + w_xgb
        w_lgbm, w_xgb = w_lgbm / s, w_xgb / s

    comps = blend.get("components", {})
    base = str(Path(blend_json_path).parent)
    lgbm_path = comps.get("lgbm_full_model_path")
    xgb_path = comps.get("xgb_full_model_path")
    if not lgbm_path or not xgb_path:
        raise KeyError("blend JSON missing component model paths.")
    lgbm_path = resolve_path(lgbm_path, base)
    xgb_path = resolve_path(xgb_path, base)

    p_lgbm = predict_with_joblib(feature_vec, lgbm_path)
    p_xgb = predict_with_joblib(feature_vec, xgb_path)
    p_blend = w_lgbm * p_lgbm + w_xgb * p_xgb
    return {
        "pred_blend": float(p_blend),
        "pred_lgbm": float(p_lgbm),
        "pred_xgb": float(p_xgb),
        "w_lgbm": float(w_lgbm),
        "w_xgb": float(w_xgb),
    }


def predict_one(feature_vec: np.ndarray, model_path: str):
    if model_path.endswith(".json"):
        return predict_with_blend(feature_vec, model_path)
    pred = predict_with_joblib(feature_vec, model_path)
    return {"pred_blend": pred}


def print_rows(rows):
    header = ["id", "length", "pred_blend", "pred_lgbm", "pred_xgb", "w_lgbm", "w_xgb"]
    print(",".join(header))
    for r in rows:
        print(
            f"{r.get('id','')},{r.get('length','')},"
            f"{r.get('pred_blend','')},{r.get('pred_lgbm','')},"
            f"{r.get('pred_xgb','')},{r.get('w_lgbm','')},{r.get('w_xgb','')}"
        )
    return header


def save_rows_csv(rows, out_path: str, header=None):
    if header is None:
        header = ["id", "length", "pred_blend", "pred_lgbm", "pred_xgb", "w_lgbm", "w_xgb"]
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                f"{r.get('id','')},{r.get('length','')},"
                f"{r.get('pred_blend','')},{r.get('pred_lgbm','')},"
                f"{r.get('pred_xgb','')},{r.get('w_lgbm','')},{r.get('w_xgb','')}\n"
            )
    print(f"[Info] Saved predictions to: {out}")


def run_interactive(model_path, tokenizer, esm_model, device, topo_priors):
    print("[Interactive] Paste protein sequence and press Enter.")
    print("[Interactive] Type 'quit' or 'exit' to stop.")
    idx = 1
    while True:
        raw = input("seq> ").strip()
        if not raw:
            continue
        if raw.lower() in {"quit", "exit"}:
            break
        try:
            seq = clean_sequence(raw)
            feat = sequence_to_feature(
                seq=seq,
                tokenizer=tokenizer,
                esm_model=esm_model,
                device=device,
                topo_priors=topo_priors,
                topo5_override=None,
            )
            pred = predict_one(feat, model_path=model_path)
            row = {"id": f"interactive_{idx}", "length": len(seq), **pred}
            idx += 1
            print_rows([row])
        except Exception as e:
            print(f"[Error] {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Path to full_model.joblib or blend_model.json")
    ap.add_argument("--sequence", default="", help="Single protein sequence")
    ap.add_argument("--fasta", default="", help="FASTA file path")
    ap.add_argument("--pdb", default="", help="Single PDB file path")
    ap.add_argument("--pdb-chain", default="", help="Optional chain ID for PDB input, e.g. A")
    ap.add_argument("--pdb-cutoff", type=float, default=10.0, help="CA distance cutoff(Angstrom)")
    ap.add_argument("--output-csv", default="", help="Optional output CSV path")
    ap.add_argument("--feature-cache", default="", help="Optional feat_cache.npz for topology priors")
    ap.add_argument("--esm-model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--interactive", action="store_true", help="Interactive CLI mode")
    args = ap.parse_args()

    model_path = str(Path(args.model_path).resolve())
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"--model-path not found: {model_path}")

    records = [] if args.interactive else load_sequences(args)
    topo_priors = load_topology_priors(args.feature_cache) if args.feature_cache else None

    device = get_device(args.device)
    print(f"[Info] Device: {device}")
    print(f"[Info] Loading ESM model: {args.esm_model}")
    tokenizer, esm_model = load_esm(args.esm_model, device=device)

    if args.interactive:
        run_interactive(
            model_path=model_path,
            tokenizer=tokenizer,
            esm_model=esm_model,
            device=device,
            topo_priors=topo_priors,
        )
        return

    rows = []
    for rec in records:
        name, seq = rec["id"], rec["seq"]
        feat = sequence_to_feature(
            seq=seq,
            tokenizer=tokenizer,
            esm_model=esm_model,
            device=device,
            topo_priors=topo_priors,
            topo5_override=rec.get("topo5"),
        )
        pred = predict_one(feat, model_path=model_path)
        row = {"id": name, "length": len(seq), **pred}
        rows.append(row)

    # print to terminal
    header = print_rows(rows)

    if args.output_csv:
        save_rows_csv(rows, args.output_csv, header=header)


if __name__ == "__main__":
    main()
