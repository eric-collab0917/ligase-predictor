#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import io
import os
import tempfile
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, EsmModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predict_kcat_from_sequence import (
    clean_sequence,
    get_device,
    load_esm,
    load_topology_priors,
    parse_pdb_to_sequence_and_topology,
    predict_one,
    sequence_to_feature,
)
from src.ligase_multitask import LigaseMultiTaskModel, unpack_multitask_outputs

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class EsmBinaryClassifier(nn.Module):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        hidden_size = 320
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        return self.classifier(pooled)


def default_path(p: str):
    return p if os.path.exists(p) else ""


def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return str(Path(p).resolve())
    return ""


def parse_fasta_text(text: str):
    records = []
    name = None
    chunks = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append({"id": name, "seq": clean_sequence("".join(chunks)), "topo5": None})
            name = line[1:].strip() or f"seq_{len(records)+1}"
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        records.append({"id": name, "seq": clean_sequence("".join(chunks)), "topo5": None})
    if not records:
        raise ValueError("No FASTA records found.")
    return records


@st.cache_resource(show_spinner=False)
def load_runtime(esm_model_name: str, device_arg: str):
    device = get_device(device_arg)
    tokenizer, esm_model = load_esm(esm_model_name, device=device)
    return device, tokenizer, esm_model


def safe_load_state(model: nn.Module, weight_path: str):
    state = torch.load(weight_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)
    return model


@st.cache_data(show_spinner=False)
def peek_ligase_multitask_checkpoint_meta(checkpoint_path):
    meta = {
        "found": False,
        "metal_two_stage": False,
        "substrate_threshold": 0.5,
        "metal_threshold": 0.5,
        "metal_presence_threshold": 0.5,
    }
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return meta
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    thresholds = ckpt.get("decision_thresholds", {})
    meta["found"] = True
    meta["metal_two_stage"] = bool(cfg.get("metal_two_stage", False))
    meta["substrate_threshold"] = float(thresholds.get("substrate", 0.5))
    meta["metal_threshold"] = float(thresholds.get("metal_type", thresholds.get("metal", 0.5)))
    meta["metal_presence_threshold"] = float(thresholds.get("metal_presence", 0.5))
    return meta


@st.cache_resource(show_spinner=False)
def load_ligase_multitask_runtime(device_arg, checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    maps = ckpt.get("label_maps", {})

    ec_to_idx = maps.get("ec_to_idx", {})
    sub_to_idx = maps.get("substrate_to_idx", {})
    metal_to_idx = maps.get("metal_to_idx", {})
    if len(ec_to_idx) == 0:
        return None

    id2ec = [x for x, _ in sorted(ec_to_idx.items(), key=lambda kv: kv[1])]
    id2sub = [x for x, _ in sorted(sub_to_idx.items(), key=lambda kv: kv[1])]
    id2metal = [x for x, _ in sorted(metal_to_idx.items(), key=lambda kv: kv[1])]

    model_name = cfg.get("model_name", "facebook/esm2_t6_8M_UR50D")
    max_length = int(cfg.get("max_length", 512))
    dropout = float(cfg.get("dropout", 0.2))

    # Prefer local cache first to avoid network dependency in app usage.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = get_device(device_arg)
    model = LigaseMultiTaskModel(
        model_name=model_name,
        num_ec=len(id2ec),
        num_substrate=len(id2sub),
        num_metal=len(id2metal),
        dropout=dropout,
        freeze_backbone=True,
        metal_two_stage=bool(cfg.get("metal_two_stage", False)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    thresholds = ckpt.get("decision_thresholds", {})
    sub_thr_map = thresholds.get("substrate_per_label", {}) or {}
    metal_thr_map = thresholds.get("metal_type_per_label", {}) or {}
    default_sub_thr = float(thresholds.get("substrate", 0.5))
    default_metal_thr = float(thresholds.get("metal_type", thresholds.get("metal", 0.5)))
    ec_log_prior = ckpt.get("ec_log_prior", [])
    ec_logit_adjust_tau = float(cfg.get("ec_logit_adjust_tau", 0.0))
    if not bool(cfg.get("ec_logit_adjust", False)):
        ec_logit_adjust_tau = 0.0

    return {
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
        "max_length": max_length,
        "id2ec": id2ec,
        "id2sub": id2sub,
        "id2metal": id2metal,
        "default_substrate_threshold": default_sub_thr,
        "default_metal_threshold": default_metal_thr,
        "default_metal_presence_threshold": float(thresholds.get("metal_presence", 0.5)),
        "default_substrate_thresholds": [float(sub_thr_map.get(lbl, default_sub_thr)) for lbl in id2sub],
        "default_metal_thresholds": [float(metal_thr_map.get(lbl, default_metal_thr)) for lbl in id2metal],
        "ec_log_prior": ec_log_prior if ec_log_prior else None,
        "ec_logit_adjust_tau": ec_logit_adjust_tau,
    }


def _topk_with_probs(probs, labels, k=3):
    if len(labels) == 0:
        return []
    idx = np.argsort(-probs)[: min(k, len(labels))]
    return [(labels[i], float(probs[i])) for i in idx]


def _apply_ec_logit_adjustment(ec_logits, ec_log_prior=None, tau=0.0):
    if ec_log_prior is None or float(tau) <= 0.0:
        return ec_logits
    lp = torch.tensor(ec_log_prior, dtype=ec_logits.dtype, device=ec_logits.device).view(1, -1)
    return ec_logits - float(tau) * lp


@torch.no_grad()
def predict_ligase_multitask(
    sequence,
    runtime,
    substrate_threshold=None,
    metal_threshold=None,
    metal_presence_threshold=None,
):
    seq = clean_sequence(sequence)
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]
    max_length = runtime["max_length"]
    id2ec = runtime["id2ec"]
    id2sub = runtime["id2sub"]
    id2metal = runtime["id2metal"]
    default_sub_thr = float(runtime.get("default_substrate_threshold", 0.5))
    default_metal_thr = float(runtime.get("default_metal_threshold", 0.5))
    default_sub_thr_per_label = runtime.get("default_substrate_thresholds", [])
    default_metal_thr_per_label = runtime.get("default_metal_thresholds", [])
    if substrate_threshold is None:
        substrate_threshold = default_sub_thr
        substrate_thresholds = default_sub_thr_per_label if default_sub_thr_per_label else None
    else:
        substrate_thresholds = None
    if metal_threshold is None:
        metal_threshold = default_metal_thr
        metal_thresholds = default_metal_thr_per_label if default_metal_thr_per_label else None
    else:
        metal_thresholds = None
    if metal_presence_threshold is None:
        metal_presence_threshold = float(runtime.get("default_metal_presence_threshold", 0.5))

    enc = tokenizer(seq, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    outputs = model(input_ids, attention_mask)
    ec_logits, sub_logits, metal_logits, metal_presence_logits = unpack_multitask_outputs(outputs)
    ec_logits = _apply_ec_logit_adjustment(
        ec_logits,
        ec_log_prior=runtime.get("ec_log_prior", None),
        tau=float(runtime.get("ec_logit_adjust_tau", 0.0)),
    )

    ec_prob = torch.softmax(ec_logits, dim=1)[0].detach().cpu().numpy()
    sub_prob = torch.sigmoid(sub_logits)[0].detach().cpu().numpy() if len(id2sub) else np.array([])
    metal_prob = torch.sigmoid(metal_logits)[0].detach().cpu().numpy() if len(id2metal) else np.array([])
    metal_presence_prob = None
    if metal_presence_logits is not None:
        metal_presence_prob = float(torch.sigmoid(metal_presence_logits)[0].detach().cpu().item())

    ec_top1_idx = int(np.argmax(ec_prob))
    ec_top1 = id2ec[ec_top1_idx]
    ec_top1_prob = float(ec_prob[ec_top1_idx])
    ec_top3 = _topk_with_probs(ec_prob, id2ec, k=3)

    substrate_pred = []
    for i, p in enumerate(sub_prob):
        thr = float(substrate_thresholds[i]) if substrate_thresholds is not None and i < len(substrate_thresholds) else float(substrate_threshold)
        if p >= thr:
            substrate_pred.append((id2sub[i], float(p)))
    if len(substrate_pred) == 0 and len(id2sub) > 0:
        i = int(np.argmax(sub_prob))
        substrate_pred = [(id2sub[i], float(sub_prob[i]))]

    metal_pred = []
    if len(id2metal) > 0:
        if metal_presence_prob is not None and metal_presence_prob < metal_presence_threshold:
            metal_pred = [("NONE", float(1.0 - metal_presence_prob))]
        else:
            for i, p in enumerate(metal_prob):
                thr = float(metal_thresholds[i]) if metal_thresholds is not None and i < len(metal_thresholds) else float(metal_threshold)
                if p >= thr:
                    metal_pred.append((id2metal[i], float(p)))
            if len(metal_pred) == 0:
                i = int(np.argmax(metal_prob))
                metal_pred = [(id2metal[i], float(metal_prob[i]))]

    return {
        "ec_top1": ec_top1,
        "ec_top1_prob": ec_top1_prob,
        "ec_top3": ec_top3,
        "substrate_pred": substrate_pred,
        "metal_pred": metal_pred,
        "metal_presence_prob": metal_presence_prob,
    }


def _fmt_label_probs(items, max_items=3):
    if not items:
        return "N/A"
    view = items[:max_items]
    return "; ".join([f"{k} ({v:.2f})" for k, v in view])


@st.cache_resource(show_spinner=False)
def load_legacy_runtime(device_arg, sol_path, ligase_path, cofactor_path):
    device = get_device(device_arg)
    tok_8m = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    model_sol = None
    model_ligase = None
    model_cofactor = None

    if sol_path and os.path.exists(sol_path):
        model_sol = safe_load_state(EsmBinaryClassifier(), sol_path).to(device).eval()
    if ligase_path and os.path.exists(ligase_path):
        model_ligase = safe_load_state(EsmBinaryClassifier(), ligase_path).to(device).eval()
    if cofactor_path and os.path.exists(cofactor_path):
        model_cofactor = safe_load_state(EsmBinaryClassifier(), cofactor_path).to(device).eval()

    return {
        "device": device,
        "tok_8m": tok_8m,
        "sol": model_sol,
        "ligase": model_ligase,
        "cofactor": model_cofactor,
    }


def predict_legacy_suite(sequence, runtime):
    seq = clean_sequence(sequence)
    device = runtime["device"]
    tok_8m = runtime["tok_8m"]
    model_sol = runtime["sol"]
    model_ligase = runtime["ligase"]
    model_cofactor = runtime["cofactor"]

    res = {
        "ligase_prob": np.nan,
        "solubility_prob": np.nan,
        "cofactor_type": "N/A",
        "atp_prob": np.nan,
        "nad_prob": np.nan,
    }

    with torch.no_grad():
        if model_sol is not None or model_ligase is not None or model_cofactor is not None:
            enc_8m = tok_8m(seq, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
            input_ids_8m = enc_8m["input_ids"].to(device)
            mask_8m = enc_8m["attention_mask"].to(device)

            if model_ligase is not None:
                out = model_ligase(input_ids_8m, mask_8m)
                res["ligase_prob"] = float(F.softmax(out, dim=1)[0][1].item())
            if model_sol is not None:
                out = model_sol(input_ids_8m, mask_8m)
                res["solubility_prob"] = float(F.softmax(out, dim=1)[0][1].item())
            if model_cofactor is not None:
                out = model_cofactor(input_ids_8m, mask_8m)
                probs = F.softmax(out, dim=1)[0]
                # Same assumption used in old app.py: class1=ATP, class0=NAD+
                res["atp_prob"] = float(probs[1].item())
                res["nad_prob"] = float(probs[0].item())
                res["cofactor_type"] = "ATP" if probs[1].item() >= probs[0].item() else "NAD+"

    return res


def render_radar(legacy, blend_log_kcat):
    ligase = 0.0 if not np.isfinite(legacy["ligase_prob"]) else float(legacy["ligase_prob"])
    sol = 0.0 if not np.isfinite(legacy["solubility_prob"]) else float(legacy["solubility_prob"])
    if np.isfinite(blend_log_kcat):
        kcat_scaled = float(np.clip((blend_log_kcat + 2.0) / 4.0, 0.05, 0.99))
    else:
        kcat_scaled = 0.0

    categories = ["Ligase", "Solubility", "Kinetics", "Expressibility", "Confidence"]
    values = [ligase, sol, kcat_scaled, 0.9 * sol if sol > 0 else 0.0, max(ligase, sol, kcat_scaled)]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.3, 4.3), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="#f59e0b", alpha=0.28)
    ax.plot(angles, values, color="#f97316", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.0)
    return fig


def rows_to_csv(rows):
    header = ["id", "length", "pred_log_kcat", "pred_kcat", "pred_lgbm", "pred_xgb", "w_lgbm", "w_xgb"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def rows_to_csv_ligase_multitask(rows):
    header = [
        "id",
        "length",
        "ec_top1",
        "ec_top1_prob",
        "ec_top3",
        "substrate_pred",
        "metal_pred",
        "metal_presence_prob",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header)
    w.writeheader()
    for r in rows:
        rr = {
            "id": r.get("id", ""),
            "length": r.get("length", ""),
            "ec_top1": r.get("ec_top1", ""),
            "ec_top1_prob": r.get("ec_top1_prob", ""),
            "ec_top3": json.dumps(r.get("ec_top3", []), ensure_ascii=False),
            "substrate_pred": json.dumps(r.get("substrate_pred", []), ensure_ascii=False),
            "metal_pred": json.dumps(r.get("metal_pred", []), ensure_ascii=False),
            "metal_presence_prob": r.get("metal_presence_prob", ""),
        }
        w.writerow(rr)
    return buf.getvalue()


def predict_records(records, model_path, feature_cache, esm_model_name, device_arg):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    topo_priors = None
    if feature_cache:
        if not os.path.exists(feature_cache):
            raise FileNotFoundError(f"Feature cache path not found: {feature_cache}")
        topo_priors = load_topology_priors(feature_cache)

    device, tokenizer, esm_model = load_runtime(esm_model_name, device_arg)

    rows = []
    for rec in records:
        feat = sequence_to_feature(
            seq=rec["seq"],
            tokenizer=tokenizer,
            esm_model=esm_model,
            device=device,
            topo_priors=topo_priors,
            topo5_override=rec.get("topo5"),
        )
        pred = predict_one(feat, model_path=model_path)
        pred_log = float(pred.get("pred_blend"))
        pred_kcat = float(np.power(10.0, pred_log))
        row = {
            "id": rec["id"],
            "length": len(rec["seq"]),
            "pred_log_kcat": pred_log,
            "pred_kcat": pred_kcat,
            "pred_lgbm": pred.get("pred_lgbm", np.nan),
            "pred_xgb": pred.get("pred_xgb", np.nan),
            "w_lgbm": pred.get("w_lgbm", np.nan),
            "w_xgb": pred.get("w_xgb", np.nan),
        }
        rows.append(row)
    return rows


def predict_ligase_multitask_records(
    records,
    checkpoint_path,
    device_arg,
    substrate_threshold=None,
    metal_threshold=None,
    metal_presence_threshold=None,
):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Ligase multitask checkpoint not found: {checkpoint_path}")
    runtime = load_ligase_multitask_runtime(device_arg, checkpoint_path)
    if runtime is None:
        raise RuntimeError("Failed to load ligase multitask runtime.")

    rows = []
    for rec in records:
        pred = predict_ligase_multitask(
            rec["seq"],
            runtime,
            substrate_threshold=substrate_threshold,
            metal_threshold=metal_threshold,
            metal_presence_threshold=metal_presence_threshold,
        )
        rows.append(
            {
                "id": rec["id"],
                "length": len(rec["seq"]),
                **pred,
            }
        )
    return rows


def render_metric_cards(rows):
    if not rows:
        return
    vals = np.array([r["pred_log_kcat"] for r in rows], dtype=np.float64)
    mean_log = float(vals.mean())
    med_log = float(np.median(vals))
    mean_kcat = float(np.power(10.0, mean_log))
    st.markdown(
        f"""
        <div class="card-wrap">
          <div class="card">
            <div class="label">样本数</div>
            <div class="value">{len(rows)}</div>
          </div>
          <div class="card">
            <div class="label">平均 log_kcat</div>
            <div class="value">{mean_log:.4f}</div>
          </div>
          <div class="card">
            <div class="label">中位数 log_kcat</div>
            <div class="value">{med_log:.4f}</div>
          </div>
          <div class="card">
            <div class="label">估计平均 kcat</div>
            <div class="value">{mean_kcat:.3e}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_style():
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(90rem 40rem at -10% -15%, rgba(255, 170, 84, 0.22), transparent 60%),
              radial-gradient(70rem 35rem at 120% -10%, rgba(46, 160, 167, 0.18), transparent 55%),
              linear-gradient(170deg, #fffdf8 0%, #f7fbfb 52%, #f5f7fb 100%);
            color: #1f2937;
          }
          html, body, [class*="css"] {
            font-family: "Avenir Next", "IBM Plex Sans", "Segoe UI", sans-serif;
          }
          .hero {
            border-radius: 18px;
            padding: 20px 24px;
            background: linear-gradient(120deg, #fff6e9 0%, #e8f8f9 55%, #eef3ff 100%);
            border: 1px solid rgba(31, 41, 55, 0.08);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
          }
          .hero h1 {
            margin: 0 0 8px 0;
            font-size: 1.9rem;
            letter-spacing: 0.2px;
          }
          .hero p {
            margin: 0;
            color: #374151;
            font-size: 1rem;
          }
          .card-wrap {
            display: grid;
            gap: 12px;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            margin: 12px 0 14px 0;
          }
          .card {
            background: rgba(255,255,255,0.78);
            backdrop-filter: blur(3px);
            border: 1px solid rgba(31, 41, 55, 0.08);
            border-radius: 14px;
            padding: 12px 14px;
          }
          .card .label {
            font-size: 0.82rem;
            color: #4b5563;
            margin-bottom: 6px;
          }
          .card .value {
            font-size: 1.15rem;
            font-weight: 700;
            color: #111827;
          }
          .small-note {
            color: #4b5563;
            font-size: 0.86rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Enzyme kcat Predictor",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    add_style()

    st.markdown(
        """
        <div class="hero">
          <h1>Enzyme Catalytic Activity Predictor</h1>
          <p>输入序列、FASTA 或 PDB，输出 <b>log_kcat</b> 与 <b>kcat</b> 预测；并支持连接酶鉴定、ATP/NAD 偏好、水溶性、多任务综合分析。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cwd = REPO_ROOT
    expected_model_path = str(cwd / "outputs" / "kcat_blend" / "blend" / "blend_model.json")
    expected_cache_path = str(cwd / "data" / "interim" / "feat_cache.npz")
    default_model = first_existing(
        expected_model_path,
    ) or expected_model_path
    default_cache = first_existing(
        expected_cache_path,
        str(cwd / "feat_cache.npz"),
    ) or expected_cache_path
    default_ligase = first_existing(
        str(cwd / "models" / "checkpoints" / "best_ligase_model.pth"),
    )
    default_sol = first_existing(
        str(cwd / "models" / "checkpoints" / "best_solubility_model.pth"),
    )
    default_cofactor = first_existing(
        str(cwd / "models" / "checkpoints" / "best_cofactor_model.pth"),
    )
    ligase_multitask_ckpt_v3_1 = first_existing(
        str(cwd / "outputs" / "ligase_multitask_v3_1" / "best_ligase_multitask.pt"),
    )
    ligase_multitask_ckpt_v3 = first_existing(
        str(cwd / "outputs" / "ligase_multitask_v3" / "best_ligase_multitask.pt"),
    )
    ligase_multitask_ckpt_v2 = first_existing(
        str(cwd / "outputs" / "ligase_multitask_v2" / "best_ligase_multitask.pt"),
    )
    ligase_multitask_ckpt_v1 = first_existing(
        str(cwd / "outputs" / "ligase_multitask_v1" / "best_ligase_multitask.pt"),
    )
    default_ligase_multitask_ckpt = first_existing(
        ligase_multitask_ckpt_v3_1,
        ligase_multitask_ckpt_v3,
        ligase_multitask_ckpt_v2,
        ligase_multitask_ckpt_v1,
    )

    with st.sidebar:
        st.markdown(
            """
            <div style="font-size:0.98rem; line-height:1.35; color:#1f2937; font-weight:600;">
              <div>Developed by Eric Xu</div>
              <div>医药人工智能</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.subheader("Runtime")
        model_path = st.text_input("Model Path", value=default_model, help="blend_model.json or full_model.joblib")
        feature_cache = st.text_input(
            "Feature Cache (Optional)",
            value=default_cache,
            help="推荐填 feat_cache.npz 以对齐训练分布。",
        )
        if model_path and (not os.path.exists(model_path)):
            st.caption("Model path does not exist yet. Please provide a valid blend_model.json/full_model.joblib.")
        if feature_cache and (not os.path.exists(feature_cache)):
            st.caption("Feature cache not found. You can leave it empty, but topology priors will use fallback values.")
        esm_model_name = st.text_input("ESM Model", value="facebook/esm2_t12_35M_UR50D")
        device_arg = st.selectbox("Device", options=["auto", "mps", "cpu"], index=0)
        st.markdown('<div class="small-note">首次加载 ESM 可能较慢，后续会走缓存。</div>', unsafe_allow_html=True)
        st.divider()
        with st.expander("Legacy Multi-task Models (可选)", expanded=False):
            legacy_ligase_path = st.text_input("Ligase Model", value=default_ligase)
            legacy_sol_path = st.text_input("Solubility Model", value=default_sol)
            legacy_cofactor_path = st.text_input("Cofactor Model", value=default_cofactor)
        with st.expander("New Ligase Multi-task Model (EC/Substrate/Metal)", expanded=True):
            ckpt_versions = {}
            if ligase_multitask_ckpt_v3_1:
                ckpt_versions["v3.1 (Recommended)"] = ligase_multitask_ckpt_v3_1
            if ligase_multitask_ckpt_v3:
                ckpt_versions["v3 (Enhanced)"] = ligase_multitask_ckpt_v3
            if ligase_multitask_ckpt_v2:
                ckpt_versions["v2 (Recommended)"] = ligase_multitask_ckpt_v2
            if ligase_multitask_ckpt_v1:
                ckpt_versions["v1 (Baseline)"] = ligase_multitask_ckpt_v1
            version_options = list(ckpt_versions.keys()) + ["Custom"]

            default_version = "Custom"
            if ligase_multitask_ckpt_v3_1:
                default_version = "v3.1 (Recommended)"
            elif ligase_multitask_ckpt_v3:
                default_version = "v3 (Enhanced)"
            elif ligase_multitask_ckpt_v2:
                default_version = "v2 (Recommended)"
            elif ligase_multitask_ckpt_v1:
                default_version = "v1 (Baseline)"

            selected_version = st.selectbox(
                "Checkpoint Version",
                options=version_options,
                index=version_options.index(default_version),
            )

            if selected_version == "Custom":
                ligase_multitask_ckpt = st.text_input(
                    "Ligase Multitask Checkpoint (Custom)",
                    value=default_ligase_multitask_ckpt,
                )
            else:
                ligase_multitask_ckpt = ckpt_versions[selected_version]
                st.text_input("Ligase Multitask Checkpoint", value=ligase_multitask_ckpt, disabled=True)

            ckpt_meta = peek_ligase_multitask_checkpoint_meta(ligase_multitask_ckpt)
            use_ckpt_thresholds = st.checkbox("Use checkpoint tuned thresholds", value=True)
            if ckpt_meta["found"]:
                st.caption(
                    "ckpt thresholds: "
                    f"sub={ckpt_meta['substrate_threshold']:.2f}, "
                    f"metal={ckpt_meta['metal_threshold']:.2f}, "
                    f"metal_presence={ckpt_meta['metal_presence_threshold']:.2f}"
                )
            if use_ckpt_thresholds:
                ligase_sub_threshold = None
                ligase_metal_threshold = None
                ligase_metal_presence_threshold = None
            else:
                ligase_sub_threshold = st.slider(
                    "Substrate Threshold",
                    min_value=0.30,
                    max_value=0.90,
                    value=float(np.clip(ckpt_meta["substrate_threshold"], 0.30, 0.90)),
                    step=0.05,
                )
                ligase_metal_threshold = st.slider(
                    "Metal Type Threshold",
                    min_value=0.30,
                    max_value=0.90,
                    value=float(np.clip(ckpt_meta["metal_threshold"], 0.30, 0.90)),
                    step=0.05,
                )
                ligase_metal_presence_threshold = st.slider(
                    "Metal Presence Threshold",
                    min_value=0.30,
                    max_value=0.90,
                    value=float(np.clip(ckpt_meta["metal_presence_threshold"], 0.30, 0.90)),
                    step=0.05,
                )

    tab0, tab1, tab2, tab3 = st.tabs(["Integrated Lab", "Single Sequence", "FASTA Batch", "PDB"])

    with tab0:
        seq_all = st.text_area(
            "Protein Sequence (for integrated analysis)",
            height=170,
            placeholder="Paste amino-acid sequence here...",
            key="seq_all",
        )
        c0a, c0b = st.columns([1, 4])
        run_all = c0a.button("Run Full Analysis", use_container_width=True)
        if run_all:
            try:
                seq_clean = clean_sequence(seq_all)

                with st.spinner("Running blend kcat model..."):
                    blend_rows = predict_records(
                        [{"id": "integrated_input", "seq": seq_clean, "topo5": None}],
                        model_path,
                        feature_cache,
                        esm_model_name,
                        device_arg,
                    )
                blend_pred = blend_rows[0]
                blend_log_kcat = float(blend_pred["pred_log_kcat"])
                blend_kcat = float(blend_pred["pred_kcat"])

                with st.spinner("Running legacy multi-task models..."):
                    legacy_runtime = load_legacy_runtime(
                        device_arg,
                        legacy_sol_path,
                        legacy_ligase_path,
                        legacy_cofactor_path,
                    )
                    legacy = predict_legacy_suite(seq_clean, legacy_runtime)

                ligase_mt = None
                if ligase_multitask_ckpt and os.path.exists(ligase_multitask_ckpt):
                    with st.spinner("Running new ligase multi-task model (EC/Substrate/Metal)..."):
                        mt_runtime = load_ligase_multitask_runtime(device_arg, ligase_multitask_ckpt)
                        if mt_runtime is not None:
                            ligase_mt = predict_ligase_multitask(
                                seq_clean,
                                mt_runtime,
                                substrate_threshold=ligase_sub_threshold,
                                metal_threshold=ligase_metal_threshold,
                                metal_presence_threshold=ligase_metal_presence_threshold,
                            )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    v = legacy["ligase_prob"]
                    st.metric("连接酶概率", "N/A" if not np.isfinite(v) else f"{v*100:.1f}%")
                with col2:
                    v = legacy["solubility_prob"]
                    st.metric("水溶性概率", "N/A" if not np.isfinite(v) else f"{v*100:.1f}%")
                with col3:
                    st.metric("辅酶偏好", legacy["cofactor_type"])
                with col4:
                    st.metric("新版 blend log_kcat", f"{blend_log_kcat:.3f}")

                st.caption(f"新版估计 kcat ≈ {blend_kcat:.3e}")
                if ligase_mt is not None:
                    col5, col6, col7 = st.columns(3)
                    with col5:
                        st.metric("EC 子类 (Top1)", ligase_mt["ec_top1"], f"{ligase_mt['ec_top1_prob']*100:.1f}%")
                    with col6:
                        st.metric("底物谱 (Top1)", _fmt_label_probs(ligase_mt["substrate_pred"], max_items=1))
                    with col7:
                        st.metric("金属依赖 (Top)", _fmt_label_probs(ligase_mt["metal_pred"], max_items=2))

                lcol, rcol = st.columns([1, 1])
                with lcol:
                    st.markdown("#### 多维性能雷达图")
                    fig = render_radar(legacy, blend_log_kcat)
                    st.pyplot(fig, use_container_width=True)
                with rcol:
                    st.markdown("#### AI 决策建议")
                    lig = legacy["ligase_prob"]
                    sol = legacy["solubility_prob"]
                    if np.isfinite(lig) and lig < 0.5:
                        st.error("该序列更可能不是连接酶，建议先做家族注释复核。")
                    else:
                        st.success("序列具备连接酶特征，可进入后续实验设计。")
                    if np.isfinite(sol):
                        if sol > 0.5:
                            st.info("预测可溶性较好，E. coli 表达可优先尝试。")
                        else:
                            st.warning("预测可溶性偏低，建议考虑低温诱导/融合标签/复性策略。")
                    if legacy["cofactor_type"] in {"ATP", "NAD+"}:
                        st.info(f"辅酶优先建议：{legacy['cofactor_type']}")
                    if ligase_mt is not None:
                        st.info(f"EC 子类预测：{ligase_mt['ec_top1']} ({ligase_mt['ec_top1_prob']*100:.1f}%)")
                        st.info(f"底物谱建议：{_fmt_label_probs(ligase_mt['substrate_pred'])}")
                        st.info(f"金属依赖建议：{_fmt_label_probs(ligase_mt['metal_pred'])}")
                        if ligase_mt.get("metal_presence_prob") is not None:
                            st.info(f"金属依赖存在概率：{ligase_mt['metal_presence_prob']*100:.1f}%")

                st.markdown("#### 综合结果表")
                merged = {
                    "id": "integrated_input",
                    "length": len(seq_clean),
                    "ligase_prob": legacy["ligase_prob"],
                    "solubility_prob": legacy["solubility_prob"],
                    "cofactor_type": legacy["cofactor_type"],
                    "atp_prob": legacy["atp_prob"],
                    "nad_prob": legacy["nad_prob"],
                    "blend_log_kcat": blend_log_kcat,
                    "blend_kcat": blend_kcat,
                }
                if ligase_mt is not None:
                    merged["ec_top1"] = ligase_mt["ec_top1"]
                    merged["ec_top1_prob"] = ligase_mt["ec_top1_prob"]
                    merged["ec_top3"] = json.dumps(ligase_mt["ec_top3"], ensure_ascii=False)
                    merged["substrate_pred"] = json.dumps(ligase_mt["substrate_pred"], ensure_ascii=False)
                    merged["metal_pred"] = json.dumps(ligase_mt["metal_pred"], ensure_ascii=False)
                    merged["metal_presence_prob"] = ligase_mt.get("metal_presence_prob", np.nan)
                st.dataframe([merged], use_container_width=True)
            except Exception as e:
                st.error(str(e))

    with tab1:
        seq = st.text_area(
            "Protein Sequence",
            height=180,
            placeholder="Paste amino-acid sequence here...",
        )
        c1, c2 = st.columns([1, 4])
        run_seq = c1.button("Predict Sequence", use_container_width=True)
        if run_seq:
            try:
                records = [{"id": "input_sequence", "seq": clean_sequence(seq), "topo5": None}]
                with st.spinner("Running prediction..."):
                    rows = predict_records(records, model_path, feature_cache, esm_model_name, device_arg)
                render_metric_cards(rows)
                st.dataframe(rows, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    data=rows_to_csv(rows),
                    file_name="sequence_prediction.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(str(e))

    with tab2:
        fasta_file = st.file_uploader("Upload FASTA", type=["fasta", "fa", "faa", "txt"], key="fasta")
        fasta_text = st.text_area("Or paste FASTA text", height=220, placeholder=">seq1\nMKK...\n>seq2\nGAS...")
        c1, c2, c3 = st.columns([1, 1, 4])
        run_fa = c1.button("Predict FASTA", use_container_width=True)
        run_fa_mt = c2.button("FASTA: EC/Sub/Metal", use_container_width=True)
        if run_fa:
            try:
                text = ""
                if fasta_file is not None:
                    text = fasta_file.read().decode("utf-8", errors="ignore")
                if fasta_text.strip():
                    text = fasta_text
                if not text.strip():
                    raise ValueError("Please upload or paste FASTA content.")
                records = parse_fasta_text(text)
                with st.spinner(f"Running batch prediction for {len(records)} sequences..."):
                    rows = predict_records(records, model_path, feature_cache, esm_model_name, device_arg)
                render_metric_cards(rows)
                st.dataframe(rows, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    data=rows_to_csv(rows),
                    file_name="fasta_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(str(e))
        if run_fa_mt:
            try:
                text = ""
                if fasta_file is not None:
                    text = fasta_file.read().decode("utf-8", errors="ignore")
                if fasta_text.strip():
                    text = fasta_text
                if not text.strip():
                    raise ValueError("Please upload or paste FASTA content.")
                records = parse_fasta_text(text)
                with st.spinner(f"Running ligase multitask prediction for {len(records)} sequences..."):
                    rows = predict_ligase_multitask_records(
                        records=records,
                        checkpoint_path=ligase_multitask_ckpt,
                        device_arg=device_arg,
                        substrate_threshold=ligase_sub_threshold,
                        metal_threshold=ligase_metal_threshold,
                        metal_presence_threshold=ligase_metal_presence_threshold,
                    )
                st.dataframe(rows, use_container_width=True)
                st.download_button(
                    "Download EC/Sub/Metal CSV",
                    data=rows_to_csv_ligase_multitask(rows),
                    file_name="fasta_ec_sub_metal_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(str(e))

    with tab3:
        pdb_file = st.file_uploader("Upload PDB", type=["pdb"], key="pdb")
        cset1, cset2 = st.columns([1, 1])
        pdb_chain = cset1.text_input("Chain ID (Optional)", value="")
        pdb_cutoff = cset2.slider("CA Distance Cutoff (Å)", min_value=6.0, max_value=15.0, value=10.0, step=0.5)
        c1, c2 = st.columns([1, 4])
        run_pdb = c1.button("Predict PDB", use_container_width=True)
        if run_pdb:
            try:
                if pdb_file is None:
                    raise ValueError("Please upload a PDB file.")
                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                    tmp.write(pdb_file.read())
                    tmp_path = tmp.name
                try:
                    seq, topo5 = parse_pdb_to_sequence_and_topology(
                        tmp_path, chain=pdb_chain.strip(), cutoff=float(pdb_cutoff)
                    )
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                records = [{"id": Path(pdb_file.name).stem, "seq": seq, "topo5": topo5}]
                with st.spinner("Running structure-aware prediction..."):
                    rows = predict_records(records, model_path, feature_cache, esm_model_name, device_arg)
                render_metric_cards(rows)
                st.dataframe(rows, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    data=rows_to_csv(rows),
                    file_name="pdb_prediction.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
