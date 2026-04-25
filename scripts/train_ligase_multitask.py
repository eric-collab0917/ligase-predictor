#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ligase_multitask import (
    LigaseMultiTaskModel,
    build_label_map,
    clean_sequence,
    get_device,
    is_explicit_none_label,
    multilabel_micro_f1,
    parse_multilabel_cell,
    unpack_multitask_outputs,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        df,
        ec_map,
        sub_map,
        metal_map,
        seq_col="sequence",
        ec_col="ec_subclass",
        sub_col="substrate_labels",
        metal_col="metal_labels",
        sep=";",
    ):
        self.samples = []
        for _, row in df.iterrows():
            seq = clean_sequence(row[seq_col])
            if not seq:
                continue

            ec_raw = str(row.get(ec_col, "")).strip()
            has_ec = bool(ec_raw and ec_raw.lower() not in {"nan", "none", "null"})
            ec_target = -100
            if has_ec:
                if ec_raw not in ec_map:
                    has_ec = False
                else:
                    ec_target = ec_map[ec_raw]

            sub_tokens = parse_multilabel_cell(row.get(sub_col, ""), sep=sep)
            has_sub = len(sub_tokens) > 0
            sub_vec = np.zeros(len(sub_map), dtype=np.float32)
            sub_mask = np.zeros(len(sub_map), dtype=np.float32)
            if has_sub and not is_explicit_none_label(sub_tokens):
                mapped = 0
                for t in sub_tokens:
                    if t in sub_map:
                        sub_vec[sub_map[t]] = 1.0
                        sub_mask[sub_map[t]] = 1.0
                        mapped += 1
                if mapped == 0:
                    has_sub = False
            elif has_sub and is_explicit_none_label(sub_tokens):
                # Explicit "none" means known-negative for all substrate labels.
                sub_mask[:] = 1.0

            metal_tokens = parse_multilabel_cell(row.get(metal_col, ""), sep=sep)
            has_metal = len(metal_tokens) > 0
            metal_vec = np.zeros(len(metal_map), dtype=np.float32)
            metal_present_target = 0.0
            if has_metal and is_explicit_none_label(metal_tokens):
                metal_present_target = 0.0
            elif has_metal:
                mapped = 0
                for t in metal_tokens:
                    if t in metal_map:
                        metal_vec[metal_map[t]] = 1.0
                        mapped += 1
                if mapped > 0:
                    metal_present_target = 1.0
                else:
                    has_metal = False

            self.samples.append(
                {
                    "sequence": seq,
                    "ec_target": ec_target,
                    "has_ec": float(has_ec),
                    "sub_target": sub_vec,
                    "sub_mask": sub_mask,
                    "has_sub": float(has_sub and sub_mask.sum() > 0),
                    "metal_target": metal_vec,
                    "has_metal": float(has_metal),
                    "metal_present_target": float(metal_present_target),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(tokenizer, max_length):
    def _collate(batch):
        seqs = [x["sequence"] for x in batch]
        enc = tokenizer(
            seqs,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ec_target = torch.tensor([x["ec_target"] for x in batch], dtype=torch.long)
        has_ec = torch.tensor([x["has_ec"] for x in batch], dtype=torch.float32)
        has_sub = torch.tensor([x["has_sub"] for x in batch], dtype=torch.float32)
        has_metal = torch.tensor([x["has_metal"] for x in batch], dtype=torch.float32)
        metal_present_target = torch.tensor([x["metal_present_target"] for x in batch], dtype=torch.float32)

        sub_target = torch.tensor(np.stack([x["sub_target"] for x in batch], axis=0), dtype=torch.float32)
        sub_mask = torch.tensor(np.stack([x["sub_mask"] for x in batch], axis=0), dtype=torch.float32)
        metal_target = torch.tensor(np.stack([x["metal_target"] for x in batch], axis=0), dtype=torch.float32)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "ec_target": ec_target,
            "has_ec": has_ec,
            "sub_target": sub_target,
            "sub_mask": sub_mask,
            "has_sub": has_sub,
            "metal_target": metal_target,
            "has_metal": has_metal,
            "metal_present_target": metal_present_target,
        }

    return _collate


def compute_pos_weight(df, col, label_map, sep=";"):
    if len(label_map) == 0:
        return None
    rows = []
    for _, row in df.iterrows():
        tokens = parse_multilabel_cell(row.get(col, ""), sep=sep)
        if len(tokens) == 0:
            continue
        vec = np.zeros(len(label_map), dtype=np.float32)
        if not is_explicit_none_label(tokens):
            mapped = 0
            for t in tokens:
                if t in label_map:
                    vec[label_map[t]] = 1.0
                    mapped += 1
            if mapped == 0:
                continue
        rows.append(vec)
    if len(rows) == 0:
        return torch.ones(len(label_map), dtype=torch.float32)
    arr = np.stack(rows, axis=0)
    pos = arr.sum(axis=0)
    neg = arr.shape[0] - pos
    pos = np.clip(pos, 1e-6, None)
    w = neg / pos
    w = np.clip(w, 1.0, 50.0)
    return torch.tensor(w, dtype=torch.float32)


def compute_ec_class_weight(df, ec_col, ec_map):
    if len(ec_map) == 0:
        return None
    counts = np.zeros(len(ec_map), dtype=np.float32)
    for v in df[ec_col].fillna("").astype(str):
        vv = v.strip()
        if vv in ec_map:
            counts[ec_map[vv]] += 1.0
    valid = counts > 0
    if not valid.any():
        return None
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (len(counts) * counts)
    weights = np.clip(weights, 0.2, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_ec_log_prior(df, ec_col, ec_map, smoothing=1.0):
    if len(ec_map) == 0:
        return None
    counts = np.zeros(len(ec_map), dtype=np.float64)
    for v in df[ec_col].fillna("").astype(str):
        vv = v.strip()
        if vv in ec_map:
            counts[ec_map[vv]] += 1.0
    smooth = float(max(smoothing, 0.0))
    probs = counts + smooth
    probs = probs / np.clip(probs.sum(), 1e-12, None)
    return torch.tensor(np.log(np.clip(probs, 1e-12, None)), dtype=torch.float32)


def compute_ec_cb_class_weight(df, ec_col, ec_map, beta=0.999):
    if len(ec_map) == 0:
        return None
    beta = float(np.clip(beta, 0.0, 0.999999))
    counts = np.zeros(len(ec_map), dtype=np.float64)
    for v in df[ec_col].fillna("").astype(str):
        vv = v.strip()
        if vv in ec_map:
            counts[ec_map[vv]] += 1.0
    counts = np.clip(counts, 1.0, None)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.clip(effective_num, 1e-12, None)
    weights = weights / np.clip(weights.mean(), 1e-12, None)
    weights = np.clip(weights, 0.2, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metal_presence_pos_weight(df, metal_col, metal_map, sep=";"):
    pos = 0
    neg = 0
    for _, row in df.iterrows():
        tokens = parse_multilabel_cell(row.get(metal_col, ""), sep=sep)
        if len(tokens) == 0:
            continue
        if is_explicit_none_label(tokens):
            neg += 1
            continue
        mapped = any(t in metal_map for t in tokens)
        if mapped:
            pos += 1
    if pos <= 0:
        w = 1.0
    else:
        w = float(np.clip(neg / max(pos, 1), 1.0, 50.0))
    return torch.tensor([w], dtype=torch.float32)


def focal_cross_entropy(logits, target, class_weight=None, gamma=2.0):
    ce = F.cross_entropy(logits, target, weight=class_weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def apply_ec_logit_adjustment(ec_logits, ec_log_prior=None, tau=0.0):
    if ec_log_prior is None or float(tau) <= 0.0:
        return ec_logits
    return ec_logits - float(tau) * ec_log_prior.unsqueeze(0)


def compute_ec_loss(
    ec_logits,
    ec_target,
    ec_class_weight=None,
    ec_cb_class_weight=None,
    ec_loss_type="ce",
    ec_focal_gamma=2.0,
):
    if ec_loss_type == "focal":
        return focal_cross_entropy(ec_logits, ec_target, class_weight=ec_class_weight, gamma=ec_focal_gamma)
    if ec_loss_type == "cb_focal":
        cw = ec_cb_class_weight if ec_cb_class_weight is not None else ec_class_weight
        return focal_cross_entropy(ec_logits, ec_target, class_weight=cw, gamma=ec_focal_gamma)
    return F.cross_entropy(ec_logits, ec_target, weight=ec_class_weight)


def masked_bce_with_logits(logits, target, mask, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    w = mask.to(loss.dtype)
    denom = torch.clamp(w.sum(), min=1.0)
    return (loss * w).sum() / denom


def build_threshold_grid(min_thr, max_thr, step):
    lo = float(np.clip(min(min_thr, max_thr), 0.01, 0.99))
    hi = float(np.clip(max(min_thr, max_thr), 0.01, 0.99))
    step = float(max(step, 1e-3))
    vals = np.arange(lo, hi + 1e-8, step, dtype=np.float32)
    if vals.size == 0:
        vals = np.array([0.5], dtype=np.float32)
    return [float(v) for v in vals]


def multilabel_micro_f1_masked(y_true, y_pred, y_mask=None):
    if y_true is None or y_pred is None or y_true.size == 0:
        return float("nan")
    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred, dtype=np.int32)
    if y_mask is None:
        return float(multilabel_micro_f1(yt, yp))
    m = np.asarray(y_mask) > 0.5
    if not np.any(m):
        return float("nan")
    return float(f1_score(yt[m].reshape(-1), yp[m].reshape(-1), average="binary", zero_division=0))


def apply_thresholds(y_prob, threshold):
    yp = np.asarray(y_prob, dtype=np.float32)
    if np.isscalar(threshold):
        return (yp >= float(threshold)).astype(np.int32)
    thr = np.asarray(threshold, dtype=np.float32).reshape(1, -1)
    return (yp >= thr).astype(np.int32)


def search_best_threshold(y_true, y_prob, candidates, y_mask=None):
    if y_true is None or y_prob is None or y_true.size == 0:
        return {"threshold": 0.5, "f1": float("nan")}
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        y_pred = apply_thresholds(y_prob, float(t))
        f1 = multilabel_micro_f1_masked(y_true, y_pred, y_mask=y_mask)
        if np.isnan(f1):
            continue
        if f1 > best_f1 + 1e-12:
            best_f1 = float(f1)
            best_t = float(t)
        elif abs(f1 - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5):
            best_t = float(t)
    if best_f1 < 0:
        best_f1 = float("nan")
    return {"threshold": best_t, "f1": best_f1}


def search_best_threshold_per_label(y_true, y_prob, candidates, y_mask=None):
    if y_true is None or y_prob is None or y_true.size == 0:
        return {"thresholds": [], "micro_f1": float("nan")}
    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_prob, dtype=np.float32)
    n_labels = yp.shape[1]
    if y_mask is None:
        y_mask_arr = np.ones_like(yt, dtype=np.float32)
    else:
        y_mask_arr = np.asarray(y_mask, dtype=np.float32)

    thr = np.full(n_labels, 0.5, dtype=np.float32)
    for j in range(n_labels):
        m = y_mask_arr[:, j] > 0.5
        if not np.any(m):
            continue
        yt_j = yt[m, j]
        yp_j = yp[m, j]
        best_t = 0.5
        best_f1 = -1.0
        for t in candidates:
            pred_j = (yp_j >= float(t)).astype(np.int32)
            f1 = float(f1_score(yt_j, pred_j, average="binary", zero_division=0))
            if f1 > best_f1 + 1e-12:
                best_f1 = f1
                best_t = float(t)
            elif abs(f1 - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5):
                best_t = float(t)
        thr[j] = best_t

    pred = apply_thresholds(yp, thr)
    micro = multilabel_micro_f1_masked(yt, pred, y_mask=y_mask_arr)
    return {"thresholds": thr.tolist(), "micro_f1": float(micro)}


def search_best_metal_two_stage(
    y_true,
    y_prob,
    presence_prob,
    presence_candidates,
    type_candidates,
    type_thresholds=None,
):
    if y_true is None or y_prob is None or presence_prob is None or y_true.size == 0:
        return {
            "metal_presence_threshold": 0.5,
            "metal_type_threshold": 0.5,
            "f1": float("nan"),
        }
    best_presence_t = 0.5
    best_type_t = 0.5
    best_f1 = -1.0
    for p_t in presence_candidates:
        active = presence_prob >= p_t
        for t_t in type_candidates:
            pred = np.zeros_like(y_true, dtype=np.int32)
            if np.any(active):
                if type_thresholds is None:
                    pred[active] = (y_prob[active] >= t_t).astype(np.int32)
                else:
                    pred[active] = apply_thresholds(y_prob[active], type_thresholds)
            f1 = float(multilabel_micro_f1(y_true, pred))
            if f1 > best_f1 + 1e-12:
                best_f1 = f1
                best_presence_t = float(p_t)
                best_type_t = float(t_t)
            elif abs(f1 - best_f1) <= 1e-12:
                old_dist = abs(best_presence_t - 0.5) + abs(best_type_t - 0.5)
                new_dist = abs(p_t - 0.5) + abs(t_t - 0.5)
                if new_dist < old_dist:
                    best_presence_t = float(p_t)
                    best_type_t = float(t_t)
    return {
        "metal_presence_threshold": best_presence_t,
        "metal_type_threshold": best_type_t,
        "f1": best_f1,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    sub_pos_weight,
    metal_pos_weight,
    ec_class_weight=None,
    ec_cb_class_weight=None,
    ec_log_prior=None,
    ec_logit_adjust_tau=0.0,
    ec_loss_weight=1.0,
    sub_loss_weight=1.0,
    metal_loss_weight=1.0,
    substrate_loss_mode="dense",
    ec_loss_type="ce",
    ec_focal_gamma=2.0,
    metal_two_stage=False,
    metal_presence_pos_weight=None,
    metal_presence_loss_weight=1.0,
    metal_type_loss_weight=1.0,
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    sub_pw = sub_pos_weight.to(device) if sub_pos_weight is not None else None
    metal_pw = metal_pos_weight.to(device) if metal_pos_weight is not None else None
    ec_cw = ec_class_weight.to(device) if ec_class_weight is not None else None
    ec_cb_cw = ec_cb_class_weight.to(device) if ec_cb_class_weight is not None else None
    ec_lp = ec_log_prior.to(device) if ec_log_prior is not None else None
    metal_presence_pw = metal_presence_pos_weight.to(device) if metal_presence_pos_weight is not None else None

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ec_target = batch["ec_target"].to(device)
        has_ec = batch["has_ec"].to(device)
        sub_target = batch["sub_target"].to(device)
        sub_mask = batch["sub_mask"].to(device)
        has_sub = batch["has_sub"].to(device)
        metal_target = batch["metal_target"].to(device)
        has_metal = batch["has_metal"].to(device)
        metal_present_target = batch["metal_present_target"].to(device)

        outputs = model(input_ids, attention_mask)
        ec_logits, sub_logits, metal_logits, metal_presence_logits = unpack_multitask_outputs(outputs)

        loss = torch.tensor(0.0, device=device)

        if has_ec.sum() > 0:
            idx = has_ec > 0.5
            ec_logits_adj = apply_ec_logit_adjustment(ec_logits[idx], ec_log_prior=ec_lp, tau=ec_logit_adjust_tau)
            loss_ec = compute_ec_loss(
                ec_logits_adj,
                ec_target[idx],
                ec_class_weight=ec_cw,
                ec_cb_class_weight=ec_cb_cw,
                ec_loss_type=ec_loss_type,
                ec_focal_gamma=ec_focal_gamma,
            )
            loss = loss + ec_loss_weight * loss_ec

        if sub_logits.shape[1] > 0:
            if substrate_loss_mode == "masked":
                if torch.sum(sub_mask) > 0:
                    loss_sub = masked_bce_with_logits(sub_logits, sub_target, sub_mask, pos_weight=sub_pw)
                    loss = loss + sub_loss_weight * loss_sub
            else:
                if has_sub.sum() > 0:
                    idx = has_sub > 0.5
                    loss_sub = F.binary_cross_entropy_with_logits(sub_logits[idx], sub_target[idx], pos_weight=sub_pw)
                    loss = loss + sub_loss_weight * loss_sub

        if metal_logits.shape[1] > 0 and has_metal.sum() > 0:
            idx_known = has_metal > 0.5
            if metal_two_stage and metal_presence_logits is not None:
                loss_presence = F.binary_cross_entropy_with_logits(
                    metal_presence_logits[idx_known],
                    metal_present_target[idx_known],
                    pos_weight=metal_presence_pw,
                )
                loss = loss + metal_loss_weight * metal_presence_loss_weight * loss_presence

                idx_pos = idx_known & (metal_present_target > 0.5)
                if idx_pos.sum() > 0:
                    loss_metal_type = F.binary_cross_entropy_with_logits(
                        metal_logits[idx_pos],
                        metal_target[idx_pos],
                        pos_weight=metal_pw,
                    )
                    loss = loss + metal_loss_weight * metal_type_loss_weight * loss_metal_type
            else:
                loss_metal = F.binary_cross_entropy_with_logits(metal_logits[idx_known], metal_target[idx_known], pos_weight=metal_pw)
                loss = loss + metal_loss_weight * loss_metal

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    sub_pos_weight,
    metal_pos_weight,
    ec_class_weight=None,
    ec_cb_class_weight=None,
    ec_log_prior=None,
    ec_logit_adjust_tau=0.0,
    ec_loss_weight=1.0,
    sub_loss_weight=1.0,
    metal_loss_weight=1.0,
    substrate_loss_mode="dense",
    ec_loss_type="ce",
    ec_focal_gamma=2.0,
    metal_two_stage=False,
    metal_presence_pos_weight=None,
    metal_presence_loss_weight=1.0,
    metal_type_loss_weight=1.0,
    substrate_threshold=0.5,
    metal_threshold=0.5,
    metal_presence_threshold=0.5,
    return_buffers=False,
):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    sub_pw = sub_pos_weight.to(device) if sub_pos_weight is not None else None
    metal_pw = metal_pos_weight.to(device) if metal_pos_weight is not None else None
    ec_cw = ec_class_weight.to(device) if ec_class_weight is not None else None
    ec_cb_cw = ec_cb_class_weight.to(device) if ec_cb_class_weight is not None else None
    ec_lp = ec_log_prior.to(device) if ec_log_prior is not None else None
    metal_presence_pw = metal_presence_pos_weight.to(device) if metal_presence_pos_weight is not None else None

    ec_true, ec_pred, ec_prob = [], [], []
    sub_true, sub_prob, sub_mask_buf = [], [], []
    metal_true, metal_prob = [], []
    metal_presence_true, metal_presence_prob = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ec_target = batch["ec_target"].to(device)
        has_ec = batch["has_ec"].to(device)
        sub_target = batch["sub_target"].to(device)
        sub_mask = batch["sub_mask"].to(device)
        has_sub = batch["has_sub"].to(device)
        metal_target = batch["metal_target"].to(device)
        has_metal = batch["has_metal"].to(device)
        metal_present_target = batch["metal_present_target"].to(device)

        outputs = model(input_ids, attention_mask)
        ec_logits, sub_logits, metal_logits, metal_presence_logits = unpack_multitask_outputs(outputs)

        loss = torch.tensor(0.0, device=device)

        if has_ec.sum() > 0:
            idx = has_ec > 0.5
            ec_logits_adj = apply_ec_logit_adjustment(ec_logits[idx], ec_log_prior=ec_lp, tau=ec_logit_adjust_tau)
            loss_ec = compute_ec_loss(
                ec_logits_adj,
                ec_target[idx],
                ec_class_weight=ec_cw,
                ec_cb_class_weight=ec_cb_cw,
                ec_loss_type=ec_loss_type,
                ec_focal_gamma=ec_focal_gamma,
            )
            loss = loss + ec_loss_weight * loss_ec
            ec_true.append(ec_target[idx].cpu().numpy())
            ec_pred.append(torch.argmax(ec_logits_adj, dim=1).cpu().numpy())
            ec_prob.append(torch.softmax(ec_logits_adj, dim=1).cpu().numpy())

        if sub_logits.shape[1] > 0:
            if substrate_loss_mode == "masked":
                if torch.sum(sub_mask) > 0:
                    loss_sub = masked_bce_with_logits(sub_logits, sub_target, sub_mask, pos_weight=sub_pw)
                    loss = loss + sub_loss_weight * loss_sub
                    sub_true.append(sub_target.cpu().numpy())
                    sub_prob.append(torch.sigmoid(sub_logits).cpu().numpy())
                    sub_mask_buf.append(sub_mask.cpu().numpy())
            else:
                if has_sub.sum() > 0:
                    idx = has_sub > 0.5
                    loss_sub = F.binary_cross_entropy_with_logits(sub_logits[idx], sub_target[idx], pos_weight=sub_pw)
                    loss = loss + sub_loss_weight * loss_sub
                    sub_true.append(sub_target[idx].cpu().numpy())
                    sub_prob.append(torch.sigmoid(sub_logits[idx]).cpu().numpy())

        if metal_logits.shape[1] > 0 and has_metal.sum() > 0:
            idx_known = has_metal > 0.5
            if metal_two_stage and metal_presence_logits is not None:
                loss_presence = F.binary_cross_entropy_with_logits(
                    metal_presence_logits[idx_known],
                    metal_present_target[idx_known],
                    pos_weight=metal_presence_pw,
                )
                loss = loss + metal_loss_weight * metal_presence_loss_weight * loss_presence

                idx_pos = idx_known & (metal_present_target > 0.5)
                if idx_pos.sum() > 0:
                    loss_metal_type = F.binary_cross_entropy_with_logits(
                        metal_logits[idx_pos],
                        metal_target[idx_pos],
                        pos_weight=metal_pw,
                    )
                    loss = loss + metal_loss_weight * metal_type_loss_weight * loss_metal_type

                metal_true.append(metal_target[idx_known].cpu().numpy())
                metal_prob.append(torch.sigmoid(metal_logits[idx_known]).cpu().numpy())
                metal_presence_true.append(metal_present_target[idx_known].cpu().numpy())
                metal_presence_prob.append(torch.sigmoid(metal_presence_logits[idx_known]).cpu().numpy())
            else:
                loss_metal = F.binary_cross_entropy_with_logits(metal_logits[idx_known], metal_target[idx_known], pos_weight=metal_pw)
                loss = loss + metal_loss_weight * loss_metal
                metal_true.append(metal_target[idx_known].cpu().numpy())
                metal_prob.append(torch.sigmoid(metal_logits[idx_known]).cpu().numpy())

        total_loss += float(loss.item())
        n_batches += 1

    metrics = {"val_loss": total_loss / max(n_batches, 1)}

    if ec_true:
        yt = np.concatenate(ec_true)
        yp = np.concatenate(ec_pred)
        metrics["ec_acc"] = float((yt == yp).mean())
        metrics["ec_macro_f1"] = float(f1_score(yt, yp, average="macro", zero_division=0))
    else:
        metrics["ec_acc"] = float("nan")
        metrics["ec_macro_f1"] = float("nan")

    sub_true_arr = None
    sub_prob_arr = None
    sub_mask_arr = None
    if sub_true:
        sub_true_arr = np.concatenate(sub_true, axis=0)
        sub_prob_arr = np.concatenate(sub_prob, axis=0)
        sub_pred = apply_thresholds(sub_prob_arr, substrate_threshold)
        if substrate_loss_mode == "masked":
            sub_mask_arr = np.concatenate(sub_mask_buf, axis=0) if sub_mask_buf else None
            metrics["substrate_micro_f1"] = float(multilabel_micro_f1_masked(sub_true_arr, sub_pred, y_mask=sub_mask_arr))
        else:
            metrics["substrate_micro_f1"] = float(multilabel_micro_f1(sub_true_arr, sub_pred))
    else:
        metrics["substrate_micro_f1"] = float("nan")

    metal_true_arr = None
    metal_prob_arr = None
    metal_presence_true_arr = None
    metal_presence_prob_arr = None
    if metal_true:
        metal_true_arr = np.concatenate(metal_true, axis=0)
        metal_prob_arr = np.concatenate(metal_prob, axis=0)
        if metal_two_stage and metal_presence_true:
            metal_presence_true_arr = np.concatenate(metal_presence_true, axis=0)
            metal_presence_prob_arr = np.concatenate(metal_presence_prob, axis=0)
            active = metal_presence_prob_arr >= metal_presence_threshold
            metal_pred = np.zeros_like(metal_true_arr, dtype=np.int32)
            if np.any(active):
                metal_pred[active] = (metal_prob_arr[active] >= metal_threshold).astype(np.int32)
            metrics["metal_presence_acc"] = float(
                ((metal_presence_prob_arr >= metal_presence_threshold).astype(np.int32) == metal_presence_true_arr.astype(np.int32)).mean()
            )
        else:
            metal_pred = (metal_prob_arr >= metal_threshold).astype(np.int32)
            metrics["metal_presence_acc"] = float("nan")
        metrics["metal_micro_f1"] = float(multilabel_micro_f1(metal_true_arr, metal_pred))
    else:
        metrics["metal_micro_f1"] = float("nan")
        metrics["metal_presence_acc"] = float("nan")

    if return_buffers:
        return metrics, {
            "sub_true": sub_true_arr,
            "sub_prob": sub_prob_arr,
            "sub_mask": sub_mask_arr,
            "metal_true": metal_true_arr,
            "metal_prob": metal_prob_arr,
            "metal_presence_true": metal_presence_true_arr,
            "metal_presence_prob": metal_presence_prob_arr,
            "ec_true": np.concatenate(ec_true) if ec_true else None,
            "ec_prob": np.concatenate(ec_prob, axis=0) if ec_prob else None,
        }
    return metrics


def build_label_maps(df, ec_col, sub_col, metal_col, sep):
    ec_items = []
    sub_items = []
    metal_items = []

    for _, row in df.iterrows():
        ec = str(row.get(ec_col, "")).strip()
        if ec and ec.lower() not in {"nan", "none", "null"}:
            ec_items.append(ec)

        st = parse_multilabel_cell(row.get(sub_col, ""), sep=sep)
        if st and not is_explicit_none_label(st):
            sub_items.extend(st)

        mt = parse_multilabel_cell(row.get(metal_col, ""), sep=sep)
        if mt and not is_explicit_none_label(mt):
            metal_items.extend(mt)

    ec_map = build_label_map(ec_items)
    sub_map = build_label_map(sub_items)
    metal_map = build_label_map(metal_items)
    return ec_map, sub_map, metal_map


def split_df(df, valid_size, seed, ec_col):
    ec_vals = df[ec_col].fillna("__MISSING__").astype(str)
    class_counts = ec_vals.value_counts()
    can_stratify = bool((class_counts >= 2).all()) and class_counts.shape[0] > 1
    stratify = ec_vals if can_stratify else None
    try:
        return train_test_split(df, test_size=valid_size, random_state=seed, shuffle=True, stratify=stratify)
    except Exception:
        return train_test_split(df, test_size=valid_size, random_state=seed, shuffle=True, stratify=None)


def fmt_float(v):
    try:
        if v is None:
            return "nan"
        if np.isnan(v):
            return "nan"
    except Exception:
        pass
    return f"{float(v):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with sequence/labels")
    ap.add_argument("--outdir", default="./outputs/ligase_multitask")
    ap.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--ec-col", default="ec_subclass")
    ap.add_argument("--substrate-col", default="substrate_labels")
    ap.add_argument("--metal-col", default="metal_labels")
    ap.add_argument("--sep", default=";")
    ap.add_argument("--valid-size", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    ap.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    ap.set_defaults(freeze_backbone=True)
    ap.add_argument("--unfreeze-last-n-layers", type=int, default=0)

    ap.add_argument("--ec-loss-weight", type=float, default=1.0)
    ap.add_argument("--substrate-loss-weight", type=float, default=1.0)
    ap.add_argument("--metal-loss-weight", type=float, default=1.0)
    ap.add_argument("--substrate-loss-mode", choices=["dense", "masked"], default="dense")

    ap.add_argument("--ec-loss-type", choices=["ce", "focal", "cb_focal"], default="cb_focal")
    ap.add_argument("--ec-focal-gamma", type=float, default=2.0)
    ap.add_argument("--ec-use-class-weight", dest="ec_use_class_weight", action="store_true")
    ap.add_argument("--no-ec-use-class-weight", dest="ec_use_class_weight", action="store_false")
    ap.set_defaults(ec_use_class_weight=True)
    ap.add_argument("--ec-cb-beta", type=float, default=0.999)
    ap.add_argument("--ec-logit-adjust", dest="ec_logit_adjust", action="store_true")
    ap.add_argument("--no-ec-logit-adjust", dest="ec_logit_adjust", action="store_false")
    ap.set_defaults(ec_logit_adjust=True)
    ap.add_argument("--ec-logit-adjust-tau", type=float, default=1.0)
    ap.add_argument("--ec-logit-adjust-smoothing", type=float, default=1.0)

    ap.add_argument("--metal-two-stage", dest="metal_two_stage", action="store_true")
    ap.add_argument("--no-metal-two-stage", dest="metal_two_stage", action="store_false")
    ap.set_defaults(metal_two_stage=True)
    ap.add_argument("--metal-presence-loss-weight", type=float, default=1.0)
    ap.add_argument("--metal-type-loss-weight", type=float, default=1.0)

    ap.add_argument("--search-thresholds", dest="search_thresholds", action="store_true")
    ap.add_argument("--no-search-thresholds", dest="search_thresholds", action="store_false")
    ap.set_defaults(search_thresholds=True)
    ap.add_argument("--search-per-label-substrate-thresholds", dest="search_per_label_substrate_thresholds", action="store_true")
    ap.add_argument("--no-search-per-label-substrate-thresholds", dest="search_per_label_substrate_thresholds", action="store_false")
    ap.set_defaults(search_per_label_substrate_thresholds=False)
    ap.add_argument("--search-per-label-metal-thresholds", dest="search_per_label_metal_thresholds", action="store_true")
    ap.add_argument("--no-search-per-label-metal-thresholds", dest="search_per_label_metal_thresholds", action="store_false")
    ap.set_defaults(search_per_label_metal_thresholds=True)
    ap.add_argument("--threshold-min", type=float, default=0.2)
    ap.add_argument("--threshold-max", type=float, default=0.8)
    ap.add_argument("--threshold-step", type=float, default=0.05)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--early-stop-metric", choices=["val_loss", "ec_macro_f1", "ec_acc"], default="ec_macro_f1")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.seq_col not in df.columns:
        raise KeyError(f"Sequence column not found: {args.seq_col}")

    df = df.copy()
    df[args.seq_col] = df[args.seq_col].astype(str).map(clean_sequence)
    df = df[df[args.seq_col].str.len() > 0].reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(f"Too few valid sequences after cleaning: {len(df)}")

    ec_map, sub_map, metal_map = build_label_maps(df, args.ec_col, args.substrate_col, args.metal_col, args.sep)
    if len(ec_map) == 0:
        raise ValueError("No valid EC subclass labels found.")
    if len(sub_map) == 0:
        print("[Warn] No substrate labels found; substrate head will be size 1 and ignored in loss.")
    if len(metal_map) == 0:
        print("[Warn] No metal labels found; metal head will be size 1 and ignored in loss.")

    train_df, val_df = split_df(df, args.valid_size, args.seed, args.ec_col)
    print(f"[Info] train={len(train_df)}, val={len(val_df)}")
    print(f"[Info] EC classes={len(ec_map)}, substrate classes={len(sub_map)}, metal classes={len(metal_map)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = MultiTaskDataset(
        train_df,
        ec_map,
        sub_map,
        metal_map,
        seq_col=args.seq_col,
        ec_col=args.ec_col,
        sub_col=args.substrate_col,
        metal_col=args.metal_col,
        sep=args.sep,
    )
    val_ds = MultiTaskDataset(
        val_df,
        ec_map,
        sub_map,
        metal_map,
        seq_col=args.seq_col,
        ec_col=args.ec_col,
        sub_col=args.substrate_col,
        metal_col=args.metal_col,
        sep=args.sep,
    )

    collate_fn = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = get_device(args.device)
    model = LigaseMultiTaskModel(
        model_name=args.model_name,
        num_ec=len(ec_map),
        num_substrate=len(sub_map),
        num_metal=len(metal_map),
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        metal_two_stage=args.metal_two_stage,
    ).to(device)

    sub_pos_weight = compute_pos_weight(train_df, args.substrate_col, sub_map, sep=args.sep)
    metal_pos_weight = compute_pos_weight(train_df, args.metal_col, metal_map, sep=args.sep)
    ec_class_weight = compute_ec_class_weight(train_df, args.ec_col, ec_map) if args.ec_use_class_weight else None
    ec_cb_class_weight = (
        compute_ec_cb_class_weight(train_df, args.ec_col, ec_map, beta=args.ec_cb_beta)
        if args.ec_loss_type == "cb_focal"
        else None
    )
    ec_log_prior = (
        compute_ec_log_prior(train_df, args.ec_col, ec_map, smoothing=args.ec_logit_adjust_smoothing)
        if args.ec_logit_adjust and float(args.ec_logit_adjust_tau) > 0.0
        else None
    )
    metal_presence_pos_weight = (
        compute_metal_presence_pos_weight(train_df, args.metal_col, metal_map, sep=args.sep)
        if args.metal_two_stage
        else None
    )

    if ec_class_weight is not None:
        print(f"[Info] EC class-weight enabled, min={ec_class_weight.min().item():.3f}, max={ec_class_weight.max().item():.3f}")
    if ec_cb_class_weight is not None:
        print(
            f"[Info] EC CB weight enabled (beta={args.ec_cb_beta:.6f}), "
            f"min={ec_cb_class_weight.min().item():.3f}, max={ec_cb_class_weight.max().item():.3f}"
        )
    if ec_log_prior is not None:
        print(f"[Info] EC logit adjustment enabled, tau={args.ec_logit_adjust_tau:.3f}")
    if metal_presence_pos_weight is not None:
        print(f"[Info] Metal presence pos_weight={metal_presence_pos_weight.item():.3f}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    threshold_grid = build_threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)

    if args.early_stop_metric == "val_loss":
        best_score = float("inf")
        better = lambda cur, best: cur < best - 1e-12
    else:
        best_score = -float("inf")
        better = lambda cur, best: cur > best + 1e-12

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            sub_pos_weight=sub_pos_weight,
            metal_pos_weight=metal_pos_weight,
            ec_class_weight=ec_class_weight,
            ec_cb_class_weight=ec_cb_class_weight,
            ec_log_prior=ec_log_prior,
            ec_logit_adjust_tau=args.ec_logit_adjust_tau,
            ec_loss_weight=args.ec_loss_weight,
            sub_loss_weight=args.substrate_loss_weight,
            metal_loss_weight=args.metal_loss_weight,
            substrate_loss_mode=args.substrate_loss_mode,
            ec_loss_type=args.ec_loss_type,
            ec_focal_gamma=args.ec_focal_gamma,
            metal_two_stage=args.metal_two_stage,
            metal_presence_pos_weight=metal_presence_pos_weight,
            metal_presence_loss_weight=args.metal_presence_loss_weight,
            metal_type_loss_weight=args.metal_type_loss_weight,
        )

        val_metrics, val_buf = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            sub_pos_weight=sub_pos_weight,
            metal_pos_weight=metal_pos_weight,
            ec_class_weight=ec_class_weight,
            ec_cb_class_weight=ec_cb_class_weight,
            ec_log_prior=ec_log_prior,
            ec_logit_adjust_tau=args.ec_logit_adjust_tau,
            ec_loss_weight=args.ec_loss_weight,
            sub_loss_weight=args.substrate_loss_weight,
            metal_loss_weight=args.metal_loss_weight,
            substrate_loss_mode=args.substrate_loss_mode,
            ec_loss_type=args.ec_loss_type,
            ec_focal_gamma=args.ec_focal_gamma,
            metal_two_stage=args.metal_two_stage,
            metal_presence_pos_weight=metal_presence_pos_weight,
            metal_presence_loss_weight=args.metal_presence_loss_weight,
            metal_type_loss_weight=args.metal_type_loss_weight,
            substrate_threshold=0.5,
            metal_threshold=0.5,
            metal_presence_threshold=0.5,
            return_buffers=True,
        )

        sub_best = {"threshold": 0.5, "f1": val_metrics.get("substrate_micro_f1", float("nan"))}
        sub_best_per_label = {"thresholds": [], "micro_f1": val_metrics.get("substrate_micro_f1", float("nan"))}
        metal_best = {
            "metal_type_threshold": 0.5,
            "metal_presence_threshold": 0.5,
            "f1": val_metrics.get("metal_micro_f1", float("nan")),
        }
        metal_best_per_label = {"thresholds": [], "micro_f1": val_metrics.get("metal_micro_f1", float("nan"))}

        if args.search_thresholds:
            sub_best = search_best_threshold(
                val_buf["sub_true"],
                val_buf["sub_prob"],
                threshold_grid,
                y_mask=val_buf.get("sub_mask"),
            )
            if args.search_per_label_substrate_thresholds:
                sub_best_per_label = search_best_threshold_per_label(
                    val_buf["sub_true"],
                    val_buf["sub_prob"],
                    threshold_grid,
                    y_mask=val_buf.get("sub_mask"),
                )
                if not np.isnan(sub_best_per_label["micro_f1"]):
                    sub_best["f1"] = float(sub_best_per_label["micro_f1"])
            if args.metal_two_stage:
                if args.search_per_label_metal_thresholds:
                    metal_best_per_label = search_best_threshold_per_label(
                        val_buf["metal_true"],
                        val_buf["metal_prob"],
                        threshold_grid,
                    )
                metal_best = search_best_metal_two_stage(
                    val_buf["metal_true"],
                    val_buf["metal_prob"],
                    val_buf["metal_presence_prob"],
                    threshold_grid,
                    threshold_grid,
                    type_thresholds=metal_best_per_label["thresholds"] if metal_best_per_label["thresholds"] else None,
                )
                if metal_best_per_label["thresholds"]:
                    pred = np.zeros_like(val_buf["metal_true"], dtype=np.int32)
                    active = val_buf["metal_presence_prob"] >= metal_best["metal_presence_threshold"]
                    if np.any(active):
                        pred[active] = apply_thresholds(val_buf["metal_prob"][active], metal_best_per_label["thresholds"])
                    metal_best["f1"] = float(multilabel_micro_f1(val_buf["metal_true"], pred))
            else:
                mt = search_best_threshold(val_buf["metal_true"], val_buf["metal_prob"], threshold_grid)
                if args.search_per_label_metal_thresholds:
                    metal_best_per_label = search_best_threshold_per_label(
                        val_buf["metal_true"],
                        val_buf["metal_prob"],
                        threshold_grid,
                    )
                metal_best = {
                    "metal_type_threshold": mt["threshold"],
                    "metal_presence_threshold": 0.5,
                    "f1": (
                        float(metal_best_per_label["micro_f1"])
                        if metal_best_per_label["thresholds"] and not np.isnan(metal_best_per_label["micro_f1"])
                        else mt["f1"]
                    ),
                }

        val_metrics["substrate_best_threshold"] = float(sub_best["threshold"])
        val_metrics["substrate_micro_f1_best"] = float(sub_best["f1"])
        if sub_best_per_label["thresholds"]:
            val_metrics["substrate_micro_f1_best_per_label"] = float(sub_best_per_label["micro_f1"])
        val_metrics["metal_type_best_threshold"] = float(metal_best["metal_type_threshold"])
        val_metrics["metal_presence_best_threshold"] = float(metal_best["metal_presence_threshold"])
        val_metrics["metal_micro_f1_best"] = float(metal_best["f1"])
        if metal_best_per_label["thresholds"]:
            val_metrics["metal_micro_f1_best_per_label"] = float(metal_best_per_label["micro_f1"])

        row = {"epoch": epoch, "train_loss": tr_loss, **val_metrics}
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={tr_loss:.4f} val_loss={val_metrics['val_loss']:.4f} "
            f"ec_acc={fmt_float(val_metrics['ec_acc'])} "
            f"ec_macro_f1={fmt_float(val_metrics.get('ec_macro_f1'))} "
            f"sub_f1@0.5={fmt_float(val_metrics['substrate_micro_f1'])} "
            f"sub_f1@best={fmt_float(val_metrics['substrate_micro_f1_best'])} (t={val_metrics['substrate_best_threshold']:.2f}) "
            f"metal_f1@0.5={fmt_float(val_metrics['metal_micro_f1'])} "
            f"metal_f1@best={fmt_float(val_metrics['metal_micro_f1_best'])} "
            f"(t_type={val_metrics['metal_type_best_threshold']:.2f}, t_presence={val_metrics['metal_presence_best_threshold']:.2f})"
        )

        score = float(val_metrics.get(args.early_stop_metric, float("nan")))
        if np.isnan(score):
            score = -float("inf") if args.early_stop_metric != "val_loss" else float("inf")

        if better(score, best_score):
            best_score = score
            best_val = val_metrics["val_loss"]
            best_epoch = epoch
            bad_epochs = 0
            ckpt = {
                "model_state": model.state_dict(),
                "config": {
                    "model_name": args.model_name,
                    "max_length": args.max_length,
                    "dropout": args.dropout,
                    "freeze_backbone": args.freeze_backbone,
                    "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
                    "seq_col": args.seq_col,
                    "ec_col": args.ec_col,
                    "substrate_col": args.substrate_col,
                    "metal_col": args.metal_col,
                    "ec_loss_type": args.ec_loss_type,
                    "ec_focal_gamma": args.ec_focal_gamma,
                    "ec_cb_beta": args.ec_cb_beta,
                    "ec_use_class_weight": bool(args.ec_use_class_weight),
                    "ec_logit_adjust": bool(args.ec_logit_adjust),
                    "ec_logit_adjust_tau": float(args.ec_logit_adjust_tau),
                    "ec_logit_adjust_smoothing": float(args.ec_logit_adjust_smoothing),
                    "substrate_loss_mode": args.substrate_loss_mode,
                    "metal_two_stage": bool(args.metal_two_stage),
                    "metal_presence_loss_weight": args.metal_presence_loss_weight,
                    "metal_type_loss_weight": args.metal_type_loss_weight,
                    "search_thresholds": bool(args.search_thresholds),
                    "search_per_label_substrate_thresholds": bool(args.search_per_label_substrate_thresholds),
                    "search_per_label_metal_thresholds": bool(args.search_per_label_metal_thresholds),
                    "early_stop_metric": args.early_stop_metric,
                },
                "label_maps": {
                    "ec_to_idx": ec_map,
                    "substrate_to_idx": sub_map,
                    "metal_to_idx": metal_map,
                },
                "decision_thresholds": {
                    "substrate": float(val_metrics["substrate_best_threshold"]),
                    "substrate_per_label": (
                        {
                            k: float(sub_best_per_label["thresholds"][v])
                            for k, v in sorted(sub_map.items(), key=lambda kv: kv[1])
                        }
                        if sub_best_per_label["thresholds"]
                        else {}
                    ),
                    "metal_type": float(val_metrics["metal_type_best_threshold"]),
                    "metal_type_per_label": (
                        {
                            k: float(metal_best_per_label["thresholds"][v])
                            for k, v in sorted(metal_map.items(), key=lambda kv: kv[1])
                        }
                        if metal_best_per_label["thresholds"]
                        else {}
                    ),
                    "metal_presence": float(val_metrics["metal_presence_best_threshold"]),
                },
                "ec_log_prior": (
                    ec_log_prior.detach().cpu().tolist()
                    if ec_log_prior is not None
                    else []
                ),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "best_score": float(best_score),
                "best_metrics": {
                    "ec_acc": float(val_metrics.get("ec_acc", float("nan"))),
                    "ec_macro_f1": float(val_metrics.get("ec_macro_f1", float("nan"))),
                    "substrate_micro_f1_best": float(val_metrics.get("substrate_micro_f1_best", float("nan"))),
                    "metal_micro_f1_best": float(val_metrics.get("metal_micro_f1_best", float("nan"))),
                },
            }
            torch.save(ckpt, outdir / "best_ligase_multitask.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[Info] Early stopping at epoch {epoch}, best epoch={best_epoch}")
                break

    pd.DataFrame(history).to_csv(outdir / "train_history.csv", index=False)
    with open(outdir / "label_schema.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ec_classes": sorted(ec_map.keys()),
                "substrate_classes": sorted(sub_map.keys()),
                "metal_classes": sorted(metal_map.keys()),
                "threshold_search": {
                    "enabled": bool(args.search_thresholds),
                    "per_label_substrate_enabled": bool(args.search_per_label_substrate_thresholds),
                    "per_label_metal_enabled": bool(args.search_per_label_metal_thresholds),
                    "grid": threshold_grid,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[Done] Best model: {outdir / 'best_ligase_multitask.pt'}")
    print(f"[Done] Best {args.early_stop_metric}: {best_score:.4f} at epoch {best_epoch}")
    print(f"[Done] History: {outdir / 'train_history.csv'}")
    print(f"[Done] Label schema: {outdir / 'label_schema.json'}")


if __name__ == "__main__":
    main()
