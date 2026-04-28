#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESM-2 based multi-label classification model for predicting ligase subcellular
localization.

Architecture:
  ESM-2 backbone (frozen, optionally unfreeze last N layers)
  → mean pooling over residue embeddings
  → dropout
  → multi-label classification head (Linear → sigmoid)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import EsmModel


def clean_sequence(seq: str) -> str:
    import re
    s = re.sub(r"\s+", "", str(seq)).upper()
    if not s:
        return ""
    if re.search(r"[^ACDEFGHIKLMNPQRSTVWYBXZUOJ]", s):
        return ""
    return s


def parse_multilabel_cell(value, sep: str = ";") -> List[str]:
    if value is None:
        return []
    v = str(value).strip()
    if not v or v.lower() in {"nan", "none", "null"}:
        return []
    return [x.strip() for x in v.split(sep) if x.strip()]


def is_explicit_none_label(tokens: List[str]) -> bool:
    if len(tokens) != 1:
        return False
    return tokens[0].strip().lower() in {"none", "no", "negative", "null"}


def build_label_map(items: List[str]) -> Dict[str, int]:
    uniq = sorted(set([x for x in items if x]))
    return {name: i for i, name in enumerate(uniq)}


def get_device(device_arg: str = "auto"):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_pos_weight(label_matrix: np.ndarray) -> np.ndarray:
    """Compute per-label positive weight = (#neg / #pos). Clamp to [1, 100]."""
    pos = label_matrix.sum(axis=0)
    neg = label_matrix.shape[0] - pos
    pw = np.divide(neg, pos.clip(min=1), where=pos > 0, out=np.ones_like(pos, dtype=np.float64))
    return np.clip(pw, 1.0, 10.0)


def multilabel_f1_score(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred_bin = y_pred_bin.astype(np.int32)
    micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    per_label = {}
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0 and y_pred_bin[:, i].sum() == 0:
            per_label[i] = 1.0
        else:
            per_label[i] = float(f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0))
    return {"micro": micro, "macro": macro, "per_label": per_label}


class LigaseSubcellularModel(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        num_subcellular: int = 1,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.model_name = model_name
        try:
            self.backbone = EsmModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.backbone = EsmModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(1, num_subcellular)),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if freeze_backbone and unfreeze_last_n_layers > 0:
            layers = getattr(getattr(self.backbone, "encoder", None), "layer", None)
            if layers is not None and len(layers) > 0:
                n = min(unfreeze_last_n_layers, len(layers))
                for layer in layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return self.head(pooled)  # (batch, num_subcellular)


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
    )
    loss = loss.mean(dim=1)
    w = mask.to(loss.dtype)
    if sample_weight is not None:
        w = w * sample_weight.view(-1, 1).to(loss.dtype)
    denom = torch.clamp(w.sum(dim=1), min=1.0)
    return (loss * w.sum(dim=1) / denom).mean()
