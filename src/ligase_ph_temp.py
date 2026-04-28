#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESM-2 based regression model for predicting ligase optimal pH and temperature.

Architecture:
  ESM-2 backbone (frozen, optionally unfreeze last N layers)
  → mean pooling over residue embeddings
  → dropout
  → two independent regression heads (pH, temperature)
  → optional pH auxiliary bin-classification head

Supports masked loss: if a sample has only pH or only temperature,
the missing target is masked out during training.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import EsmModel


def clean_sequence(seq: str) -> str:
    import re
    s = re.sub(r"\s+", "", str(seq)).upper()
    if not s:
        return ""
    if re.search(r"[^ACDEFGHIKLMNPQRSTVWYBXZUOJ]", s):
        return ""
    return s


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


class LigasePhTempModel(nn.Module):
    """
    Multi-task regression model for optimal pH and temperature prediction.

    Outputs:
        ph_pred:   (batch,) predicted optimal pH
        temp_pred: (batch,) predicted optimal temperature (°C)
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        hidden_dim: int = 128,
        num_ph_bins: int = 0,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_ph_bins = int(num_ph_bins)
        try:
            self.backbone = EsmModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.backbone = EsmModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)

        self.dropout = nn.Dropout(dropout)

        # pH regression head: Linear → ReLU → Dropout → Linear → 1
        self.ph_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Temperature regression head
        self.temp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.ph_bin_head = None
        if self.num_ph_bins > 1:
            self.ph_bin_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_ph_bins),
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if freeze_backbone and unfreeze_last_n_layers > 0:
            layers = getattr(
                getattr(self.backbone, "encoder", None), "layer", None
            )
            if layers is not None and len(layers) > 0:
                n = min(unfreeze_last_n_layers, len(layers))
                for layer in layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)

        ph_pred = self.ph_head(pooled).squeeze(-1)      # (batch,)
        temp_pred = self.temp_head(pooled).squeeze(-1)   # (batch,)

        if return_aux:
            aux = {}
            if self.ph_bin_head is not None:
                aux["ph_bin_logits"] = self.ph_bin_head(pooled)
            return ph_pred, temp_pred, aux
        return ph_pred, temp_pred


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MSE loss that only considers samples where mask > 0.5.
    Returns 0 if no valid samples.
    """
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = pred[valid] - target[valid]
    loss = diff ** 2
    if sample_weight is not None:
        w = sample_weight[valid].to(loss.dtype)
        loss = loss * w
        return loss.sum() / w.sum().clamp(min=1e-12)
    return loss.mean()


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Huber (smooth L1) loss with masking. More robust to outliers than MSE.
    """
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    loss = nn.functional.huber_loss(
        pred[valid], target[valid], reduction="none", delta=delta
    )
    if sample_weight is not None:
        w = sample_weight[valid].to(loss.dtype)
        loss = loss * w
        return loss.sum() / w.sum().clamp(min=1e-12)
    return loss.mean()


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy with masking and optional class/sample weighting."""
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    loss = nn.functional.cross_entropy(
        logits[valid],
        target[valid].long(),
        reduction="none",
        weight=class_weight,
        label_smoothing=label_smoothing,
    )
    if sample_weight is not None:
        w = sample_weight[valid].to(loss.dtype)
        loss = loss * w
        return loss.sum() / w.sum().clamp(min=1e-12)
    return loss.mean()


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute MAE, RMSE, R², Pearson R for regression evaluation."""
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "pearson_r": float("nan")}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    if len(y_true) < 2:
        pearson_r = float("nan")
    else:
        from scipy.stats import pearsonr
        pearson_r, _ = pearsonr(y_true, y_pred)
        pearson_r = float(pearson_r)

    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson_r": pearson_r}
