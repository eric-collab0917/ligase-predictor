#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import EsmModel


def clean_sequence(seq: str) -> str:
    seq = re.sub(r"\s+", "", str(seq)).upper()
    if not seq:
        return ""
    if re.search(r"[^ACDEFGHIKLMNPQRSTVWYBXZUOJ]", seq):
        return ""
    return seq


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


def multilabel_micro_f1(y_true: np.ndarray, y_pred_bin: np.ndarray):
    y_true = y_true.astype(np.int32)
    y_pred_bin = y_pred_bin.astype(np.int32)
    tp = int(((y_true == 1) & (y_pred_bin == 1)).sum())
    fp = int(((y_true == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_bin == 0)).sum())
    if tp == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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


def unpack_multitask_outputs(
    outputs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 3:
            ec_logits, substrate_logits, metal_logits = outputs
            return ec_logits, substrate_logits, metal_logits, None
        if len(outputs) == 4:
            ec_logits, substrate_logits, metal_logits, metal_presence_logits = outputs
            return ec_logits, substrate_logits, metal_logits, metal_presence_logits
    raise ValueError("Unexpected model outputs; expected tuple/list of length 3 or 4.")


class LigaseMultiTaskModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_ec: int,
        num_substrate: int,
        num_metal: int,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        metal_two_stage: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.metal_two_stage = bool(metal_two_stage)
        try:
            self.backbone = EsmModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.backbone = EsmModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.ec_head = nn.Linear(hidden_size, max(1, num_ec))
        self.substrate_head = nn.Linear(hidden_size, max(1, num_substrate))
        self.metal_head = nn.Linear(hidden_size, max(1, num_metal))
        self.metal_presence_head = nn.Linear(hidden_size, 1) if self.metal_two_stage else None

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

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        ec_logits = self.ec_head(pooled)
        substrate_logits = self.substrate_head(pooled)
        metal_logits = self.metal_head(pooled)
        if self.metal_presence_head is None:
            return ec_logits, substrate_logits, metal_logits
        metal_presence_logits = self.metal_presence_head(pooled).squeeze(-1)
        return ec_logits, substrate_logits, metal_logits, metal_presence_logits
