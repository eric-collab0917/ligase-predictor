#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tune LightGBM + PCA for small-sample log_kcat regression.

Workflow:
1) Load dataset.pt (graphs/y) and build (or load) flattened graph features.
2) Random search on quick CV (default: 5-fold, 1 repeat).
3) Re-evaluate top-k configs on robust CV (default: 5-fold, 5 repeats).
4) Save ranked results + best config JSON.
"""

import os
import json
import math
import argparse
import random
import inspect
import warnings
from pathlib import Path

import numpy as np
import torch
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.decomposition\._base")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pt_compat(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg:
            print(
                "[Info] Detected PyTorch safe-loading block for non-tensor objects. "
                "Retrying with weights_only=False (trusted source required)."
            )
            return torch.load(path, map_location="cpu", weights_only=False)
        raise


def pearsonr_np(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(y_true, y_pred):
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    r = pearsonr_np(y_true, y_pred)
    return {"pearson_r": r, "rmse": rmse, "mae": mae, "r2": r2}


def _get_graph_attr(g, key):
    if hasattr(g, key):
        return getattr(g, key)
    if isinstance(g, dict) and key in g:
        return g[key]
    return None


def graph_to_feature(graph):
    x = _get_graph_attr(graph, "x")
    if x is None:
        raise ValueError("Graph has no node feature `x`.")
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x = x.to(dtype=torch.float32)
    n, _ = x.shape

    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    mx = x.max(dim=0).values
    mn = x.min(dim=0).values

    node_l2_mean = torch.linalg.norm(x, dim=1).mean().unsqueeze(0)
    global_mean = x.mean().unsqueeze(0)
    global_std = x.std(unbiased=False).unsqueeze(0)

    edge_index = _get_graph_attr(graph, "edge_index")
    if edge_index is not None:
        if not torch.is_tensor(edge_index):
            edge_index = torch.tensor(edge_index)
        e = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
        if e > 0:
            src = edge_index[0]
            deg = torch.bincount(src, minlength=n).float()
            deg_mean = deg.mean().unsqueeze(0)
            deg_std = deg.std(unbiased=False).unsqueeze(0)
        else:
            deg_mean = torch.tensor([0.0])
            deg_std = torch.tensor([0.0])
    else:
        e = 0
        deg_mean = torch.tensor([0.0])
        deg_std = torch.tensor([0.0])

    topo = torch.tensor([float(n), float(e), float(e) / max(float(n), 1.0)], dtype=torch.float32)

    feat = torch.cat(
        [mean, std, mx, mn, node_l2_mean, global_mean, global_std, deg_mean, deg_std, topo],
        dim=0,
    )
    return feat.detach().cpu().numpy().astype(np.float32)


def load_or_build_features(dataset_path, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        print(f"[Info] Loaded feature cache: {cache_path}")
        X = cache["X"].astype(np.float64)
        y = cache["y"].astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y

    obj = load_pt_compat(dataset_path)
    if not isinstance(obj, dict) or "graphs" not in obj or "y" not in obj:
        raise ValueError("dataset.pt must be a dict with keys: graphs, y")

    graphs = obj["graphs"]
    y = obj["y"]
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(graphs) != len(y):
        raise ValueError(f"graphs({len(graphs)}) != y({len(y)})")

    feats = [graph_to_feature(g) for g in graphs]
    X = np.stack(feats, axis=0).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if cache_path:
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"[Info] Saved feature cache: {cache_path}")
    return X, y


def sample_config(rng, pca_grid):
    return {
        "pca_dim": int(rng.choice(pca_grid)),
        "n_estimators": 5000,
        "learning_rate": float(rng.choice([0.01, 0.015, 0.02, 0.03])),
        "num_leaves": int(rng.choice([15, 23, 31, 47, 63])),
        "max_depth": int(rng.choice([-1, 4, 6, 8])),
        "min_child_samples": int(rng.choice([6, 8, 12, 20, 30])),
        "subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        "subsample_freq": 1,
        "colsample_bytree": float(rng.choice([0.6, 0.75, 0.85, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 0.05, 0.1, 0.3, 1.0])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 3.0, 8.0])),
        "min_split_gain": float(rng.choice([0.0, 0.01, 0.05, 0.1])),
    }


def config_key(cfg):
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False)


def make_lgbm(cfg, seed):
    return LGBMRegressor(
        objective="regression",
        n_jobs=8,
        random_state=seed,
        force_col_wise=True,
        verbosity=-1,
        **cfg,
    )


def evaluate_cfg(cfg, X, y, seed, n_splits, n_repeats):
    n = len(y)
    oof_sum = np.zeros(n, dtype=np.float64)
    oof_cnt = np.zeros(n, dtype=np.int32)

    if n_repeats <= 1:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X, y), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        pca_comp = min(int(cfg["pca_dim"]), X_tr_s.shape[1], X_tr_s.shape[0] - 1)
        pca_comp = max(2, pca_comp)
        pca = PCA(n_components=pca_comp, svd_solver="full", random_state=seed + fold_idx)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_va_p = pca.transform(X_va_s)

        lgb_cfg = dict(cfg)
        lgb_cfg.pop("pca_dim", None)
        model = make_lgbm(lgb_cfg, seed + fold_idx)
        model.fit(
            X_tr_p,
            y_tr,
            eval_set=[(X_va_p, y_va)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(X_va_p)
        oof_sum[va_idx] += pred
        oof_cnt[va_idx] += 1

    valid = oof_cnt > 0
    oof = np.zeros(n, dtype=np.float64)
    oof[valid] = oof_sum[valid] / oof_cnt[valid]
    return compute_metrics(y[valid], oof[valid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset.pt")
    ap.add_argument("--outdir", default="./outputs/kcat_tune")
    ap.add_argument("--feature-cache", default="", help="Optional npz cache path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=30, help="Random search trials")
    ap.add_argument("--topk", type=int, default=5, help="Top-k configs for robust re-check")
    ap.add_argument("--quick-splits", type=int, default=5)
    ap.add_argument("--quick-repeats", type=int, default=1)
    ap.add_argument("--refine-splits", type=int, default=5)
    ap.add_argument("--refine-repeats", type=int, default=5)
    ap.add_argument("--pca-grid", default="24,32,48,64,80,96,128")
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    pca_grid = [int(x.strip()) for x in args.pca_grid.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    X, y = load_or_build_features(args.dataset, args.feature_cache if args.feature_cache else None)
    print(f"[Info] X shape={X.shape}, y shape={y.shape}")

    tried = set()
    quick_results = []
    for t in range(1, args.trials + 1):
        while True:
            cfg = sample_config(rng, pca_grid)
            k = config_key(cfg)
            if k not in tried:
                tried.add(k)
                break
        m = evaluate_cfg(cfg, X, y, args.seed, args.quick_splits, args.quick_repeats)
        rec = {"trial": t, "config": cfg, "quick_metrics": m}
        quick_results.append(rec)
        print(
            f"[Quick {t:02d}/{args.trials}] "
            f"R={m['pearson_r']:.4f} RMSE={m['rmse']:.4f} "
            f"PCA={cfg['pca_dim']} leaves={cfg['num_leaves']} mcs={cfg['min_child_samples']}"
        )

    quick_sorted = sorted(
        quick_results,
        key=lambda z: (-z["quick_metrics"]["pearson_r"], z["quick_metrics"]["rmse"]),
    )
    top = quick_sorted[: max(1, args.topk)]

    refined = []
    for i, rec in enumerate(top, start=1):
        cfg = rec["config"]
        rm = evaluate_cfg(cfg, X, y, args.seed + i * 100, args.refine_splits, args.refine_repeats)
        out = {
            "rank_in_quick": i,
            "config": cfg,
            "quick_metrics": rec["quick_metrics"],
            "refine_metrics": rm,
        }
        refined.append(out)
        print(
            f"[Refine {i:02d}/{len(top)}] "
            f"R={rm['pearson_r']:.4f} RMSE={rm['rmse']:.4f} "
            f"(quick R={rec['quick_metrics']['pearson_r']:.4f})"
        )

    refined_sorted = sorted(refined, key=lambda z: (-z["refine_metrics"]["pearson_r"], z["refine_metrics"]["rmse"]))
    best = refined_sorted[0]

    summary = {
        "dataset": args.dataset,
        "n_samples": int(len(y)),
        "feature_dim": int(X.shape[1]),
        "search_space": {
            "pca_grid": pca_grid,
            "trials": args.trials,
            "topk": args.topk,
            "quick_cv": {"splits": args.quick_splits, "repeats": args.quick_repeats},
            "refine_cv": {"splits": args.refine_splits, "repeats": args.refine_repeats},
        },
        "quick_results_sorted": quick_sorted,
        "refined_results_sorted": refined_sorted,
        "best_config": best["config"],
        "best_refine_metrics": best["refine_metrics"],
    }

    summary_path = Path(args.outdir) / "tune_lgbm_summary.json"
    best_cfg_path = Path(args.outdir) / "best_lgbm_config.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best["config"], f, ensure_ascii=False, indent=2)

    print(f"[Done] Best refine metrics: {best['refine_metrics']}")
    print(f"[Done] Summary saved: {summary_path}")
    print(f"[Done] Best config saved: {best_cfg_path}")

    fit_params = inspect.signature(LGBMRegressor.fit).parameters
    if "callbacks" not in fit_params:
        print("[Warn] Current lightgbm.fit signature seems unusual; please verify LightGBM version.")


if __name__ == "__main__":
    main()
