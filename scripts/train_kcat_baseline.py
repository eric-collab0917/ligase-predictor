#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_kcat_baseline.py

Mac-friendly baseline for log_kcat regression:
1) Flatten each graph into graph-level stats from node embeddings.
2) PCA + tree ensembles (XGB/LGBM/RF).
3) K-fold / Repeated K-fold with OOF predictions.
4) Early stopping (XGB/LGBM) + model artifact saving.
"""

import os
import json
import math
import argparse
import logging
import random
import inspect
import warnings
from pathlib import Path

import numpy as np
import torch
import joblib

from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.decomposition\._base")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


def set_seed(seed: int = 42):
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


def get_device(use_mps: bool):
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_graph_attr(g, key):
    if hasattr(g, key):
        return getattr(g, key)
    if isinstance(g, dict) and key in g:
        return g[key]
    return None


def graph_to_feature(graph, device: torch.device):
    x = _get_graph_attr(graph, "x")
    if x is None:
        raise ValueError("Graph has no node feature `x`.")
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x = x.to(device=device, dtype=torch.float32)  # [N, D]
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
        edge_index = edge_index.to(device)
        e = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
        if e > 0:
            src = edge_index[0]
            deg = torch.bincount(src, minlength=n).float()
            deg_mean = deg.mean().unsqueeze(0)
            deg_std = deg.std(unbiased=False).unsqueeze(0)
        else:
            deg_mean = torch.tensor([0.0], device=device)
            deg_std = torch.tensor([0.0], device=device)
    else:
        e = 0
        deg_mean = torch.tensor([0.0], device=device)
        deg_std = torch.tensor([0.0], device=device)

    topo = torch.tensor(
        [float(n), float(e), float(e) / max(float(n), 1.0)],
        device=device,
        dtype=torch.float32,
    )

    feat = torch.cat(
        [mean, std, mx, mn, node_l2_mean, global_mean, global_std, deg_mean, deg_std, topo],
        dim=0,
    )
    return feat.detach().cpu().numpy().astype(np.float32)


def build_feature_matrix(graphs, use_mps_feature=False, log=None):
    device = get_device(use_mps_feature)
    if log:
        log.info(f"Feature extraction device: {device}")
    feats = []
    total = len(graphs)
    for i, g in enumerate(graphs):
        feats.append(graph_to_feature(g, device))
        if log and (i + 1) % max(1, total // 10) == 0:
            log.info(f"Feature extraction progress: {i + 1}/{total}")
    return np.stack(feats, axis=0)


def load_dataset(path, target_key="y", graph_key="graphs", group_key="groups"):
    obj = load_pt_compat(path)

    if isinstance(obj, dict):
        if graph_key in obj:
            graphs = obj[graph_key]
        elif "data_list" in obj:
            graphs = obj["data_list"]
        else:
            raise KeyError(f"Cannot find graph list key `{graph_key}` or `data_list`.")
        if target_key not in obj:
            raise KeyError(f"Cannot find target key `{target_key}`.")
        y = obj[target_key]
        groups = obj.get(group_key, None)
    else:
        raise ValueError("Dataset must be a dict with keys like graphs/y/(groups).")

    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if groups is not None:
        if torch.is_tensor(groups):
            groups = groups.detach().cpu().numpy()
        groups = np.asarray(groups)

    if len(graphs) != len(y):
        raise ValueError(f"graphs({len(graphs)}) != y({len(y)})")
    return graphs, y, groups


def build_model(model_name, seed, xgb_early_stopping_rounds=None):
    if model_name == "xgb":
        params = dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=4000,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.3,
            reg_lambda=3.0,
            gamma=0.1,
            tree_method="hist",
            random_state=seed,
            n_jobs=8,
        )
        if xgb_early_stopping_rounds is not None:
            init_params = inspect.signature(XGBRegressor.__init__).parameters
            if "early_stopping_rounds" in init_params:
                params["early_stopping_rounds"] = int(xgb_early_stopping_rounds)
        return XGBRegressor(**params)
    if model_name == "lgbm":
        return LGBMRegressor(
            objective="regression",
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.3,
            reg_lambda=3.0,
            force_col_wise=True,
            verbosity=-1,
            random_state=seed,
            n_jobs=8,
        )
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=1200,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=seed,
            n_jobs=8,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_model_with_overrides(model_name, seed, model_overrides=None, xgb_early_stopping_rounds=None):
    model = build_model(model_name, seed, xgb_early_stopping_rounds=xgb_early_stopping_rounds)
    if model_overrides:
        model.set_params(**model_overrides)
    return model


def fit_one_fold(model_name, model, X_tr, y_tr, X_va, y_va):
    if model_name == "xgb":
        fit_params = inspect.signature(model.fit).parameters
        if "early_stopping_rounds" in fit_params:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=120,
                verbose=False,
            )
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif model_name == "lgbm":
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_tr, y_tr)
    return model


def get_best_iter(model_name, model):
    if model_name == "xgb":
        bi = getattr(model, "best_iteration", None)
        if bi is not None and bi > 0:
            return int(bi)
    if model_name == "lgbm":
        bi = getattr(model, "best_iteration_", None)
        if bi is not None and bi > 0:
            return int(bi)
    return None


def compute_metrics(y_true, y_pred):
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    r = pearsonr_np(y_true, y_pred)
    return {"pearson_r": r, "rmse": rmse, "mae": mae, "r2": r2}


def load_model_config(path, tag):
    if not path:
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    if not isinstance(cfg_raw, dict):
        raise ValueError(f"--{tag}-config JSON must be an object/dict.")
    cfg = dict(cfg_raw)
    pca_dim_override = None
    if "pca_dim" in cfg:
        pca_dim_override = int(cfg.pop("pca_dim"))
    return (cfg if cfg else None), pca_dim_override


def search_blend_weights(y_true, pred_lgbm, pred_xgb, step=0.01):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred_lgbm = np.asarray(pred_lgbm, dtype=np.float64).reshape(-1)
    pred_xgb = np.asarray(pred_xgb, dtype=np.float64).reshape(-1)

    best = None
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        w_lgbm = i * step
        w_xgb = 1.0 - w_lgbm
        pred = w_lgbm * pred_lgbm + w_xgb * pred_xgb
        m = compute_metrics(y_true, pred)
        candidate = {
            "w_lgbm": float(w_lgbm),
            "w_xgb": float(w_xgb),
            "metrics": m,
            "pred": pred,
        }
        if best is None:
            best = candidate
            continue
        if m["pearson_r"] > best["metrics"]["pearson_r"]:
            best = candidate
            continue
        if (
            abs(m["pearson_r"] - best["metrics"]["pearson_r"]) <= 1e-12
            and m["rmse"] < best["metrics"]["rmse"]
        ):
            best = candidate
    return best


def run_cv_for_model(
    model_name,
    X,
    y,
    outdir,
    seed=42,
    n_splits=5,
    n_repeats=5,
    pca_dim=64,
    groups=None,
    use_repeated=True,
    model_overrides=None,
    return_oof=False,
    log=None,
):
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(y)
    oof_sum = np.zeros(n, dtype=np.float64)
    oof_cnt = np.zeros(n, dtype=np.int32)
    fold_records = []
    best_iters = []

    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        split_gen = splitter.split(X, y, groups)
        total_folds = n_splits
    else:
        if use_repeated and n_repeats > 1:
            splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
            total_folds = n_splits * n_repeats
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            total_folds = n_splits
        split_gen = splitter.split(X, y)

    model_dir = Path(outdir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(split_gen, start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        pca_comp = min(pca_dim, X_tr_s.shape[1], X_tr_s.shape[0] - 1)
        pca_comp = max(2, pca_comp)
        pca = PCA(n_components=pca_comp, svd_solver="full", random_state=seed)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_va_p = pca.transform(X_va_s)

        if model_name == "xgb":
            model = build_model_with_overrides(
                model_name,
                seed + fold_idx,
                model_overrides=model_overrides,
                xgb_early_stopping_rounds=120,
            )
        else:
            model = build_model_with_overrides(
                model_name, seed + fold_idx, model_overrides=model_overrides
            )
        model = fit_one_fold(model_name, model, X_tr_p, y_tr, X_va_p, y_va)

        pred = model.predict(X_va_p).astype(np.float64)
        oof_sum[va_idx] += pred
        oof_cnt[va_idx] += 1

        fold_metric = compute_metrics(y_va, pred)
        bi = get_best_iter(model_name, model)
        if bi is not None:
            best_iters.append(bi)

        artifact = {
            "scaler": scaler,
            "pca": pca,
            "model": model,
            "model_name": model_name,
            "pca_dim": pca_comp,
            "fold_idx": fold_idx,
        }
        joblib.dump(artifact, model_dir / f"fold_{fold_idx:03d}.joblib")

        fold_records.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "best_iter": bi,
                **fold_metric,
            }
        )

        if log:
            log.info(
                f"[{model_name}] fold {fold_idx}/{total_folds} "
                f"R={fold_metric['pearson_r']:.4f} RMSE={fold_metric['rmse']:.4f} "
                f"R2={fold_metric['r2']:.4f} best_iter={bi}"
            )

    valid_mask = oof_cnt > 0
    oof_pred = np.zeros(n, dtype=np.float64)
    oof_pred[valid_mask] = oof_sum[valid_mask] / oof_cnt[valid_mask]
    overall = compute_metrics(y[valid_mask], oof_pred[valid_mask])

    scaler_full = StandardScaler()
    X_s = scaler_full.fit_transform(X)
    pca_comp_full = min(pca_dim, X_s.shape[1], X_s.shape[0] - 1)
    pca_comp_full = max(2, pca_comp_full)
    pca_full = PCA(n_components=pca_comp_full, svd_solver="full", random_state=seed)
    X_p = pca_full.fit_transform(X_s)

    full_model = build_model_with_overrides(model_name, seed, model_overrides=model_overrides)
    if best_iters:
        bi = int(np.median(best_iters))
        if model_name == "xgb":
            full_model.set_params(n_estimators=max(200, bi))
        if model_name == "lgbm":
            full_model.set_params(n_estimators=max(300, bi))
    full_model.fit(X_p, y)

    full_artifact = {
        "scaler": scaler_full,
        "pca": pca_full,
        "model": full_model,
        "model_name": model_name,
        "train_size": int(len(y)),
        "pca_dim": pca_comp_full,
        "median_best_iter": int(np.median(best_iters)) if best_iters else None,
    }
    full_path = Path(outdir) / model_name / "full_model.joblib"
    joblib.dump(full_artifact, full_path)

    oof_path = Path(outdir) / model_name / "oof_predictions.csv"
    with open(oof_path, "w", encoding="utf-8") as f:
        f.write("idx,y_true,y_pred,pred_count\n")
        for i in range(n):
            f.write(f"{i},{float(y[i])},{float(oof_pred[i])},{int(oof_cnt[i])}\n")

    result = {
        "model_name": model_name,
        "overall": overall,
        "folds": fold_records,
        "median_best_iter": int(np.median(best_iters)) if best_iters else None,
        "full_model_path": str(full_path),
        "oof_path": str(oof_path),
    }
    if return_oof:
        return result, oof_pred
    return result


def setup_logger(outdir):
    log = logging.getLogger("kcat_train")
    log.setLevel(logging.INFO)
    log.handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    log.addHandler(ch)

    fh = logging.FileHandler(Path(outdir) / "train.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    log.addHandler(fh)
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .pt")
    parser.add_argument("--outdir", type=str, default="./outputs/kcat")
    parser.add_argument("--model", type=str, default="all", choices=["all", "xgb", "lgbm", "rf", "blend"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--no-repeat", action="store_true")
    parser.add_argument("--pca-dim", type=int, default=64)

    parser.add_argument("--target-key", type=str, default="y")
    parser.add_argument("--graph-key", type=str, default="graphs")
    parser.add_argument("--group-key", type=str, default="groups")
    parser.add_argument("--disable-groups", action="store_true")

    parser.add_argument("--use-mps-feature", action="store_true")
    parser.add_argument("--feature-cache", type=str, default="")
    parser.add_argument("--lgbm-config", type=str, default="", help="Path to JSON overrides for LGBM")
    parser.add_argument("--xgb-config", type=str, default="", help="Path to JSON overrides for XGBoost")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    log = setup_logger(args.outdir)
    set_seed(args.seed)

    log.info("Loading dataset...")
    graphs, y, groups = load_dataset(
        args.dataset, target_key=args.target_key, graph_key=args.graph_key, group_key=args.group_key
    )
    if args.disable_groups:
        groups = None

    X = None
    if args.feature_cache and os.path.exists(args.feature_cache):
        cache = np.load(args.feature_cache, allow_pickle=True)
        X = cache["X"]
        y_cache = cache["y"]
        if len(y_cache) == len(y):
            y = y_cache.astype(np.float32)
        groups_cache = cache["groups"] if "groups" in cache.files else None
        if groups_cache is not None and len(groups_cache) == len(y):
            groups = groups_cache
        log.info(f"Loaded feature cache: {args.feature_cache}")

    if X is None:
        log.info("Building graph-level features...")
        X = build_feature_matrix(graphs, use_mps_feature=args.use_mps_feature, log=log)
        if args.feature_cache:
            np.savez_compressed(
                args.feature_cache,
                X=X,
                y=y,
                groups=np.array(groups, dtype=object) if groups is not None else np.array([]),
            )
            log.info(f"Saved feature cache to: {args.feature_cache}")

    log.info(f"Feature shape: {X.shape}, target shape: {y.shape}")
    if groups is not None:
        log.info(f"Using GroupKFold with {len(np.unique(groups))} groups.")
    else:
        log.info("Using KFold/RepeatedKFold.")

    lgbm_overrides, lgbm_pca_dim_override = load_model_config(args.lgbm_config, "lgbm")
    if args.lgbm_config:
        log.info(
            f"Loaded LGBM overrides from {args.lgbm_config}. "
            f"pca_dim_override={lgbm_pca_dim_override}"
        )

    xgb_overrides, xgb_pca_dim_override = load_model_config(args.xgb_config, "xgb")
    if args.xgb_config:
        log.info(
            f"Loaded XGB overrides from {args.xgb_config}. "
            f"pca_dim_override={xgb_pca_dim_override}"
        )

    if args.model == "all":
        model_list = ["xgb", "lgbm", "rf"]
    elif args.model == "blend":
        model_list = ["xgb", "lgbm"]
    else:
        model_list = [args.model]

    all_results = []
    oof_by_model = {}
    for m in model_list:
        log.info(f"Start CV for model: {m}")
        model_overrides = None
        model_pca_dim = args.pca_dim
        if m == "lgbm" and lgbm_overrides is not None:
            model_overrides = lgbm_overrides
            if lgbm_pca_dim_override is not None:
                model_pca_dim = lgbm_pca_dim_override
        if m == "xgb" and xgb_overrides is not None:
            model_overrides = xgb_overrides
            if xgb_pca_dim_override is not None:
                model_pca_dim = xgb_pca_dim_override
        if args.model == "blend":
            res, oof_pred = run_cv_for_model(
                model_name=m,
                X=X,
                y=y,
                outdir=args.outdir,
                seed=args.seed,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                pca_dim=model_pca_dim,
                groups=groups,
                use_repeated=(not args.no_repeat),
                model_overrides=model_overrides,
                return_oof=True,
                log=log,
            )
            oof_by_model[m] = oof_pred
        else:
            res = run_cv_for_model(
                model_name=m,
                X=X,
                y=y,
                outdir=args.outdir,
                seed=args.seed,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                pca_dim=model_pca_dim,
                groups=groups,
                use_repeated=(not args.no_repeat),
                model_overrides=model_overrides,
                log=log,
            )
        all_results.append(res)
        o = res["overall"]
        log.info(
            f"[{m}] OOF Overall: R={o['pearson_r']:.4f}, RMSE={o['rmse']:.4f}, "
            f"MAE={o['mae']:.4f}, R2={o['r2']:.4f}"
        )

    if args.model == "blend":
        blend_best = search_blend_weights(y, oof_by_model["lgbm"], oof_by_model["xgb"], step=0.01)
        blend_dir = Path(args.outdir) / "blend"
        blend_dir.mkdir(parents=True, exist_ok=True)

        blend_oof_path = blend_dir / "oof_predictions.csv"
        with open(blend_oof_path, "w", encoding="utf-8") as f:
            f.write("idx,y_true,y_pred_blend,y_pred_lgbm,y_pred_xgb\n")
            for i in range(len(y)):
                f.write(
                    f"{i},{float(y[i])},{float(blend_best['pred'][i])},"
                    f"{float(oof_by_model['lgbm'][i])},{float(oof_by_model['xgb'][i])}\n"
                )

        component_paths = {r["model_name"]: r["full_model_path"] for r in all_results}
        blend_artifact = {
            "model_name": "blend",
            "weights": {"lgbm": blend_best["w_lgbm"], "xgb": blend_best["w_xgb"]},
            "overall": blend_best["metrics"],
            "components": {
                "lgbm_full_model_path": component_paths.get("lgbm"),
                "xgb_full_model_path": component_paths.get("xgb"),
            },
            "note": "Final prediction = w_lgbm * pred_lgbm + w_xgb * pred_xgb",
        }
        blend_model_path = blend_dir / "blend_model.json"
        with open(blend_model_path, "w", encoding="utf-8") as f:
            json.dump(blend_artifact, f, ensure_ascii=False, indent=2)

        blend_result = {
            "model_name": "blend",
            "overall": blend_best["metrics"],
            "weights": {"lgbm": blend_best["w_lgbm"], "xgb": blend_best["w_xgb"]},
            "full_model_path": str(blend_model_path),
            "oof_path": str(blend_oof_path),
        }
        all_results.append(blend_result)
        o = blend_result["overall"]
        log.info(
            f"[blend] OOF Overall: R={o['pearson_r']:.4f}, RMSE={o['rmse']:.4f}, "
            f"MAE={o['mae']:.4f}, R2={o['r2']:.4f}, "
            f"w_lgbm={blend_best['w_lgbm']:.2f}, w_xgb={blend_best['w_xgb']:.2f}"
        )

    all_results = sorted(all_results, key=lambda r: (-r["overall"]["pearson_r"], r["overall"]["rmse"]))
    best = all_results[0]

    summary = {
        "dataset": args.dataset,
        "n_samples": int(len(y)),
        "feature_dim": int(X.shape[1]),
        "models": all_results,
        "best_model": {
            "name": best["model_name"],
            "overall": best["overall"],
            "full_model_path": best["full_model_path"],
        },
    }
    summary_path = Path(args.outdir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"Best model: {best['model_name']}")
    log.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
