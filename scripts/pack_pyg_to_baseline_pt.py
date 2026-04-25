#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pack_pyg_to_baseline_pt.py

输出格式：
torch.save({
    "graphs": list_of_Data,   # 每个有 .x，最好有 .edge_index
    "y": np.ndarray(float32), # shape [N]
    "groups": np.ndarray      # 可选，shape [N]
}, output_path)
"""

import argparse
import numpy as np
import torch


def load_pt_compat(path):
    """
    PyTorch >=2.6 defaults torch.load(weights_only=True), which cannot deserialize
    PyG Data objects. If trusted source, retry with weights_only=False.
    """
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


def scalarize(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        v = v.detach().cpu().view(-1)
        if v.numel() == 0:
            return None
        return float(v[0].item())
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v).reshape(-1)
        if arr.size == 0:
            return None
        return float(arr[0])
    return float(v)


def get_attr(obj, key, default=None):
    if key is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_data_list(obj, data_key=None):
    # 1) dict + 指定 data_key
    if isinstance(obj, dict) and data_key and data_key in obj:
        x = obj[data_key]
        if isinstance(x, list):
            return x
        if hasattr(x, "__len__") and hasattr(x, "get"):
            return [x.get(i) for i in range(len(x))]

    # 2) 直接是 list[Data]
    if isinstance(obj, list):
        return obj

    # 3) PyG Dataset 风格
    if hasattr(obj, "__len__") and hasattr(obj, "get"):
        return [obj.get(i) for i in range(len(obj))]

    raise ValueError("无法识别输入数据结构，请检查 --data-key 或输入文件内容。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 .pt 文件")
    ap.add_argument("--output", required=True, help="输出 .pt 文件")
    ap.add_argument("--data-key", default=None, help="如果输入是 dict，图列表所在 key（如 data_list/graphs）")
    ap.add_argument("--target-key", default="y", help="每个样本上的回归标签字段（如 y/log_kcat）")
    ap.add_argument("--global-target-key", default=None, help="如果标签在顶层 dict（如 obj['y']），填这个")
    ap.add_argument("--group-key", default=None, help="每个样本上的分组字段（如 cluster_id/family_id）")
    args = ap.parse_args()

    obj = load_pt_compat(args.input)

    data_list = to_data_list(obj, data_key=args.data_key)

    # 1) 构建 y
    y = []
    if isinstance(obj, dict) and args.global_target_key and args.global_target_key in obj:
        gy = obj[args.global_target_key]
        if torch.is_tensor(gy):
            gy = gy.detach().cpu().numpy()
        gy = np.asarray(gy).reshape(-1)
        if len(gy) != len(data_list):
            raise ValueError(f"global y 长度 {len(gy)} != data_list 长度 {len(data_list)}")
        y = gy.astype(np.float32).tolist()
    else:
        for i, d in enumerate(data_list):
            yi = scalarize(get_attr(d, args.target_key, None))
            if yi is None:
                raise ValueError(f"样本 {i} 缺少目标字段 `{args.target_key}`")
            y.append(yi)

    # 2) groups（可选）
    groups = None
    if args.group_key:
        gvals = []
        for d in data_list:
            gv = get_attr(d, args.group_key, None)
            if torch.is_tensor(gv):
                gv = gv.detach().cpu().view(-1)
                gv = gv[0].item() if gv.numel() > 0 else "NA"
            gvals.append(gv if gv is not None else "NA")
        groups = np.asarray(gvals)

    # 3) 基础检查
    for i, d in enumerate(data_list):
        if get_attr(d, "x", None) is None:
            raise ValueError(f"样本 {i} 缺少 `x` 节点特征")

    out = {
        "graphs": data_list,
        "y": np.asarray(y, dtype=np.float32),
    }
    if groups is not None:
        out["groups"] = groups

    torch.save(out, args.output)

    print(f"Saved to: {args.output}")
    print(f"N = {len(data_list)}")
    print(f"y shape = {out['y'].shape}, y mean = {out['y'].mean():.4f}, y std = {out['y'].std():.4f}")
    if groups is not None:
        print(f"groups unique = {len(np.unique(groups))}")


if __name__ == "__main__":
    main()
