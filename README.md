# Enzyme AI for Science (Ligase + kcat)

连接酶多任务分类与催化活性回归项目，面向 Apple Silicon（MPS）环境，提供训练脚本、评估脚本和 Streamlit Web 界面。

## 项目简介

本项目聚焦连接酶相关任务，支持：

- 分类任务：连接酶鉴定、ATP/NAD 偏好、水溶性、EC 子类、底物谱、金属离子依赖
- 回归任务：`log_kcat` / `kcat`
- 输入形式：蛋白序列、FASTA 批量、PDB 结构
- 部署方式：本地 Web 应用（Streamlit）

项目设计目标是：在小样本生物数据场景中，提供可复现、可展示、可快速迭代的工程流程。

## 主要特性

- Apple Silicon 友好：默认支持 `torch.mps`
- 多任务模型：ESM-2 backbone + 轻量分类头
- kcat 集成回归：`LightGBM + XGBoost` 加权融合（blend）
- 全流程脚本化：数据整理、训练、评估、可视化、Web 推理

## 环境配置

### 依赖环境

- Python >= 3.9
- PyTorch
- transformers
- scikit-learn
- lightgbm
- xgboost
- pandas / numpy / matplotlib
- streamlit

### 安装方式

```bash
# 1) 进入项目目录
cd /Users/xu/Desktop/course\ project

# 2) 安装依赖（或使用你现有的 protein_dl 环境）
pip install -r requirements.txt
```

## 数据来源与说明

项目数据主要来自：

- UniProt (reviewed, EC=6.*) 自动抓取与解析
- 本地补充数据（备份恢复的 fasta/csv/pdb）

当前关键数据文件位于：

- `data/raw/`：原始 fasta/csv（训练输入）
- `data/processed/`：训练就绪 csv 与报告
- `data/interim/feat_cache.npz`：kcat 特征缓存
- `data/structures/pdb_files/`：PDB 结构文件

## 项目结构

```text
course project/
├── app/
│   └── app.py                           # Streamlit 主应用
├── scripts/
│   ├── train_ligase_multitask.py        # 连接酶多任务训练
│   ├── train_kcat_baseline.py           # kcat基线/融合训练
│   ├── predict_kcat_from_sequence.py    # kcat推理（序列/FASTA/PDB）
│   ├── predict_ligase_multitask.py      # 多任务分类推理
│   ├── evaluate_full_task_suite.py      # 全任务评估
│   └── ...
├── src/
│   └── ligase_multitask.py              # 多任务模型与工具函数
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── structures/pdb_files/
├── models/
│   └── checkpoints/                     # *.pth 模型权重
├── outputs/
│   ├── ligase_multitask_v3_1/
│   └── kcat_blend/
├── docs/
│   └── reports/
├── app.py                               # 兼容入口（调用 app/app.py）
└── README.md
```

## 快速开始

### 1) 启动 Web 应用

```bash
cd /Users/xu/Desktop/course\ project
streamlit run app.py
```

### 2) 重建连接酶训练数据

```bash
python scripts/build_ligase_multitask_dataset.py \
  --fetch-uniprot \
  --uniprot-method search \
  --uniprot-max-rows 3000 \
  --uniprot-page-size 500 \
  --uniprot-query "(reviewed:true) AND (ec:6.*)" \
  --out-csv data/processed/ligase_multitask_auto.csv \
  --report-json data/processed/ligase_multitask_auto_report.json

python scripts/prepare_ligase_multitask_trainset.py \
  --in-csv data/processed/ligase_multitask_auto.csv \
  --out-csv data/processed/ligase_multitask_train_ready.csv \
  --report-json data/processed/ligase_multitask_train_ready_report.json
```

### 3) 训练连接酶多任务模型

```bash
python scripts/train_ligase_multitask.py \
  --csv data/processed/ligase_multitask_train_ready.csv \
  --outdir outputs/ligase_multitask_v3_1
```

### 4) 训练 kcat 融合模型（blend）

```bash
python scripts/train_kcat_baseline.py \
  --dataset data/processed/dataset.pt \
  --feature-cache data/interim/feat_cache.npz \
  --outdir outputs/kcat_blend \
  --model blend
```

### 5) kcat 单序列推理（CLI）

```bash
python scripts/predict_kcat_from_sequence.py \
  --sequence "YOUR_PROTEIN_SEQUENCE" \
  --model-path outputs/kcat_blend/blend/blend_model.json \
  --feature-cache data/interim/feat_cache.npz
```

## 关键模型与产物路径

### 分类模型权重（已恢复）

- `models/checkpoints/best_ligase_model.pth`
- `models/checkpoints/best_solubility_model.pth`
- `models/checkpoints/best_cofactor_model.pth`

### kcat 模型权重（已恢复）

- `models/checkpoints/best_kcat_model.pth`
- `models/checkpoints/best_kcat_model_35M.pth`

### 新训练产物

- `outputs/ligase_multitask_v3_1/best_ligase_multitask.pt`
- `outputs/ligase_multitask_v3_1/train_history.csv`
- `outputs/ligase_multitask_v3_1/label_schema.json`
- `outputs/kcat_blend/blend/blend_model.json`
- `outputs/kcat_blend/summary.json`

## 评估与可视化

```bash
python scripts/evaluate_full_task_suite.py --outdir outputs/full_task_eval_v1
python scripts/make_classroom_visualization.py --eval-dir outputs/full_task_eval_v1 --lang zh
```

## 常见问题

### 1) Web 侧边栏 Model Path / Feature Cache 为空

如果默认路径对应文件不存在会出现此问题。当前标准路径是：

- `outputs/kcat_blend/blend/blend_model.json`
- `data/interim/feat_cache.npz`

### 2) 报错：`No such file ... outputs/kcat_blend/blend/outputs/...`

这是 blend 组件路径重复拼接问题。当前代码已修复路径解析逻辑（`scripts/predict_kcat_from_sequence.py`）。

### 3) MPS 不可用

可在命令里显式指定 CPU：

```bash
--device cpu
```

## 免责声明

本项目用于科研与教学演示，不作为临床决策依据。模型输出需结合实验验证。

## Author

Developed by Eric Xu｜医药人工智能
