# Enzyme AI for Science (Apple Silicon)

多任务酶性质与催化活性预测项目（课堂展示版 + Web 交互版）。

## Project Layout

```text
.
├─ app/
│  └─ app.py                          # Streamlit Web 主应用
├─ scripts/
│  ├─ train_ligase_multitask.py
│  ├─ predict_ligase_multitask.py
│  ├─ train_kcat_baseline.py
│  ├─ predict_kcat_from_sequence.py
│  ├─ evaluate_full_task_suite.py
│  └─ ...
├─ src/
│  └─ ligase_multitask.py             # 共享模型/工具模块
├─ data/
│  ├─ raw/                            # 原始数据
│  ├─ interim/                        # 中间缓存特征
│  └─ processed/                      # 训练/评估数据
├─ models/
│  ├─ checkpoints/                    # 模型权重
│  └─ artifacts/                      # 其他模型产物
├─ outputs/                           # 评估输出与图表
├─ docs/
│  ├─ reports/
│  └─ slides/
├─ legacy/                            # 归档的历史实验代码
├─ app.py                             # 兼容入口（调用 app/app.py）
└─ README.md
```

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Launch web app

```bash
streamlit run app.py
```

### 3) Typical training/eval commands

```bash
python scripts/train_ligase_multitask.py --help
python scripts/train_kcat_baseline.py --help
python scripts/evaluate_full_task_suite.py --help
```

## Notes

- 当前工程针对 Apple Silicon + PyTorch MPS 做了路径与脚本组织优化。
- 历史目录（`3D model/`, `atp nad/`, `solubility/`, `kcat/`）已归档到 `legacy/experiments/`。
- 新代码优先使用 `app/`, `scripts/`, `src/` 三层结构。

## Author

Developed by Eric Xu｜医药人工智能
