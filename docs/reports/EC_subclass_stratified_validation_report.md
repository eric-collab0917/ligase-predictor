# EC 子类分层验证报告（独立版）

## 1. 评估设置
- 数据来源目录：`/Users/xu/Desktop/course project/outputs/ligase_eval_classroom_v1`
- 评估切分：`valid`
- 样本数：`556`
- 决策阈值：substrate=`0.50`，metal_type=`0.50`，metal_presence=`0.50`

## 2. 总体指标
- EC Accuracy：`0.8249`
- EC Macro-F1：`0.4477`
- EC Weighted-F1：`0.8078`
- Substrate Micro-F1：`0.5814`
- Metal Micro-F1：`0.6829`
- Metal Presence Accuracy：`nan`

## 3. EC 子类表现（Top / Bottom）
Top-3（按 F1-score）：
- `6.3`: F1=`0.890`, Precision=`0.869`, Recall=`0.911`, Support=`372`
- `6.2`: F1=`0.806`, Precision=`0.881`, Recall=`0.743`, Support=`70`
- `6.1`: F1=`0.705`, Precision=`0.620`, Recall=`0.816`, Support=`76`
Bottom-3（按 F1-score）：
- `6.4`: F1=`0.000`, Precision=`0.000`, Recall=`0.000`, Support=`11`
- `6.6`: F1=`0.000`, Precision=`0.000`, Recall=`0.000`, Support=`2`
- `6.5`: F1=`0.286`, Precision=`0.800`, Recall=`0.174`, Support=`23`

## 4. 分层诊断（按样本量优先）
| EC 子类 | 样本数 | EC F1 | 底物谱 F1 | 金属依赖 F1 | 主要混淆到 | 混淆数 |
|---|---:|---:|---:|---:|---|---:|
| 6.3 | 372 | 0.890 | 0.637 | 0.716 | 6.1 | 26 |
| 6.1 | 76 | 0.705 | 0.342 | 0.774 | 6.3 | 13 |
| 6.2 | 70 | 0.806 | 0.623 | 0.500 | 6.3 | 18 |
| 6.5 | 23 | 0.286 | 0.579 | 0.000 | 6.1 | 10 |
| 6.4 | 11 | nan | 0.452 | 0.600 | 6.3 | 10 |
| 6.6 | 2 | nan | nan | 0.400 | 6.1 | 1 |

## 5. 课堂展示图（直接插图）
- EC 混淆矩阵（行归一化）: `/Users/xu/Desktop/course project/outputs/ligase_eval_classroom_v1/fig_ec_confusion_row_norm.png`
- EC 混淆矩阵（计数）: `/Users/xu/Desktop/course project/outputs/ligase_eval_classroom_v1/fig_ec_confusion_counts.png`
- EC 各子类 F1: `/Users/xu/Desktop/course project/outputs/ligase_eval_classroom_v1/fig_ec_per_class_f1.png`

## 6. 结论与下一步
- 当前模型在头部 EC 子类上较稳健，但长尾子类（样本数少）仍明显欠拟合。
- 课堂讲解建议：先讲行归一化混淆矩阵，再用分层表说明“数据量 vs 子类性能”的关系。
- 改进优先级：补齐长尾 EC 子类数据 > 类别重加权/焦点损失 > 子类条件化阈值。

## 附录
- 原始展示笔记：`/Users/xu/Desktop/course project/outputs/ligase_eval_classroom_v1/presentation_notes.md`
