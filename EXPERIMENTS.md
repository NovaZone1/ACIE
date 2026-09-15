# 实验施工

## 原生参考与四方向主表

先在独立环境运行已冻结的官方基线，核实其样本口径。原作者完整命令由 `scripts/run_official_baseline.py --help` 列出。当前脚本调用其 `training.py`，不复制其模型实现。官方 MLP、LSTM 和 ST-GCN 已按冻结公平协议完成双向五种子训练；结果分别由 `scripts/validate_official_fair_mlp.py`、`scripts/validate_official_fair_lstm.py` 和 `scripts/validate_official_fair_stgcn.py` 复核。

本库四方向实验：

```bash
python scripts/run_matrix.py --data data/all.npz --domains HUI360 SSUP-A \
  --config configs/full.yaml --out outputs/matrix/full --seeds 11 22 33 44 55
```

两种域内与两种跨域结果分别报告，协议列不得删除。域内默认本地分组；使用固定官方 test 池时先生成独立 split 并逐方向运行 `acie train/predict`。

## 已冻结的双向跨域对照

以下命令对应 2026-09-14 已完成的 40 次评估。运行器在首次目标评分前锁定数据、配置、种子和两份划分；已有运行可在锁内容完全一致时断点续跑。

```bash
PYTHONPATH=src python scripts/run_frozen_cross_domain.py \
  --data data/official_cross_domain.npz --out outputs/frozen_cross_domain \
  --domains HUI360 SSUP-A \
  --method geometry=configs/geometry.yaml \
  --method dual_no_pair=configs/dual_no_pair.yaml \
  --method weighted_no_pair=configs/dual_no_pair_weighted_selected.yaml \
  --method full_selected=configs/full_source_selected.yaml \
  --seeds 11 22 33 44 55 --split-seed 42 --val-fraction 0.2 \
  --bootstrap 2000 --confirm-frozen
PYTHONPATH=src python scripts/validate_frozen_cross_domain.py \
  --data data/official_cross_domain.npz --runs outputs/frozen_cross_domain \
  --preregistered protocols/cross_domain_preregistered_2026-09-13.json \
  --out validation/frozen_cross_domain_validation.json
```

结果见 `reports/frozen_cross_domain_evaluation_2026-09-14.md`。配对贡献未通过预注册的双向为正判据；不得在查看目标结果后修改该判据或用新超参数覆盖这组冻结结果。

## 行为信息对应关系破坏对照

这组 2026-09-15 机制验证在每个划分和日期组内置换完整行为张量，保留模型容量和几何输入。它是在主结果之后设计的，不属于原始预注册。

```bash
PYTHONPATH=src python scripts/prepare_behavior_information_control.py \
  --data data/official_cross_domain.npz --frozen-root outputs/frozen_cross_domain \
  --config configs/dual_no_pair.yaml \
  --mappings protocols/behavior_information_control_mappings \
  --out protocols/behavior_information_control_2026-09-15.json
PYTHONPATH=src python scripts/run_behavior_information_control.py \
  --root . --data data/official_cross_domain.npz \
  --frozen-root outputs/frozen_cross_domain \
  --protocol protocols/behavior_information_control_2026-09-15.json \
  --out outputs/behavior_information_control --seeds 11 22 33 44 55 --bootstrap 2000
PYTHONPATH=src python scripts/validate_behavior_information_control.py \
  --root . --data data/official_cross_domain.npz \
  --frozen-root outputs/frozen_cross_domain \
  --protocol protocols/behavior_information_control_2026-09-15.json \
  --runs outputs/behavior_information_control \
  --out validation/behavior_information_control_validation.json
PYTHONPATH=src python scripts/analyze_behavior_gain_by_group.py \
  --root . --out outputs/behavior_information_control/group_analysis.json
```

结果见 `reports/behavior_information_control_2026-09-15.md`。

## E3 冻结配对诊断正式汇总

E3 直接读取 40 份冻结跨域测试指标，不重新训练。标准化器、匹配半径和支持规则来自源域；目标标签只在预测完成后构造离线异标签诊断对。

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_e3_pair_diagnostic.py \
  --runs outputs/frozen_cross_domain \
  --out outputs/e3_pair_diagnostic/summary.json \
  --report reports/e3_pair_diagnostic_2026-09-15.md
PYTHONPATH=src .venv/bin/python scripts/validate_e3_pair_diagnostic.py \
  --runs outputs/frozen_cross_domain \
  --summary outputs/e3_pair_diagnostic/summary.json \
  --out validation/e3_pair_diagnostic_validation.json
```

这里的排序准确率不是分类准确率，也不参与训练、选模或阈值确定。结果用于说明局部配对诊断的支持范围，不能恢复已经失败的配对损失主张。

## 收敛范围补充对照

在最终主张收敛后，补充一个与 `full_selected` 相同冻结双分支结构、只交换负例身份的 `random_pair`，以及 HistGradientBoosting、Random Forest 两种同信息树模型。补充协议在新一轮目标评分前冻结，但属于查看主结果后的 post-hoc 稳健性实验。

```bash
PYTHONPATH=src .venv/bin/python scripts/run_revised_scope_controls.py \
  --data data/official_cross_domain.npz \
  --parent outputs/frozen_cross_domain \
  --protocol protocols/revised_scope_controls_2026-09-15.json \
  --random-config configs/random_pair_source_selected.yaml \
  --full-config configs/full_source_selected.yaml \
  --out outputs/revised_scope_controls --bootstrap 2000 --confirm-frozen
PYTHONPATH=src .venv/bin/python scripts/validate_revised_scope_controls.py \
  --data data/official_cross_domain.npz \
  --parent outputs/frozen_cross_domain \
  --runs outputs/revised_scope_controls \
  --out validation/revised_scope_controls_validation.json
PYTHONPATH=src .venv/bin/python scripts/report_revised_scope_controls.py \
  --summary outputs/revised_scope_controls/summary.json \
  --runs outputs/revised_scope_controls \
  --out reports/revised_scope_controls_2026-09-15.md
```

结果见 `reports/revised_scope_controls_2026-09-15.md`。`hard_negative`、全方法 E4、完整域内矩阵和 MotionBERT 不属于当前收敛主张的必要实验，不在本轮运行。

## 六组源域筛选对照

六组方法共用同一日期级 train/val/test 划分：几何、普通融合、匹配重采样、融合配对、冻结双路无配对、完整冻结双路配对。该命令用于检查方法机制，不替代官方测试池或跨域 benchmark。

```bash
python scripts/run_controls.py --data data/source_development.npz \
  --domains HUI360 SSUP-A --out outputs/source_controls \
  --seeds 11 22 33 44 55 --split-seed 42
python scripts/summarize_controls.py --runs outputs/source_controls \
  --out outputs/source_controls/comparison.json --bootstrap 2000
```

## 同信息强对手

```bash
python -m acie tree --data data/all.npz --split splits/hui_to_ssup.json --kind histgb --out outputs/hui_to_ssup/tree
python -m acie tree --data data/all.npz --split splits/hui_to_ssup.json --kind random_forest --out outputs/hui_to_ssup/forest
```

其余对照以 `configs/` 下相同名称 YAML 运行 train/predict。对当前收敛主张，`geometry`、`dual_no_pair`、行为对应关系置乱和官方参考模型构成核心比较；`random_pair` 与树模型作为 post-hoc 稳健性补充。`hard_negative` 只服务于已放弃的配对机制主张，不再列为必要实验。

## 源域调参与配对诊断

`python scripts/tune_source.py --help` 给出 λ 与 hidden 的源域开发搜索入口。该程序不运行目标测试。锁定最优配置后再统一执行目标推断，保留全部试验配置。

当前配对机制筛选分两步执行，并严格只读取 `val`：先在统一类别权重下扫描配对损失，再固定联合选择的 λ 扫描匹配半径。输出中的 `test_evaluated` 必须保持为 `false`。

```bash
PYTHONPATH=src python scripts/tune_pair_lambda.py \
  --data data/source_development.npz --splits-root outputs/source_controls \
  --out outputs/pair_lambda_tuning --domains HUI360 SSUP-A \
  --lambdas 0 0.01 0.03 0.1 0.3 --seeds 11 22 33 44 55 --bootstrap 300
PYTHONPATH=src python scripts/tune_matching_radius.py \
  --data data/source_development.npz --splits-root outputs/source_controls \
  --lambda-root outputs/pair_lambda_tuning --out outputs/matching_radius_tuning \
  --domains HUI360 SSUP-A --quantiles 0.25 0.5 0.75 0.9 1.0 \
  --pair-lambda 0.03 --seeds 11 22 33 44 55 --bootstrap 300
```

本轮联合选择为 `pair_lambda=0.03`、`radius_quantile=0.75`，已冻结为 `configs/full_source_selected.yaml`。这是源域开发选择，不应写入测试成绩；详细结果和是否保留配对贡献的判断见 `reports/pair_mechanism_tuning_2026-09-13.md`。

日期与录制上下文审计以及固定半径的上下文策略对照：

```bash
PYTHONPATH=src python scripts/audit_pair_context.py \
  --data data/source_development.npz --splits-root outputs/source_controls \
  --pairs-root outputs/matching_radius_tuning --domains HUI360 SSUP-A \
  --quantile 0.75 --seeds 11 22 33 44 55 --out outputs/pair_context_audit.json
PYTHONPATH=src python scripts/compare_pair_context_policy.py \
  --data data/source_development.npz --splits-root outputs/source_controls \
  --baseline-root outputs/matching_radius_tuning --out outputs/pair_context_policy \
  --config configs/full_source_selected.yaml --domains HUI360 SSUP-A \
  --policies same_group different_group --quantile 0.75 \
  --seeds 11 22 33 44 55 --bootstrap 300
```

上下文策略实验复用原始 `any` 策略的半径，使比较只改变日期约束。结果见 `reports/pair_context_audit_2026-09-13.md`。

配对结果在训练目录的 `matching.json`／`pairs.train.jsonl` 中。预测同时报告相同冻结半径下的几何、行为及最终分数配对正确率。诊断覆盖低或同特征简单模型已达到相同收益时，缩小条件增量主张。

## 提前量与统计

E4 使用冻结官方窗口和已有事件时间，不补标心理起始时刻。32帧窗口从官方16帧截断点整体向前移动0、8、16、24、32帧；负例保持最大人体尺度代理锚点并采用同样位移。窗口重新执行连续性、有效性、人体尺度与关键点过滤。协议在任何提前量模型评分前冻结于 `protocols/horizon_evaluation_2026-09-15.json`。

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_horizon_bundles.py \
  --root . --raw-root data/hui_light --out data/horizon_evaluation \
  --protocol protocols/horizon_evaluation_2026-09-15.json
PYTHONPATH=src .venv/bin/python scripts/run_horizon_evaluation.py \
  --root . --protocol protocols/horizon_evaluation_2026-09-15.json \
  --out outputs/horizon_evaluation --device cpu --bootstrap 2000
PYTHONPATH=src .venv/bin/python scripts/validate_horizon_evaluation.py \
  --root . --protocol protocols/horizon_evaluation_2026-09-15.json \
  --outputs outputs/horizon_evaluation \
  --out validation/horizon_evaluation_validation.json
```

当前 E4 复用 `geometry` 与 `dual_no_pair` 的五种子冻结检查点，不重新训练。主要分析使用五点共同可用轨迹；完整可用集作为样本构成敏感性结果。每个种子沿用自身源域开发阈值计算 FPR／Recall，五种子概率集成只用于 AP／AUROC。自动验证为492/492项通过，结果见 `reports/horizon_evaluation_2026-09-15.md`。

`aggregate_horizons.py` 保留为通用的单模型已有预测汇总入口。`plot_horizon_results.py` 只读取实测冻结汇总，输出 PDF 与300 DPI PNG，不补造样本、误差条或平滑趋势。

## 停止判据

首先核实匹配支持和原生简单模型。缺少可比正负例时，不应靠挑选可视化例子继续主张；随机同计数配对、重采样或融合 MLP 已解释全部提升时，应删除对应方法贡献。讨论只限公开数据行为预测，不外推真实机器人安全和人的内部心理状态。
