# 官方 LSTM 五种子公平跨域对照（2026-09-14）

## 结论

官方 HUI360-Baselines LSTM 已在服务器完成两个方向、五个随机种子的公平复现。五种子集成 AP/AUROC 分别为：HUI360→SSUP-A **0.1340/0.7878**，SSUP-A→HUI360 **0.4124/0.7734**。

LSTM 在前向迁移中明显优于官方 MLP：AP 提高 0.0408，目标日期组 bootstrap 95% 区间为 [+0.0115,+0.0676]。反向迁移中 LSTM 比 MLP 低 0.0140 AP，区间为 [−0.0785,+0.0543]，当前数据无法区分两者；LSTM 的反向 AUROC 则从 MLP 的 0.7449 提高到 0.7734。

与 ACIE 相比，LSTM 前向 AP 点估计低于四个 ACIE 变体，但四个差值区间均跨过 0；反向中 LSTM 明确高于纯几何模型，与三个行为模型的差值区间均跨过 0。LSTM 缩小了官方模型的前向迁移缺口，但没有取代 ACIE `dual_no_pair` 的前向最佳结果，也没有改变“模型排序依赖迁移方向”的结论。

## 公平协议

- 上游代码 commit、数据 revision、窗口、标签、五个种子和概率集成方法与官方 MLP/ACIE 公平对照完全相同。
- 官方 LSTM 使用隐藏维度 128、3 层、dropout 0.0；HUI360→SSUP-A 使用官方规定的 75 epoch，SSUP-A→HUI360 使用 30 epoch。
- 检查点只按源域验证 AP 选择。目标标签没有参与选模、超参数选择或阈值调整。
- 官方 MLP 与 LSTM 配置经自动核对，除模型类型、epoch 和实验名外，其余官方参数一致；公平配置沿用已经逐窗口验证的数据范围。

## 同口径总表

| 方向 | 方法 | 单种子 AP | 五种子集成 AP | 五种子集成 AUROC |
|---|---|---:|---:|---:|
| HUI360→SSUP-A | 官方 MLP | 0.0882±0.0156 | 0.0932 | 0.7296 |
|  | 官方 LSTM | 0.1221±0.0155 | 0.1340 | 0.7878 |
|  | ACIE geometry | 0.1480±0.0119 | 0.1514 | 0.8682 |
|  | ACIE dual_no_pair | 0.1577±0.0194 | **0.1689** | **0.8900** |
|  | ACIE weighted_no_pair | 0.1412±0.0296 | 0.1345 | 0.8532 |
|  | ACIE full_selected | 0.1496±0.0190 | 0.1538 | 0.8730 |
| SSUP-A→HUI360 | 官方 MLP | 0.4179±0.0194 | **0.4264** | 0.7449 |
|  | 官方 LSTM | 0.4098±0.0358 | 0.4124 | 0.7734 |
|  | ACIE geometry | 0.2825±0.0158 | 0.2847 | 0.7400 |
|  | ACIE dual_no_pair | 0.3265±0.0336 | 0.3311 | 0.7703 |
|  | ACIE weighted_no_pair | 0.3492±0.0444 | 0.3721 | **0.7934** |
|  | ACIE full_selected | 0.3538±0.0535 | 0.3556 | 0.7877 |

## 日期组 bootstrap 对照

差值按“官方 LSTM 五种子集成 AP − 对照模型五种子集成 AP”计算，使用 2,000 次目标日期组重采样。

| 方向 | 对照 | AP 差值 | 95% 区间 |
|---|---|---:|---:|
| HUI360→SSUP-A | 官方 MLP | +0.0408 | [+0.0115,+0.0676] |
|  | ACIE geometry | −0.0174 | [−0.0761,+0.0291] |
|  | ACIE dual_no_pair | −0.0349 | [−0.0790,+0.0030] |
|  | ACIE weighted_no_pair | −0.0004 | [−0.0314,+0.0277] |
|  | ACIE full_selected | −0.0198 | [−0.0563,+0.0151] |
| SSUP-A→HUI360 | 官方 MLP | −0.0140 | [−0.0785,+0.0543] |
|  | ACIE geometry | +0.1277 | [+0.0379,+0.2319] |
|  | ACIE dual_no_pair | +0.0812 | [−0.0044,+0.1884] |
|  | ACIE weighted_no_pair | +0.0402 | [−0.0642,+0.1601] |
|  | ACIE full_selected | +0.0568 | [−0.0571,+0.1729] |

目标测试只有 5 个或 9 个日期组，百分位区间可能不稳定。

## 单种子结果

| 方向 | seed 11 | seed 22 | seed 33 | seed 44 | seed 55 |
|---|---:|---:|---:|---:|---:|
| HUI360→SSUP-A AP | 0.1065 | 0.1460 | 0.1240 | 0.1103 | 0.1239 |
| SSUP-A→HUI360 AP | 0.4177 | 0.3731 | 0.4546 | 0.4297 | 0.3737 |

## 审计产物

- 机器摘要：`outputs/official_baselines/fair_v1/five_seed_lstm_summary.json`
- 自动验证：`validation/official_fair_lstm_validation.json`
- 验证入口：`scripts/validate_official_fair_lstm.py`
- 逐种子结果：`outputs/official_baselines/fair_v1/<方向>/lstm/seed<种子>/`
- 公平配置：`official_baselines/protocol_v1/<方向>/fair_*_lstm.yaml`

自动验证执行 97 项检查，全部通过；共核验 10 个 LSTM 模型和 26,410 条目标预测。冻结数据缺少官方列表中的三个录制，因此结果仍标为“可用官方配置子集”，不能称为完整官方 benchmark。
