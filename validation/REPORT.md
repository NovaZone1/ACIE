# 运行验证记录

最近验证日期：2026-09-15。当前实验环境为 Python 3.10.12、CPU；安装版本可由 `scripts/check_environment.py` 重现查询。

## 已执行

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -p no:cacheprovider -q`：**46 项通过，0 项失败**。覆盖提前窗口合同、配对控制与完整模型的结构等价性、两阶段训练、12 个神经分支、树模型、日期分组、源域标准化、配对约束与上下文策略、匹配重采样类别计数、检查点恢复和官方窗口适配合同。

真实处理后数据已按冻结 revision 导出：HUI360 1,417 个窗口，SSUP-A 6,098 个窗口。六种核心方法在两个域各运行五个随机种子，共 **60 个完整运行**。`source_controls_validation.json` 已检查每个运行状态、固定划分和预测 ID，结果为 passed。随后在源域验证集完成 **50 次 λ 扫描、50 次匹配半径扫描和 20 次日期上下文策略训练**；`pair_tuning_validation.json` 与 `pair_context_validation.json` 已检查模型、验证指标、记录唯一性以及零测试输出。`pip check` 报告 `No broken requirements found`。

冻结跨域协议随后完成 **2 个方向×4 个方法×5 个种子，共 40 次运行**。`frozen_cross_domain_validation.json` 复核了预注册配置哈希、数据与 split 冻结哈希、40 组模型和预测、源域／目标域池边界、组间零泄漏、预测 ID 与标签一致性以及有限分数，结果为 `passed`。目标比较使用五种子概率集成和 2,000 次目标日期组 bootstrap。

详细方法结果和限制见 `../reports/source_control_pilot_2026-09-13.md`、`../reports/pair_mechanism_tuning_2026-09-13.md`、`../reports/pair_context_audit_2026-09-13.md` 与 `../reports/frozen_cross_domain_evaluation_2026-09-14.md`。

## 边界与后续复现

真机实验与缺少官方预训练权重的 MotionBERT 复现尚未执行。MLP、LSTM 和 ST-GCN 已调用固定上游实现完成 GPU 训练；冻结的本库方法仍应与这些官方架构分栏说明。

冻结数据版本缺少官方 HUI 训练列表中的一个录制和 SSUP-A 测试列表中的两个录制；缺口已写入预注册和 staging provenance。相关方向不能标为完整官方 benchmark。目标测试只有 5 个或 9 个日期组，bootstrap 区间可能不稳定；结果不能作为部署安全或人类内部心理状态的证据。

官方 HUI360-Baselines MLP 随后在服务器完成双向五种子公平复现。双方使用相同种子与逐窗口相同的数据，选模仅使用源域验证 AP；`official_fair_mlp_validation.json` 对 10 个模型和 26,410 条目标预测执行 93 项检查，结果为 `passed`。五种子集成 AP/AUROC 分别为 0.0932/0.7296 和 0.4264/0.7449。

官方 LSTM 也完成双向五种子公平复现。`official_fair_lstm_validation.json` 对 10 个 LSTM 模型和 26,410 条预测执行 97 项检查，结果为 `passed`；五种子集成 AP/AUROC 分别为 0.1340/0.7878 和 0.4124/0.7734。

官方 ST-GCN 已完成双向五种子公平复现。`official_fair_stgcn_validation.json` 对 10 个 ST-GCN 模型和 26,410 条预测执行 99 项检查，结果为 `passed`；五种子集成 AP/AUROC 分别为 0.0416/0.5650 和 0.4609/0.7269。

行为信息对应关系破坏对照随后完成双向五种子共 10 个模型。真实 `dual_no_pair` 相对容量匹配置换行为模型的集成 AP 在两个方向分别提高 0.0185 和 0.0489，日期组 bootstrap 下界均高于 0；置换模型与纯几何模型无可区分差异。`behavior_information_control_validation.json` 对 26,410 条预测和 12,797 条映射执行 65 项检查，结果为 `passed`。该实验明确标为主结果后的机制验证。留一目标日期分析中，真实行为相对置换行为的整体 AP 差值在前向为 +0.0149～+0.0233、反向为 +0.0458～+0.0580，所有留一结果均为正；逐日期差值仍有负值或零值。


E4提前量与误报实验已复用冻结的 `geometry`／`dual_no_pair` 五种子检查点，在两个方向的1.07、1.60、2.13、2.67、3.20秒名义截断上完成100个推理文件。1.07秒重新导出的两个域特征与原冻结数据逐元素相同；旧、新概率最大差异为6.52e-09。`horizon_evaluation_validation.json` 检查协议、数据包与检查点哈希、标签不变、阈值冻结、预测覆盖、共同轨迹集合和基准复现，共492/492项通过。结果只稳健支持约1.07秒处的双向行为增量排序证据；更长提前量区间跨0，源域阈值迁移后的目标召回很低。


E3 正式表格随后从冻结跨域目录的 40 份 `test_predictions.metrics.json` 生成，没有重新训练或重新评分。`e3_pair_diagnostic_validation.json` 逐字段复算四种方法、两个方向、五个种子的原始值、均值、标准差及输入哈希，40/40 份输入通过。

收敛范围补充对照完成 **2 个方向×3 个方法×5 个种子，共 30 次运行**。`random_pair` 与 `full_selected` 使用相同 41,730 参数的冻结几何加行为残差结构，仅交换配对负例身份；实际更换比例为 0.9725～1.0000；10/10 个模型的正例顺序、配对权重、负例多重集合及冻结几何状态与 `full_selected` 精确一致。HistGradientBoosting 与 Random Forest 只按源域验证 AP 选择复杂度。`revised_scope_controls_validation.json` 对 30 组运行的冻结协议、split、预测 ID、标签、逐运行指标、随机配对审计和树模型源域选模执行复核，结果为 `passed`。随机配对未产生双向一致的配对身份收益；HistGB 集成 AP 在前向/反向分别为 0.1318/0.4712，显示明显方向依赖。
