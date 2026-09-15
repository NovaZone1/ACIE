# 接近条件化的交互意图证据学习

ACIE 是与研究方案对应的可训练研究实现。输入为公开处理后骨架及接近几何，输出为交互分数。主方法采用源域几何配对、冻结几何分支和行为增量学习。代码不采集新数据、不访问摄像头、不执行机器人控制。

**验证边界：自动化测试、HUI360／SSUP-A 官方训练池上的日期分组源域对照、冻结的双向跨域测试，以及官方 MLP、LSTM、ST-GCN 五种子公平复现均已执行。数据 revision 缺少三个配置列出的录制，因此涉及这些录制的方向必须称为可用官方配置子集，不能称为完整官方 benchmark。** 软件测试见 `validation/REPORT.md`，跨域总结果见 `reports/frozen_cross_domain_evaluation_2026-09-14.md`，行为信息机制验证见 `reports/behavior_information_control_2026-09-15.md`，E3 正式诊断见 `reports/e3_pair_diagnostic_2026-09-15.md`，随机配对与树模型补充对照见 `reports/revised_scope_controls_2026-09-15.md`，提前量与误报结果见 `reports/horizon_evaluation_2026-09-15.md`。

## 快速运行

在解压后的本项目目录运行；两个项目使用不同的虚拟环境。

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 对应命令：.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m acie demo --out outputs/demo
python -m pytest -q
```

合成演示会生成窗口数据，拟合源域标准化器和几何模型，训练增量分支，保存检查点，再重新加载检查点做目标域预测。`outputs/demo/DEMO_ONLY.json` 明确标记为软件集成测试。已有演示目录不会被覆盖；再次运行请换输出目录。

离线机器已经安装依赖时，可不安装项目：`PYTHONPATH=src python -m acie demo --out outputs/demo`。Windows 可先设置 `$env:PYTHONPATH="src"`。

## 已实现内容

| 层次 | 实现 |
|---|---|
| 数据 | 统一 NPZ＋JSON 清单、HUI360 官方合法窗口导出、SSUP-A 域标识、窗口与版本哈希 |
| 特征 | 360° 接缝处理、32 维几何统计、17 关节×7 通道局部行为、缺失点与速度有效位 |
| 方法 | 源域近邻匹配、不同轨迹约束、复用上限、支持覆盖审计、两阶段冻结几何训练、加权排序损失 |
| 对照 | 几何、行为、融合 MLP、双路无配对、融合配对、随机配对、困难负例、重采样、类别权重、LSTM、简化图模型、两种树模型 |
| 评价 | AP、AUROC、源域阈值下 FPR／Recall、轨迹配对正确率、录制单元 bootstrap、成对差值区间 |
| 工程 | YAML 配置、检查点与随机状态恢复、预测导出、四方向批跑、源域调参、严格提前窗口导出与共同轨迹汇总、绘图脚本、CI 配置 |

简化图模型是本库控制模型，**不是**官方 ST-GCN；不把自行实现的对照标成 MotionBERT 或其他原作者实现。官方参考模型通过外部仓库调用，详见后文。

## 目录

```text
src/acie/                数据、特征、配对、模型、训练、评价和命令行
configs/                 主方法及各个消融的独立 YAML
scripts/                 官方数据适配、下载、协议划分、实验矩阵和汇总
examples/                清单格式与配置说明
tests/                   单元测试、全部训练分支和适配器合同测试
validation/              本次实际测试日志和验证范围
README.md
METHOD.md                方法与数学定义
DATA_PROTOCOL.md         原生接口与派生协议的边界
EXPERIMENTS.md           论文证据对应的实验命令
SOURCES.md               上游论文、代码与授权说明
```

公开代码仓库不包含原始数据、训练输出、模型检查点、上游代码副本、老师原始构想文档或组内服务器资料。论文数值、协议、验证摘要和图表保存在 `reports/`、`protocols/` 与 `validation/`。`CHECKSUMS_SHA256.txt` 校验公开仓库实际包含的文件。完整本地研究工作区另有未提交的实验产物清单，用于校验数据、预测和检查点。

## 真实数据工作流

### 1. 获取处理后数据与官方代码

先审阅原始数据授权。下载会消耗本地磁盘与带宽，本库不会在安装或测试时触发下载。`--ref main`、`--revision main` 只是首次解析入口；下载脚本保存实际 commit，正式实验必须记录并复用审计后的 commit。

```bash
python -m pip install -e ".[data]"
python scripts/fetch_upstream.py --url https://github.com/hucebot/HUI360-Baselines --ref main --out external/HUI360-Baselines
python scripts/download_data.py --out data/hui_processed --revision main --accept-data-terms
```

官方数据加载器还有自己的依赖，按照其冻结版本安装。不要直接升级已完成实验的环境。原始 RGB 视频不属于本库必需输入。

### 2. 导出官方合法观察窗口

`export_hui360.py` 调用官方 `load_hui_dataset`，读取官方窗口提案和标签，保存每个样本具体取到的帧。它不会自行创造交互标签，也不会将未来交互位置作为预测特征。

导出前必须确定三项：所选官方配置、相应数据帧率、该域是否全景。示例使用 shell 环境变量让这些值显式出现；不能把未核实帧率抄成秒制实验口径。

```bash
# 在当前 shell 设置 HUI_CONFIG 为已审计的官方 YAML 路径，HUI_FPS 为该版本的帧率。
# 官方配置入口示例：external/HUI360-Baselines/experiments/configs/in_hui/mlp_base.yaml
python scripts/export_hui360.py --upstream external/HUI360-Baselines \
  --config "$HUI_CONFIG" --data-root data/hui_processed --out data/hui_train \
  --split train --domain HUI360 --fps "$HUI_FPS" --panoramic
python scripts/export_hui360.py --upstream external/HUI360-Baselines \
  --config "$HUI_CONFIG" --data-root data/hui_processed --out data/hui_test \
  --split val --domain HUI360 --fps "$HUI_FPS" --panoramic
```

SSUP-A 按所选官方配置重复两次，改为 `--domain SSUP-A --panoramic`，使用其自身已核实帧率。当前处理后数据的横坐标会跨越图像左右边界，必须按循环坐标归一化。配置目录可用 `find external/HUI360-Baselines/experiments/configs -name '*.yaml'` 查阅。HUI360 官方 loader 某些版本的 offline 文件映射只接受 `main` 或特定 legacy 标识；数据真实 commit 由下载 lock 单独固定，不能混用旧／新 CSV 与配置。

```bash
python scripts/merge_bundles.py --inputs data/hui_train/bundle.npz data/hui_test/bundle.npz \
  data/ssup_train/bundle.npz data/ssup_test/bundle.npz --out data/all.npz
python -m acie split --data data/all.npz --source HUI360 --target SSUP-A --out splits/hui_to_ssup.json
python -m acie audit --data data/all.npz --split splits/hui_to_ssup.json
```

导出器使用原始可见窗口建立本库特征，并非逐比特复制官方归一化张量。检测到重投影配置时会拒绝运行；源帧修复／重投影不能被静默混入本方法。

### 3. 训练与预测

```bash
python -m acie train --data data/all.npz --split splits/hui_to_ssup.json \
  --config configs/full.yaml --out outputs/hui_to_ssup/full
python -m acie predict --data data/all.npz --checkpoint outputs/hui_to_ssup/full/best.pt \
  --out outputs/hui_to_ssup/full/predictions.jsonl --bootstrap 1000
```

默认 CPU；需要 GPU 时在 YAML 中显式设置 `device: cuda:0`。没有 GPU 时会报错，不静默切换并改变时间统计。中断恢复使用相同数据、划分、配置和 `--resume`；配置变化须开新运行目录。

源域方法筛选使用 `scripts/make_light_csvs.py` 控制内存，`scripts/export_source_streaming.py` 逐录制调用官方 loader，`scripts/run_controls.py` 在同一日期划分上运行六种方法，`scripts/summarize_controls.py` 生成成对差值。原始逐种子结果位于 `outputs/source_controls/`。

## 结果与主要配置

冻结跨域对照共完成 40 次运行。五种子集成中，冻结几何后的行为残差 `dual_no_pair` 相对几何基线在两个方向均提高 AP；配对模型相对同类别权重的无配对消融只在一个方向提高，未通过预注册的双向为正判据。当前证据支持行为增量，不支持把配对损失作为稳健跨域贡献。容量匹配的日期组内行为置换对照进一步表明：破坏行为—样本对应关系后，模型回到纯几何水平；真实行为相对置换行为的 AP 增益在两个方向均通过组 bootstrap。完整数值和区间见 `reports/frozen_cross_domain_evaluation_2026-09-14.md` 与 `reports/behavior_information_control_2026-09-15.md`。E4 将冻结目标窗口从约 1.07 秒逐级前移到 3.20 秒：行为残差相对几何的 AP 点估计在两方向、五个位置均为正，但共同轨迹集只有约 1.07 秒处的日期组 bootstrap 区间在两方向均高于 0；源域冻结阈值的目标召回很低，因此不主张可部署的跨域告警性能。详见 `reports/horizon_evaluation_2026-09-15.md`。E3 已将 40 份冻结运行的接近条件诊断正式汇总；中心 `dual_no_pair` 在两个方向的局部“最终−几何”排序准确率变化为 +0.0175 和 +0.0054，但种子方差较大，因此只作诊断，不作为配对机制贡献。补充的同结构随机配对控制也未显示双向一致收益。两种树模型呈方向依赖：HistGB 前向集成 AP 为 0.1318、反向为 0.4712；这排除了“ACIE 全面优于简单非线性模型”的表述，但不改变行为正确对应相对同容量置乱对照的双向增量结论。详见 `reports/e3_pair_diagnostic_2026-09-15.md` 与 `reports/revised_scope_controls_2026-09-15.md`。

`best.pt` 保存模型、源域标准化器、匹配半径、源域阈值及划分。`geometry.last.pt`、`evidence.last.pt` 保存优化器和随机状态；`.history.json` 保存每轮源域开发 AP。`matching.json` 和 `pairs.train.jsonl` 记录支持覆盖及真实配对。预测 JSONL 每行包含样本 ID、分数、几何分数和行为增量；对应 `.metrics.json` 包含区间和诊断。

默认 `hidden=64`、几何训练 30 轮、增量训练 50 轮、`pair_lambda=0.3`、每个正例至多 3 对、候选近邻至多 128。它们是可调实现起点，**不是经过真实数据选出的最优配置**。所有选择只看源域开发集。

## 协议与边界

跨域命令使用源域 `train` 池分组留出开发集，目标域 `test` 池只评分。域内默认重新按录制分组，输出明确标为本地协议。保留官方测试池时用 `scripts/split_from_pools.py`；仍需单列源域开发选择的差异。原生参考基线调用 `scripts/run_official_baseline.py`，不能将本库派生协议直接拼到原论文成绩列。

不把 sigmoid 输出自动称为已校准部署概率；不把增量解释为因果效应、心理状态或真实部署安全收益。主版本实现冻结几何分支，未实现折外几何训练、多模态情绪、新数据集或真机试验。
