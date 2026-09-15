# 来源与实现归属

本库为所述研究方案的原创实现，第三方源码没有随包复制。外部链接和接口依据在 2026-09-13 核查；下载时应锁定真实 commit。只确认公开入口与已阅读代码合同，不将其等同于完整复现。

- HUI360 原论文：<https://arxiv.org/abs/2608.11051>
- 作者代码与数据入口：<https://github.com/hucebot/HUI360-Baselines>；<https://hucebot.github.io/hui360/>
- 处理后数据仓库：<https://huggingface.co/datasets/rlorlou/HUI360>
- 适配器接口来源：官方 `datasets/HUIDataset.py`、`utils/loader_utils.py`、`utils/data_utils.py` 和所选实验 YAML。
- Self-Supervised Prediction of the Intention to Interact with a Service Robot：<https://arxiv.org/abs/2309.07477>
- Predicting the Intention to Interact with a Service Robot: The Role of Gaze Cues：<https://arxiv.org/abs/2404.01986>
- MINT-RVAE：<https://arxiv.org/abs/2509.22573>

数据集、预训练权重、原作者模型及论文各遵循其自身许可。附带 MIT 只授权本包新增代码。代码中的图模型、树模型及重采样对照是独立实现，不可当作原论文官方检查点或已复现成绩。
