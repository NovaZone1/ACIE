#!/usr/bin/env python3
"""Render the frozen E4 horizon summary as a reviewable Markdown report."""
from __future__ import annotations
import argparse
from pathlib import Path
from acie.io import read_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--summary',type=Path,required=True);p.add_argument('--coverage',type=Path,required=True)
p.add_argument('--validation',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();s=read_json(a.summary);coverage=read_json(a.coverage);validation=read_json(a.validation)

def rows_for(set_name):
    lines=[]
    for direction,result in s['directions'].items():
        for h in result['horizons']:
            z=h['sets'][set_name];g=z['methods']['geometry']['ensemble_ranking'];d=z['methods']['dual_no_pair']['ensemble_ranking'];c=z['dual_no_pair_minus_geometry'];lo,hi=c['interval']
            mark=' **' if lo>0 else ''
            lines.append(f"| {direction.replace('_to_',' → ')} | {h['cutoff_seconds']:.2f} | {z['n']} / {z['positives']} | {g['AP']:.4f} | {d['AP']:.4f} | {c['point']:+.4f} [{lo:+.4f}, {hi:+.4f}]{mark} |")
    return '\n'.join(lines)

def operating_rows():
    lines=[]
    for direction,result in s['directions'].items():
        for h in result['horizons']:
            z=h['sets']['common_tracks'];g=z['methods']['geometry']['seed_summary_at_source_frozen_thresholds'];d=z['methods']['dual_no_pair']['seed_summary_at_source_frozen_thresholds']
            lines.append(f"| {direction.replace('_to_',' → ')} | {h['cutoff_seconds']:.2f} | {100*g['recall_mean']:.2f}% | {100*g['false_positive_rate_mean']:.2f}% | {100*d['recall_mean']:.2f}% | {100*d['false_positive_rate_mean']:.2f}% |")
    return '\n'.join(lines)

def timing_rows():
    lines=[]
    for domain in ['SSUP-A','HUI360']:
        for h in coverage[domain]['horizons']:
            t=h['positive_event_lead_seconds']
            lines.append(f"| {domain} | {h['cutoff_seconds']:.2f} | {t['median']:.3f} [{t['min']:.3f}, {t['max']:.3f}] | {h['n']} / {h['positives']} |")
    return '\n'.join(lines)

text=f'''# E4 提前量与误报实验（2026-09-15）

## 结论

冻结几何后的行为残差在两个迁移方向、五个提前量上的 AP 点估计均高于纯几何模型，但在控制样本构成的共同轨迹分析中，只有约 1.07 秒截断点的日期组聚类 95% 区间稳定高于 0。1.60–3.20 秒的差值区间均跨 0，因此当前证据只能稳健支持约 1.07 秒处的额外行为排序证据，不能声称已经证明更长提前量上的稳定增益。

源域开发集冻结阈值直接迁移后，两种方法的目标域召回率都很低。`dual_no_pair` 在共同轨迹集上的五种子平均召回为 1.47%–5.93%，同时 FPR 为 0.03%–2.29%。这些结果支持排序分析，不支持可直接部署的跨域提前报警能力。

## 冻结设计

- 协议摘要：`{s['protocol_digest']}`。
- 输入长度固定为 32 帧，帧率 15 fps；在官方16帧截断窗口上，将完整观察前缀分别向前移动 0、8、16、24、32 帧。
- 正例沿用已有交互事件，负例沿用官方最大人体尺度代理锚点；标签、轨迹身份和正负比例均不用于重新选择截断点。
- 评估 `geometry` 与容量匹配的 `dual_no_pair`，复用既有五种子检查点和各自源域开发集阈值，不重新训练或调阈值。
- 每个提前量报告完整可用集；主要时间比较使用五点都可用的共同轨迹集，以排除样本构成变化。
- AP 差值区间按录制日期聚类、2,000次 bootstrap。粗体标记区间下界大于0。
- 这是主结果完成后的 E4 跟进实验，不表述为事前注册结果。

## 实际事件时间与覆盖率

表中“名义秒数”由冻结帧截断换算；“实际正例时间”来自观察末帧已有的 `time_to_first_interaction`，给出中位数与范围。

| 目标域 | 名义秒数 | 实际正例时间中位数 [范围]（秒） | 完整集 n / 正例 |
|---|---:|---:|---:|
{timing_rows()}

五点共同轨迹集为：SSUP-A 2,290条、95个正例；HUI360 158条、27个正例。HUI360录制日期组较少且正例数小，长提前量区间较宽。

## 主要结果：共同轨迹集

| 迁移方向 | 名义秒数 | n / 正例 | Geometry AP | Dual AP | Dual − Geometry AP [95% CI] |
|---|---:|---:|---:|---:|---:|
{rows_for('common_tracks')}

共同轨迹结果表明，行为残差的点估计在所有位置为正，但只有两个方向的1.07秒点通过“差值区间下界大于0”的稳健判据。SSUP-A → HUI360 在2.67秒处的差值较大，但只有27个共同正例且区间跨0，不能据此选择性宣称长提前量有效。

## 补充结果：各点完整可用集

| 迁移方向 | 名义秒数 | n / 正例 | Geometry AP | Dual AP | Dual − Geometry AP [95% CI] |
|---|---:|---:|---:|---:|---:|
{rows_for('complete')}

完整集会随提前量变化。HUI360 → SSUP-A 只有1.07秒差值区间高于0；SSUP-A → HUI360 的1.07和1.60秒区间高于0。由于完整集同时改变了轨迹构成，论文关于提前量变化的主解释采用共同轨迹表。

## 源域冻结阈值下的误报与召回

下表为共同轨迹集上五个种子的均值；每个种子使用训练时保存的源域开发阈值。

| 迁移方向 | 名义秒数 | Geometry Recall | Geometry FPR | Dual Recall | Dual FPR |
|---|---:|---:|---:|---:|---:|
{operating_rows()}

阈值没有利用目标域校准，因此低误报同时伴随很低的召回。论文应明确区分阈值无关的跨域排序能力与工作点可迁移性。

## 一致性与可复现性

- 1.07秒数据包与冻结主实验数据的 `a`、`q` 特征逐元素相同，两个域最大绝对误差均为0。
- 旧、新基准概率在20个方法／种子组合上的最大差异为 `{validation['max_base_score_difference']:.3g}`；logit 最大差异为 `1.91e-6`，来自CPU浮点归约，未改变任何排序或指标。
- 自动验证通过 `{validation['passed']}/{validation['total']}` 项，覆盖100个种子预测文件、数据包与检查点哈希、标签不变、阈值冻结、共同集合和基准复现。
- 图：`reports/figures/horizon_e4_common_tracks.png` 与同名 PDF。

复现实验：

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_horizon_bundles.py \\
  --root . --raw-root data/hui_light --out data/horizon_evaluation \\
  --protocol protocols/horizon_evaluation_2026-09-15.json
PYTHONPATH=src .venv/bin/python scripts/run_horizon_evaluation.py \\
  --root . --protocol protocols/horizon_evaluation_2026-09-15.json \\
  --out outputs/horizon_evaluation --device cpu --bootstrap 2000
PYTHONPATH=src .venv/bin/python scripts/validate_horizon_evaluation.py \\
  --root . --protocol protocols/horizon_evaluation_2026-09-15.json \\
  --outputs outputs/horizon_evaluation \\
  --out validation/horizon_evaluation_validation.json
```

## 论文主张边界

E4支持的表述是：“在约1.07秒的冻结观察截断处，与样本正确对应的行为残差在两个跨域方向都提供了超出接近几何的排序信息。” 更长提前量上的差值点估计仍为正，但证据不足。固定源域阈值的召回很低，因此不把该结果外推为实时系统告警性能。

本次只完成与论文中央主张直接对应的同容量 `geometry`／`dual_no_pair` 比较。官方架构在多提前量上的扩展应单列为后续补充实验，不能把当前结果描述成“所有官方方法均已完成E4”。
'''
a.out.write_text(text,encoding='utf-8')
