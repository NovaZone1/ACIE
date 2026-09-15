"""Render the revised-scope control summary as a concise Chinese report."""
from __future__ import annotations

import argparse
from pathlib import Path

from acie.io import read_json

METHOD_LABELS = {
    "random_pair": "随机配对（同结构）",
    "histgb": "HistGradientBoosting",
    "random_forest": "Random Forest",
}


def f(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def mean_sd(item, digits=4):
    return f"{f(item['mean'], digits)}±{f(item['std'], digits)}"


def interval(item):
    values = item["bootstrap"]["interval"]
    return "NA" if values is None else f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = read_json(args.summary)
    lock = read_json(args.runs / "FROZEN_PROTOCOL.json")

    lines = [
        "# 收敛范围补充对照：随机配对与树模型（2026-09-15）",
        "",
        "## 目的与协议",
        "",
        "本轮只补充与收敛后跨域主张直接相关的稳健性证据：一个与 `full_selected` 结构和超参数相同、仅交换匹配负例身份的 `random_pair`，以及使用相同可用信息的 HistGradientBoosting 和 Random Forest 强简单基线。所有复杂度、检查点和阈值均由源域验证集确定，目标域不适配。",
        "",
        "这些实验是在主结果之后冻结的补充分析，因此标为 post-hoc，不追溯计入原始预注册。`hard_negative`、全方法 E4、完整域内矩阵和 MotionBERT 已从当前论文主张的必要范围中排除。",
        "",
        f"补充协议锁摘要：`{lock['lock_digest']}`。",
        "",
        "## 双向跨域结果",
        "",
        "单种子 AP 为五次训练的均值±样本标准差；集成指标先对五个种子的概率取算术平均。",
        "",
        "| 方向 | 方法 | 单种子 AP | 集成 AP | 集成 AUROC | 源域验证 AP | 平均训练/选模时间(s) | 平均推理时间(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for direction, payload in summary["directions"].items():
        for method, item in payload["methods"].items():
            lines.append("| " + " | ".join([
                direction,
                METHOD_LABELS[method],
                mean_sd(item["AP"]),
                f(item["ensemble"]["AP"]),
                f(item["ensemble"]["AUROC"]),
                mean_sd(item["source_val_AP"]),
                f(item["training_seconds"]["mean"], 2),
                f(item["prediction_seconds"]["mean"], 3),
            ]) + " |")
    lines += [
        "",
        "HistGradientBoosting 在关闭随机早停后是确定性的；五次重复用于保持结果矩阵一致，因此其单种子标准差为 0 不代表抽样不确定性为 0。",
        "",
        "## 冻结比较",
        "",
        "| 方向 | 比较 | 集成 AP 差值 | 日期组 bootstrap 95% 区间 |",
        "|---|---|---:|---:|",
    ]
    labels = {
        "full_selected_minus_random_pair": "full_selected − random_pair",
        "random_pair_minus_weighted_no_pair": "random_pair − weighted_no_pair",
        "histgb_minus_dual_no_pair": "HistGB − dual_no_pair",
        "random_forest_minus_dual_no_pair": "Random Forest − dual_no_pair",
    }
    for direction, payload in summary["directions"].items():
        for name, item in payload["comparisons"].items():
            lines.append(f"| {direction} | {labels[name]} | {item['AP_difference']:+.4f} | {interval(item)} |")
    lines += [
        "",
        "## 随机配对控制审计",
        "",
    ]
    for direction, payload in summary["directions"].items():
        item = payload["methods"]["random_pair"]
        lines.append(
            f"- {direction}：训练配对数 {mean_sd(item['n_train_pairs'], 1)}；实际更换负例身份比例 "
            f"{mean_sd(item['changed_negative_fraction'])}；参数量 {item['parameters']}。"
        )
    lines += [
        "",
        "自动复核确认 10/10 个随机配对模型与对应 `full_selected` 的正例顺序、配对权重、负例多重集合完全一致，冻结几何分支参数逐元素一致；因此该控制实际只改变负例与正例的对应身份。",
        "",
        "`random_pair` 只检验配对损失所用负例身份，不能替代已经完成的行为张量置乱实验。当前行为残差主张仍由 `dual_no_pair`、纯几何对照和行为对应关系置乱共同支持；若随机配对没有呈现双向一致差异，也不恢复配对损失贡献主张。",
        "",
        "## 证据边界",
        "",
        "树模型将时间序列压缩为末值、均值和标准差，因此是强简单基线而非官方时序网络的替代品。所有结果仍基于可用官方配置子集，目标日期组数量有限，聚类 bootstrap 区间可能不稳定。训练和推理时间是当前机器上的过程计时，只适合相对参考。",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
