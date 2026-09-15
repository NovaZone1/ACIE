"""Aggregate the frozen cross-domain pairing diagnostics into an E3 table.

This script never retrains or rescores a model. It reads the already frozen target
prediction metrics and reports ordering accuracy on label-conditioned evaluation
pairs whose scaler, radius, and support rules were fitted on the source split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, stdev

from acie.io import digest_file, read_json, write_json

METHODS = ["geometry", "dual_no_pair", "weighted_no_pair", "full_selected"]
FIELDS = [
    "n_pairs",
    "n_positive",
    "positive_coverage",
    "mean_distance",
    "p95_distance",
    "unique_negative_tracks",
    "max_negative_track_uses_observed",
    "geometry_accuracy",
    "evidence_accuracy",
    "final_accuracy",
]


def stats(values: list[float]) -> dict:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": mean(clean),
        "std": stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
    }


def fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def mean_sd(item: dict, digits: int = 4) -> str:
    return f"{fmt(item['mean'], digits)}±{fmt(item['std'], digits)}"


def render_report(payload: dict) -> str:
    lines = [
        "# E3 接近条件配对诊断（2026-09-15）",
        "",
        "## 诊断口径",
        "",
        "本表直接汇总冻结跨域实验的 40 份测试指标，不重新训练、不重新选模。几何标准化器、匹配半径和支持约束均由源域训练集确定；目标标签只在模型预测完成后用于构造异标签诊断对。表中的准确率表示正样本得分高于几何接近负样本的比例，平分计 0.5，它不是分类准确率，也不参与训练或阈值选择。",
        "",
        f"冻结协议摘要：`{payload['frozen_protocol_digest']}`；输入运行目录哈希：`{payload['input_manifest_digest']}`。",
        "",
        "## 正式结果表",
        "",
        "五个训练种子的结果以均值±样本标准差报告。`evidence` 对纯几何模型恒为零，因此其 0.5 仅作为实现校验，不作方法结论。",
        "",
        "| 方向 | 方法 | 配对数 | 正例覆盖率 | 平均标准化距离 | 几何排序准确率 | 行为证据排序准确率 | 最终排序准确率 | 最终−几何 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for direction, direction_data in payload["directions"].items():
        for method in METHODS:
            item = direction_data["methods"][method]
            s = item["summary"]
            lines.append(
                "| " + " | ".join([
                    direction,
                    method,
                    mean_sd(s["n_pairs"], 1),
                    mean_sd(s["positive_coverage"]),
                    mean_sd(s["mean_distance"]),
                    mean_sd(s["geometry_accuracy"]),
                    mean_sd(s["evidence_accuracy"]),
                    mean_sd(s["final_accuracy"]),
                    mean_sd(item["final_minus_geometry"]),
                ]) + " |"
            )
    lines += [
        "",
        "## 与当前主张的关系",
        "",
        "`dual_no_pair` 是当前行为残差主张的中心模型。该诊断检验它能否在几何接近的正负样本之间维持局部排序；正式 AP、AUROC 和日期组 bootstrap 仍以冻结跨域主结果为准。配对诊断不提供因果识别，也不支持把目标标签用于部署时配对。",
        "",
    ]
    for direction, direction_data in payload["directions"].items():
        central = direction_data["methods"]["dual_no_pair"]
        lines.append(
            f"- {direction}：`dual_no_pair` 最终排序准确率 {mean_sd(central['summary']['final_accuracy'])}，"
            f"相对其几何分支变化 {mean_sd(central['final_minus_geometry'])}。"
        )
    lines += [
        "",
        "## 证据边界",
        "",
        "诊断对依赖目标测试标签，只能用于离线评估。两个方向的可用目标日期组较少，五种子均值反映训练随机性，但不能替代独立数据集复验。配对数与覆盖率由冻结的源域匹配规则和目标样本支持共同决定。",
        "",
        "逐种子原始值和完整字段保存在 `outputs/e3_pair_diagnostic/summary.json`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    protocol = read_json(args.runs / "FROZEN_PROTOCOL.json")
    seeds = [int(seed) for seed in protocol["seeds"]]
    directions = sorted(path.name for path in args.runs.iterdir()
                        if path.is_dir() and "_to_" in path.name)
    if len(directions) != 2:
        raise ValueError(f"Expected two frozen directions, found {directions}")

    payload = {
        "schema": "acie.e3-pair-diagnostic.v1",
        "source": "existing frozen cross-domain target prediction metrics",
        "selection": "source-fitted scaler/radius/support; target labels used after prediction for diagnostics only",
        "frozen_protocol_digest": protocol["lock_digest"],
        "frozen_protocol_sha256": digest_file(args.runs / "FROZEN_PROTOCOL.json"),
        "seeds": seeds,
        "methods": METHODS,
        "directions": {},
    }
    manifest = []
    for direction in directions:
        d_payload = {"methods": {}}
        for method in METHODS:
            runs = []
            for seed in seeds:
                path = args.runs / direction / method / f"seed{seed}" / "test_predictions.metrics.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                source = read_json(path)
                diagnostic = source["pair_diagnostic"]
                missing = [field for field in FIELDS if field not in diagnostic]
                if missing:
                    raise ValueError(f"Missing {missing} in {path}")
                row = {"seed": seed, **{field: diagnostic[field] for field in FIELDS}}
                row["final_minus_geometry"] = (
                    None if diagnostic["final_accuracy"] is None or diagnostic["geometry_accuracy"] is None
                    else diagnostic["final_accuracy"] - diagnostic["geometry_accuracy"]
                )
                runs.append(row)
                manifest.append({"path": str(path), "sha256": digest_file(path)})
            summary = {field: stats([run[field] for run in runs]) for field in FIELDS}
            delta = stats([run["final_minus_geometry"] for run in runs])
            d_payload["methods"][method] = {
                "runs": runs,
                "summary": summary,
                "final_minus_geometry": delta,
            }
        payload["directions"][direction] = d_payload

    # The digest binds this table to the exact 40 metric files without modifying them.
    from acie.io import digest_object
    payload["input_manifest"] = manifest
    payload["input_manifest_digest"] = digest_object(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"Aggregated {len(manifest)} frozen diagnostic files into {args.out}")


if __name__ == "__main__":
    main()
