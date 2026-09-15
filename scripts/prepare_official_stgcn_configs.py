#!/usr/bin/env python3
"""Derive fair ST-GCN configs from audited data scope and official hyperparameters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


SPECS = {
    "HUI360_to_SSUP-A": {"mlp": "experiments/configs/cross_hui_ssup/mlp_base.yaml", "model": "experiments/configs/cross_hui_ssup/stgcn_base.yaml"},
    "SSUP-A_to_HUI360": {"mlp": "experiments/configs/cross_ssup_hui/mlp_base.yaml", "model": "experiments/configs/cross_ssup_hui/stgcn_base.yaml"},
}
ALLOWED_DIFFERENCES = {"force_model_type", "lr_scheduler_type", "epochs", "normalize_keypoints_in_box", "standardize_data", "experiment_name"}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    upstream = root / "external" / "HUI360-Baselines"
    protocol_path = root / "official_baselines" / "protocol_v1" / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    model_spec = {}

    for direction, paths in SPECS.items():
        official_mlp = yaml.safe_load((upstream / paths["mlp"]).read_text(encoding="utf-8"))
        official_model = yaml.safe_load((upstream / paths["model"]).read_text(encoding="utf-8"))
        changed = {key for key in set(official_mlp) | set(official_model) if official_mlp.get(key) != official_model.get(key)}
        if changed != ALLOWED_DIFFERENCES:
            raise ValueError(f"Unexpected official MLP/ST-GCN differences for {direction}: {sorted(changed)}")
        config_dir = root / "official_baselines" / "protocol_v1" / direction
        written = {}
        for role in ("fair_train", "fair_target_eval", "native_available_subset"):
            config = yaml.safe_load((config_dir / f"{role}_mlp.yaml").read_text(encoding="utf-8"))
            for key in ALLOWED_DIFFERENCES - {"experiment_name"}:
                config[key] = official_model[key]
            config["experiment_name"] = config["experiment_name"].replace("_mlp_", "_stgcn_")
            config["comment"] = config.get("comment", "") + "; official ST-GCN architecture, preprocessing, scheduler and epoch count"
            output = config_dir / f"{role}_stgcn.yaml"
            payload = yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode()
            output.write_bytes(payload)
            written[f"{role}_stgcn"] = {"path": str(output.relative_to(root)), "sha256": digest(payload)}
        protocol["directions"][direction]["configs"].update(written)
        model_spec[direction] = {
            "epochs": int(official_model["epochs"]),
            "lr_scheduler_type": official_model["lr_scheduler_type"],
            "normalize_keypoints_in_box": bool(official_model["normalize_keypoints_in_box"]),
            "standardize_data": official_model["standardize_data"],
            "in_channels": int(official_model["stgcn_in_channels"]),
            "layout": official_model["stgcn_layout"],
            "edge_importance_weighting": bool(official_model["stgcn_edge_importance_weighting"]),
            "official_config": paths["model"],
        }
    protocol.setdefault("model_specs", {})["stgcn"] = model_spec
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(model_spec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
