#!/usr/bin/env python3
"""Derive fair LSTM configs from the audited MLP data scope and official hyperparameters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


SPECS = {
    "HUI360_to_SSUP-A": {
        "upstream_mlp": "experiments/configs/cross_hui_ssup/mlp_base.yaml",
        "upstream_lstm": "experiments/configs/cross_hui_ssup/lstm_base.yaml",
    },
    "SSUP-A_to_HUI360": {
        "upstream_mlp": "experiments/configs/cross_ssup_hui/mlp_base.yaml",
        "upstream_lstm": "experiments/configs/cross_ssup_hui/lstm_base.yaml",
    },
}
ALLOWED_OFFICIAL_DIFFERENCES = {"force_model_type", "epochs", "experiment_name"}


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
    for direction, spec in SPECS.items():
        official_mlp = yaml.safe_load((upstream / spec["upstream_mlp"]).read_text(encoding="utf-8"))
        official_lstm = yaml.safe_load((upstream / spec["upstream_lstm"]).read_text(encoding="utf-8"))
        changed = {key for key in set(official_mlp) | set(official_lstm) if official_mlp.get(key) != official_lstm.get(key)}
        if changed != ALLOWED_OFFICIAL_DIFFERENCES:
            raise ValueError(f"Unexpected official MLP/LSTM differences for {direction}: {sorted(changed)}")
        if official_lstm["force_model_type"] != "lstm":
            raise ValueError("Official LSTM config has unexpected model type")

        config_dir = root / "official_baselines" / "protocol_v1" / direction
        written = {}
        for role in ("fair_train", "fair_target_eval", "native_available_subset"):
            mlp_path = config_dir / f"{role}_mlp.yaml"
            config = yaml.safe_load(mlp_path.read_text(encoding="utf-8"))
            config["force_model_type"] = "lstm"
            config["epochs"] = int(official_lstm["epochs"])
            config["experiment_name"] = config["experiment_name"].replace("_mlp_", "_lstm_")
            config["comment"] = config.get("comment", "") + "; official LSTM architecture and epoch count"
            output = config_dir / f"{role}_lstm.yaml"
            payload = yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode()
            output.write_bytes(payload)
            written[f"{role}_lstm"] = {"path": str(output.relative_to(root)), "sha256": digest(payload)}
        protocol["directions"][direction]["configs"].update(written)
        model_spec[direction] = {
            "epochs": int(official_lstm["epochs"]),
            "hidden_dim": int(official_lstm["lstm_hidden_dim"]),
            "num_layers": int(official_lstm["lstm_num_layers"]),
            "dropout": float(official_lstm["lstm_dropout"]),
            "official_config": spec["upstream_lstm"],
        }

    protocol.setdefault("model_specs", {})["lstm"] = model_spec
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(model_spec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
