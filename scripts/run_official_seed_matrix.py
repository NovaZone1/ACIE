#!/usr/bin/env python3
"""Run a resumable official seed matrix under the frozen fair protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


SPECS = {
    "HUI360_to_SSUP-A": {
        "source_data": "hui_processed",
        "target_data": "hui_processed",
        "slug": "hui360_to_ssup_a",
    },
    "SSUP-A_to_HUI360": {
        "source_data": "hui_processed",
        "target_data": "hui_processed",
        "slug": "ssup_a_to_hui360",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--direction", choices=sorted(SPECS), required=True)
    parser.add_argument("--model", choices=["mlp", "lstm", "stgcn"], default="mlp")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    root = args.root.resolve()
    data_root = args.data_root.resolve()
    upstream = root / "external" / "HUI360-Baselines"
    protocol_dir = root / "official_baselines" / "protocol_v1" / args.direction
    base_config_path = protocol_dir / f"fair_train_{args.model}.yaml"
    target_config_path = protocol_dir / f"fair_target_eval_{args.model}.yaml"
    generated_dir = protocol_dir / f"seed_configs_{args.model}"
    generated_dir.mkdir(parents=True, exist_ok=True)
    result_root = root / "outputs" / "official_baselines" / "fair_v1" / args.direction / args.model
    spec = SPECS[args.direction]

    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    for seed in args.seeds:
        output = result_root / f"seed{seed}"
        metrics_path = output / "metrics.json"
        checkpoint_copy = output / "checkpoint_source_best_ap.pth"
        if metrics_path.is_file() and checkpoint_copy.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("seed") == seed and sha256(checkpoint_copy) == metrics.get("checkpoint_sha256"):
                print(f"SKIP {args.direction} seed={seed}: validated result exists", flush=True)
                continue

        config = dict(base_config)
        experiment_name = f"acie_fair_{spec['slug']}_{args.model}_seed{seed}_v2"
        config["experiment_name"] = experiment_name
        config["acie_seed"] = seed
        seed_config = generated_dir / f"fair_train_{args.model}_seed{seed}.yaml"
        seed_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        env = os.environ.copy()
        env["HUI360_SEED"] = str(seed)
        candidates_before = set((upstream / "experiments" / "results").glob(f"{experiment_name}_*"))
        train_cmd = [
            sys.executable, str(upstream / "training.py"),
            "--hp_config_file", str(seed_config), "--save_model",
            "--preload_data", "--offline_mode",
            "--hf_local_dir", str(data_root / "datasets" / spec["source_data"]),
            "--num_workers", str(args.num_workers),
        ]
        started = time.time()
        print(f"TRAIN {args.direction} seed={seed}", flush=True)
        subprocess.run(train_cmd, cwd=upstream, env=env, check=True)
        new_candidates = set((upstream / "experiments" / "results").glob(f"{experiment_name}_*")) - candidates_before
        if len(new_candidates) != 1:
            raise RuntimeError(f"Expected one new training directory, got {sorted(map(str, new_candidates))}")
        training_dir = new_candidates.pop()
        checkpoint = training_dir / f"{args.model}_interaction_model_best_ap.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        output.mkdir(parents=True, exist_ok=True)
        eval_cmd = [
            sys.executable, str(root / "scripts" / "run_official_fair_eval.py"),
            "--root", str(root), "--direction", args.direction,
            "--checkpoint", str(checkpoint), "--target-config", str(target_config_path),
            "--data-root", str(data_root / "datasets" / spec["target_data"]),
            "--out", str(output), "--num-workers", str(args.num_workers), "--seed", str(seed),
        ]
        print(f"EVAL  {args.direction} seed={seed}", flush=True)
        subprocess.run(eval_cmd, cwd=root, env=env, check=True)
        shutil.copy2(checkpoint, checkpoint_copy)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if sha256(checkpoint_copy) != metrics["checkpoint_sha256"]:
            raise RuntimeError("Copied checkpoint hash mismatch")
        provenance = {
            "schema": "acie.official-seed-run.v1",
            "direction": args.direction,
            "model": args.model,
            "seed": seed,
            "seed_control": "HUI360_SEED environment variable through patches/hui360_seed_override.patch",
            "train_config": str(seed_config.relative_to(root)),
            "train_config_sha256": sha256(seed_config),
            "training_result_dir": str(training_dir),
            "checkpoint_selection": "best source-validation AP",
            "target_labels_used_for_selection": False,
            "wall_seconds_train_and_eval": time.time() - started,
        }
        (output / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(f"DONE  {args.direction} seed={seed} AP={metrics['AP']:.6f} AUROC={metrics['AUROC']:.6f}", flush=True)


if __name__ == "__main__":
    main()
