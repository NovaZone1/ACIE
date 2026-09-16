from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from acie.data import Bundle, make_split, prepare, resolve_split
from acie.demo import make_fixture
from acie.features import SourceScaler
from acie.vv_engine import (
    DEFAULT_CONFIG,
    _fit,
    derived_seed,
    score_locked_target,
    train_source_model,
)
from acie.vv_models import (
    EXPECTED_PARAMETERS,
    MODEL_IDS,
    build_model,
    configure_training_mode,
    effective_parameter_count,
)
from acie.vv_trees import candidate_grid, score_locked_tree, train_source_tree, tree_features


@pytest.fixture(scope="module")
def vv_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("vv_contract")
    bundle = prepare(make_fixture(root / "fixture", groups=8, per_group=8, frames=8),
                     root / "bundle.npz")
    return bundle, make_split(bundle, "synthetic_A", "synthetic_B"), root


def test_vv_capacity_and_zero_head_contract():
    for model_id in MODEL_IDS:
        model = build_model(model_id)
        assert effective_parameter_count(model) == EXPECTED_PARAMETERS[model_id]
    for model_id in ("Q", "Jc", "Jw", "JwD", "F"):
        head = build_model(model_id).behavior.head
        assert torch.count_nonzero(head.weight) == 0
        assert torch.count_nonzero(head.bias) == 0
    for model_id in ("C64", "Cm"):
        model = build_model(model_id)
        assert any(isinstance(layer, torch.nn.ReLU) for layer in model.fusion)
        assert torch.count_nonzero(model.fusion[-1].weight) == 0


def test_vv_shared_reference_and_unique_treatment_factor():
    torch.manual_seed(7)
    shared_g = build_model("G").geometry.state_dict()
    models = {}
    for model_id in ("F", "Jw", "JwD"):
        torch.manual_seed(derived_seed(11, "behavior_init"))
        model = build_model(model_id)
        model.geometry.load_state_dict(shared_g)
        models[model_id] = model
    for key in models["F"].geometry.state_dict():
        assert torch.equal(models["F"].geometry.state_dict()[key], models["Jw"].geometry.state_dict()[key])
        assert torch.equal(models["F"].geometry.state_dict()[key], models["JwD"].geometry.state_dict()[key])
    for key in models["F"].behavior.state_dict():
        assert torch.equal(models["F"].behavior.state_dict()[key], models["Jw"].behavior.state_dict()[key])
    a, q, y = torch.randn(5, 32), torch.randn(5, 8, 119), torch.tensor([0., 1., 0., 1., 0.])
    models["F"].eval(); models["Jw"].eval()
    torch.testing.assert_close(models["F"](a, q)[0], models["Jw"](a, q)[0], rtol=0, atol=0)
    before = {k: v.clone() for k, v in models["F"].geometry.state_dict().items()}
    for model_id in ("F", "Jw"):
        model = models[model_id]
        configure_training_mode(model, model_id)
        model.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(model(a, q)[0], y).backward()
    assert all(parameter.grad is None for parameter in models["F"].geometry.parameters())
    assert any(parameter.grad is not None for parameter in models["Jw"].geometry.parameters())
    optimizer = torch.optim.SGD([p for p in models["F"].parameters() if p.requires_grad], lr=0.1)
    optimizer.step()
    assert all(torch.equal(before[k], models["F"].geometry.state_dict()[k]) for k in before)


def test_vv_additive_decomposition_is_logit_level():
    model = build_model("F").eval()
    a, q = torch.randn(9, 32), torch.randn(9, 8, 119)
    total, geometry, evidence = model(a, q)
    torch.testing.assert_close(total, geometry + evidence, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(torch.sigmoid(total), torch.sigmoid(geometry) + torch.sigmoid(evidence))


def test_vv_geometry_modes_survive_eval_transitions():
    frozen, warm, warm_dropout = (build_model(name) for name in ("F", "Jw", "JwD"))
    for model, model_id in ((frozen, "F"), (warm, "Jw"), (warm_dropout, "JwD")):
        model.eval()
        configure_training_mode(model, model_id)
    assert not frozen.geometry.training and not warm.geometry.training
    assert warm_dropout.geometry.training
    assert all(not p.requires_grad for p in frozen.geometry.parameters())
    assert all(p.requires_grad for p in warm.geometry.parameters())
    assert all(p.requires_grad for p in warm_dropout.geometry.parameters())


def test_vv_source_training_ignores_target_features_and_labels(vv_dataset, tmp_path):
    bundle, split, _ = vv_dataset
    altered = Bundle(bundle.a.copy(), bundle.q.copy(), bundle.y.copy(), copy.deepcopy(bundle.meta),
                     copy.deepcopy(bundle.provenance))
    test_ix = resolve_split(altered, split)["test"]
    altered.a[test_ix] += 10000
    altered.q[test_ix] -= 10000
    altered.y[test_ix] = 1 - altered.y[test_ix]
    config = {"geometry_epochs": 1, "stage2_epochs": 1, "single_stage_epochs": 1,
              "patience": 2, "num_threads": 1, "device": "cpu"}
    train_source_model(bundle, split, "Q", 11, tmp_path / "original", config)
    train_source_model(altered, split, "Q", 11, tmp_path / "altered", config)
    first = torch.load(tmp_path / "original/best.pt", map_location="cpu", weights_only=True)
    second = torch.load(tmp_path / "altered/best.pt", map_location="cpu", weights_only=True)
    assert json.loads((tmp_path / "original/scaler.json").read_text()) == json.loads((tmp_path / "altered/scaler.json").read_text())
    for key in first["state"]:
        assert torch.equal(first["state"][key], second["state"][key])


def test_vv_all_models_receive_only_a_and_q():
    a, q = torch.randn(3, 32), torch.randn(3, 8, 119)
    for model_id in MODEL_IDS:
        output = build_model(model_id)(a, q)
        assert len(output) == 3 and all(value.shape == (3,) for value in output)
    with pytest.raises(TypeError):
        build_model("F")(a, q, torch.ones(3))


def test_vv_interrupted_resume_matches_continuous(vv_dataset, tmp_path):
    bundle, split, _ = vv_dataset
    ix = resolve_split(bundle, split)
    scaler = SourceScaler.fit(bundle.a[ix["train"]], bundle.q[ix["train"]])
    a, q = scaler.transform(bundle.a, bundle.q)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(batch_size=16, patience=10, num_threads=1, device="cpu", dropout=0.1)
    torch.manual_seed(derived_seed(22, "model_init"))
    initial = build_model("Q")
    state = copy.deepcopy(initial.state_dict())

    continuous = build_model("Q"); continuous.load_state_dict(state)
    _fit(continuous, "Q", a, q, bundle.y, ix["train"], ix["val"], cfg, 22, 3,
         tmp_path / "continuous", False)
    interrupted = build_model("Q"); interrupted.load_state_dict(state)
    (tmp_path / "interrupted").mkdir()
    _fit(interrupted, "Q", a, q, bundle.y, ix["train"], ix["val"], cfg, 22, 1,
         tmp_path / "interrupted", False)
    resumed = build_model("Q"); resumed.load_state_dict(state)
    _fit(resumed, "Q", a, q, bundle.y, ix["train"], ix["val"], cfg, 22, 3,
         tmp_path / "interrupted", True)
    for key in continuous.state_dict():
        torch.testing.assert_close(continuous.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)


def test_vv_training_and_target_scoring_are_decoupled(vv_dataset, tmp_path):
    bundle, split, _ = vv_dataset
    run_dir = tmp_path / "G"
    config = {"geometry_epochs": 1, "stage2_epochs": 1, "single_stage_epochs": 1,
              "patience": 2, "num_threads": 1, "device": "cpu"}
    train_source_model(bundle, split, "G", 33, run_dir, config)
    assert (run_dir / "val_predictions.jsonl").is_file()
    assert not (run_dir / "test_predictions.jsonl").exists()
    before = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["state"]
    result = score_locked_target(bundle, run_dir / "best.pt", run_dir / "test_predictions.jsonl")
    after = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["state"]
    assert result["weights_updated"] is False and result["scaler_updated"] is False
    assert (run_dir / "test_predictions.jsonl").is_file()
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_vv_tree_grid_and_feature_contract(vv_dataset):
    bundle, split, _ = vv_dataset
    train_ix = resolve_split(bundle, split)["train"]
    scaler = SourceScaler.fit(bundle.a[train_ix], bundle.q[train_ix])
    assert tree_features(scaler, bundle.a, bundle.q).shape[1] == 389
    assert len(candidate_grid("histgb", "r1")) == 3
    assert len(candidate_grid("random_forest", "r1")) == 3
    assert len(candidate_grid("histgb", "r2")) == 6
    assert len(candidate_grid("random_forest", "r2")) == 6


def test_vv_tree_training_and_scoring_are_decoupled(vv_dataset, tmp_path):
    bundle, split, _ = vv_dataset
    run = tmp_path / "histgb"
    result = train_source_tree(bundle, split, run, "histgb", 11, "r1")
    assert result["target_scoring_performed"] is False
    assert (run / "val_predictions.jsonl").is_file()
    assert not (run / "test_predictions.jsonl").exists()
    score = score_locked_tree(bundle, run / "tree.joblib", run / "test_predictions.jsonl")
    assert score["weights_updated"] is False and score["scaler_updated"] is False
