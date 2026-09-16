"""Models for the VV follow-up protocol.

This module is intentionally separate from :mod:`acie.models`: historical
checkpoints must keep loading through the legacy implementation unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


MODEL_IDS = ("G", "Q", "C64", "Cm", "Jc", "Jw", "JwD", "F")
ADDITIVE_IDS = ("Jc", "Jw", "JwD", "F")


class CausalConv(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(cin, cout, 3, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class GeometryBranch(nn.Module):
    """Geometry encoder and its independent scalar head."""

    def __init__(self, a_dim: int = 32, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(a_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def encode(self, a: torch.Tensor) -> torch.Tensor:
        return self.encoder(a)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(a)).squeeze(-1)


class BehaviorEncoder(nn.Module):
    """Shared TCN implementation used by every behavior-aware VV model."""

    def __init__(self, q_dim: int = 119, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv(q_dim, hidden, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv(hidden, hidden, 2),
            nn.ReLU(),
        )

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        x = self.net(q.transpose(1, 2))
        return torch.cat([x.mean(-1), x[:, :, -1]], -1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.encode(q)


class BehaviorBranch(nn.Module):
    def __init__(self, q_dim: int = 119, hidden: int = 64, dropout: float = 0.1,
                 zero_head: bool = True):
        super().__init__()
        self.encoder = BehaviorEncoder(q_dim, hidden, dropout)
        self.head = nn.Linear(2 * hidden, 1)
        if zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(q)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(q)).squeeze(-1)


class GeometryModel(nn.Module):
    model_id = "G"

    def __init__(self, a_dim: int = 32, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.geometry = GeometryBranch(a_dim, hidden, dropout)
        self.spec = {"model_id": "G", "a_dim": a_dim, "q_dim": 119,
                     "geometry_hidden": hidden, "behavior_hidden": None, "dropout": dropout}

    def forward(self, a: torch.Tensor, q: torch.Tensor | None = None):
        g = self.geometry(a)
        return g, g, torch.zeros_like(g)


class BehaviorModel(nn.Module):
    model_id = "Q"

    def __init__(self, q_dim: int = 119, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.behavior = BehaviorBranch(q_dim, hidden, dropout, zero_head=True)
        self.spec = {"model_id": "Q", "a_dim": 32, "q_dim": q_dim,
                     "geometry_hidden": None, "behavior_hidden": hidden, "dropout": dropout}

    def forward(self, a: torch.Tensor | None, q: torch.Tensor):
        r = self.behavior(q)
        return r, torch.zeros_like(r), r


class AdditiveModel(nn.Module):
    def __init__(self, model_id: str, a_dim: int = 32, q_dim: int = 119,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        if model_id not in ADDITIVE_IDS:
            raise ValueError(f"Not an additive model: {model_id}")
        self.model_id = model_id
        self.geometry = GeometryBranch(a_dim, hidden, dropout)
        self.behavior = BehaviorBranch(q_dim, hidden, dropout, zero_head=True)
        self.spec = {"model_id": model_id, "a_dim": a_dim, "q_dim": q_dim,
                     "geometry_hidden": hidden, "behavior_hidden": hidden, "dropout": dropout}

    def forward(self, a: torch.Tensor, q: torch.Tensor):
        g = self.geometry(a)
        r = self.behavior(q)
        return g + r, g, r


class FusionModel(nn.Module):
    def __init__(self, model_id: str, a_dim: int = 32, q_dim: int = 119,
                 dropout: float = 0.1):
        super().__init__()
        if model_id not in ("C64", "Cm"):
            raise ValueError(f"Not a VV fusion model: {model_id}")
        behavior_hidden = 64 if model_id == "C64" else 60
        self.model_id = model_id
        self.geometry = GeometryBranch(a_dim, 64, dropout).encoder
        self.behavior = BehaviorEncoder(q_dim, behavior_hidden, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 2 * behavior_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.spec = {"model_id": model_id, "a_dim": a_dim, "q_dim": q_dim,
                     "geometry_hidden": 64, "behavior_hidden": behavior_hidden, "dropout": dropout}

    def forward(self, a: torch.Tensor, q: torch.Tensor):
        logit = self.fusion(torch.cat([self.geometry(a), self.behavior.encode(q)], -1)).squeeze(-1)
        # Fusion intermediates are not additive evidence and must not be presented as such.
        nan = torch.full_like(logit, float("nan"))
        return logit, nan, nan


def build_model(model_id: str, a_dim: int = 32, q_dim: int = 119,
                dropout: float = 0.1) -> nn.Module:
    if model_id == "G":
        return GeometryModel(a_dim, 64, dropout)
    if model_id == "Q":
        return BehaviorModel(q_dim, 64, dropout)
    if model_id in ADDITIVE_IDS:
        return AdditiveModel(model_id, a_dim, q_dim, 64, dropout)
    if model_id in ("C64", "Cm"):
        return FusionModel(model_id, a_dim, q_dim, dropout)
    raise ValueError(f"Unknown VV model ID: {model_id}")


def configure_training_mode(model: nn.Module, model_id: str) -> None:
    """Apply the registered train/eval and gradient state for one training step."""
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if model_id == "F":
        model.geometry.eval()
        for parameter in model.geometry.parameters():
            parameter.requires_grad_(False)
    elif model_id == "Jw":
        # Dropout is disabled, but gradients must still reach and update G.
        model.geometry.eval()
    elif model_id == "JwD":
        model.geometry.train()


def effective_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


EXPECTED_PARAMETERS = {
    "G": 6337,
    "Q": 35393,
    "C64": 44641,
    "Cm": 41589,
    "Jc": 41730,
    "Jw": 41730,
    "JwD": 41730,
    "F": 41730,
}


def validate_parameter_count(model: nn.Module, model_id: str) -> None:
    observed = effective_parameter_count(model)
    expected = EXPECTED_PARAMETERS[model_id]
    if observed != expected:
        raise AssertionError(f"{model_id}: expected {expected} effective parameters, got {observed}")
