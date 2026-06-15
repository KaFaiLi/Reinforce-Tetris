"""Value network scoring Tetris afterstates."""

from __future__ import annotations

import torch
import torch.nn as nn

from .env import FEATURE_DIM


class ValueNet(nn.Module):
    """MLP mapping afterstate features to a scalar value estimate."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def save_checkpoint(path, model: ValueNet, **extra) -> None:
    torch.save({"model_state": model.state_dict(), **extra}, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[ValueNet, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = ValueNet()
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
