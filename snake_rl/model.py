"""Q-network for Snake."""

from __future__ import annotations

import torch
import torch.nn as nn

from snake_rl.env import NUM_ACTIONS, STATE_DIM


class QNet(nn.Module):
    """MLP mapping the 11-feature state to Q-values for the 3 actions."""

    def __init__(self, input_dim: int = STATE_DIM, hidden: tuple[int, ...] = (128, 128),
                 num_actions: int = NUM_ACTIONS):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_checkpoint(path, model: QNet, **extra) -> None:
    torch.save({"model_state": model.state_dict(), **extra}, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[QNet, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = QNet()
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
