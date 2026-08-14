"""Size-agnostic policy/value residual network for Hex.

The network is fully convolutional and the value head pools globally, so the
*same weights* work on any board size.  That is what makes the size curriculum
(train on 5x5, transfer to 7x7, 9x9, then 11x11) possible: convolutional Hex
patterns -- bridges, ladders, edge templates -- are local and transfer almost
verbatim to bigger boards.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_PLANES = 3


@dataclass
class NetConfig:
    channels: int = 64
    blocks: int = 6
    head_channels: int = 32
    value_hidden: int = 64

    def to_dict(self) -> dict:
        return asdict(self)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class HexNet(nn.Module):
    """Outputs (policy logits over n*n cells, value in [-1, 1])."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        c = self.cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(IN_PLANES, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(c) for _ in range(self.cfg.blocks)])

        h = self.cfg.head_channels
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, h, 1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, 1, 1),  # 1 logit per cell -> board-size agnostic
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, h, 1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(2 * h, self.cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.value_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        x = self.tower(self.stem(x))
        policy = self.policy_head(x).reshape(b, -1)
        v = self.value_conv(x)
        # global average + global max pooling keeps the head size-independent
        v = torch.cat([v.mean(dim=(2, 3)), v.amax(dim=(2, 3))], dim=1)
        value = torch.tanh(self.value_fc(v)).squeeze(1)
        return policy, value


def save_checkpoint(path, net: HexNet, extra: dict | None = None) -> None:
    torch.save(
        {"cfg": net.cfg.to_dict(), "state_dict": net.state_dict(), "extra": extra or {}},
        path,
    )


def load_checkpoint(path, map_location="cpu") -> tuple[HexNet, dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    net = HexNet(NetConfig(**blob["cfg"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob.get("extra", {})


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
