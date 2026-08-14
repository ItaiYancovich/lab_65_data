"""Batched network evaluation helpers."""

from __future__ import annotations

import numpy as np
import torch

from .hex_game import HexBoard
from .net import HexNet


class BatchEvaluator:
    """Runs the net on a list of positions and returns (priors, values).

    Priors come back in the *canonical* frame (the frame the network sees), and
    are already masked to legal cells and renormalised.
    """

    def __init__(self, net: HexNet, device: str = "cpu", jit: bool = True, board_size: int = 11):
        self.net = net.to(device).eval()
        self.device = device
        self.module = self.net
        if jit and device == "cpu":
            # Folding BatchNorm into the convolutions roughly halves inference
            # cost, which is the whole self-play budget on a CPU.  Traced
            # convnets stay shape-polymorphic, so one trace serves every board
            # size and batch size in the curriculum.
            try:
                example = torch.zeros(8, 3, board_size, board_size)
                traced = torch.jit.trace(self.net, example)
                self.module = torch.jit.optimize_for_inference(traced)
            except Exception:
                self.module = self.net  # fall back to eager rather than fail

    @torch.inference_mode()
    def evaluate(self, states: list[HexBoard]) -> tuple[np.ndarray, np.ndarray]:
        planes = np.stack([s.canonical_planes() for s in states])
        x = torch.from_numpy(planes).to(self.device)
        logits, values = self.module(x)

        # Legality mask, expressed in the canonical frame.
        n = states[0].n
        mask = np.zeros((len(states), n * n), dtype=bool)
        for i, s in enumerate(states):
            occ = s.array()
            occ = occ if s.to_move == 1 else occ.T
            mask[i] = (occ == 0).reshape(-1)
        logits = logits.masked_fill(torch.from_numpy(~mask).to(self.device), -1e9)
        priors = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        return priors, values.cpu().numpy().astype(np.float32)
