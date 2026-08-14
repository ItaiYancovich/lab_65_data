#!/usr/bin/env python3
"""Dump network outputs on random positions, for the JS port to match."""
import json
import sys

sys.path.insert(0, "/home/user/lab_65_data")
import numpy as np
import torch

from alphazero_hex.hex_game import HexBoard
from alphazero_hex.net import load_checkpoint

net, _ = load_checkpoint("/home/user/lab_65_data/runs/az_hex/final.pt")
net.eval()

rng = np.random.default_rng(123)
cases = []
for n in (5, 7, 11):
    for trial in range(4):
        b = HexBoard(n)
        nmoves = int(rng.integers(0, n * n - 1))
        for _ in range(nmoves):
            if b.is_terminal():
                break
            legal = b.legal_moves()
            b.play(int(rng.choice(legal)))
        planes = b.canonical_planes()
        with torch.inference_mode():
            logits, value = net(torch.from_numpy(planes).unsqueeze(0))
        cases.append({
            "n": n,
            "planes": planes.flatten().tolist(),
            "logits": logits[0].tolist(),
            "value": float(value[0]),
        })

with open("/home/user/lab_65_data/web/parity_cases.json", "w") as fh:
    json.dump(cases, fh)
print(f"wrote {len(cases)} cases")
