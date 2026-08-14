#!/usr/bin/env python3
"""Export a trained checkpoint to JSON for the client-side (JS) GUI.

BatchNorm is folded into the preceding convolution analytically:

    y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
      = (gamma / sqrt(var + eps)) * conv(x) + (beta - gamma * mean / sqrt(var + eps))

so a BN layer becomes a per-output-channel scale on the conv weight plus a
bias -- exactly what torch.jit.optimize_for_inference does at trace time, done
here by hand so the browser only ever has to run plain convolutions.
"""

from __future__ import annotations

import argparse
import base64
import json

import numpy as np
import torch

from alphazero_hex.net import HexNet, NetConfig


def pack(t: torch.Tensor) -> dict:
    """A tensor as {shape, data: base64 float32} -- a JS typed-array view away."""
    arr = t.detach().to(torch.float32).contiguous().numpy()
    return {"shape": list(arr.shape), "data": base64.b64encode(arr.tobytes()).decode("ascii")}


def fold_bn(conv_w: torch.Tensor, bn: torch.nn.BatchNorm2d) -> tuple[dict, dict]:
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv_w * scale.view(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * scale
    return pack(w), pack(b)


def export(ckpt_path: str) -> dict:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = HexNet(NetConfig(**blob["cfg"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()

    out: dict = {"cfg": blob["cfg"]}

    stem_conv, stem_bn = net.stem[0], net.stem[1]
    w, b = fold_bn(stem_conv.weight.detach(), stem_bn)
    out["stem"] = {"w": w, "b": b}

    blocks = []
    for block in net.tower:
        w1, b1 = fold_bn(block.conv1.weight.detach(), block.bn1)
        w2, b2 = fold_bn(block.conv2.weight.detach(), block.bn2)
        blocks.append({"w1": w1, "b1": b1, "w2": w2, "b2": b2})
    out["blocks"] = blocks

    ph_conv1, ph_bn, _relu, ph_conv2 = net.policy_head
    w, b = fold_bn(ph_conv1.weight.detach(), ph_bn)
    out["policy_head"] = {
        "w1": w, "b1": b,
        "w2": pack(ph_conv2.weight.detach()),
        "b2": pack(ph_conv2.bias.detach()),
    }

    vc_conv, vc_bn, _relu = net.value_conv
    w, b = fold_bn(vc_conv.weight.detach(), vc_bn)
    out["value_conv"] = {"w": w, "b": b}

    fc1, _relu, fc2 = net.value_fc
    out["value_fc"] = {
        "w1": pack(fc1.weight.detach()), "b1": pack(fc1.bias.detach()),
        "w2": pack(fc2.weight.detach()), "b2": pack(fc2.bias.detach()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/az_hex/final.pt")
    ap.add_argument("--out", default="runs/az_hex/net.json")
    ap.add_argument("--minify", action="store_true", default=True)
    args = ap.parse_args()
    data = export(args.ckpt)
    with open(args.out, "w") as fh:
        if args.minify:
            json.dump(data, fh, separators=(",", ":"))
        else:
            json.dump(data, fh)
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
