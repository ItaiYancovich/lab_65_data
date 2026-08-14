"""Picklable agent descriptions, so worker processes can build their own.

Agents hold networks, RNGs and search trees, which are awkward to ship between
processes.  A tournament therefore passes around small ``AgentSpec`` records
and each worker instantiates from them, caching anything expensive (the loaded
network) per process.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .agents.base import Agent, RandomAgent
from .agents.az_agent import AlphaZeroAgent, PolicyOnlyAgent
from .agents.mcts_rollout import RolloutMCTSAgent
from .agents.minimax import MinimaxAgent
from .agents.rule_based import RuleBasedAgent
from .evaluator import BatchEvaluator
from .net import load_checkpoint

_EVALUATOR_CACHE: dict[tuple, BatchEvaluator] = {}


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    label: str
    params: tuple = ()  # (key, value) pairs; a tuple so the spec stays hashable

    def as_dict(self) -> dict:
        return dict(self.params)


def spec(kind: str, label: str, **params) -> AgentSpec:
    return AgentSpec(kind=kind, label=label, params=tuple(sorted(params.items())))


def _evaluator(ckpt: str, board_size: int) -> BatchEvaluator:
    key = (ckpt, board_size)
    hit = _EVALUATOR_CACHE.get(key)
    if hit is None:
        net, _ = load_checkpoint(ckpt)
        hit = BatchEvaluator(net, board_size=board_size)
        _EVALUATOR_CACHE[key] = hit
    return hit


def build_agent(s: AgentSpec, seed: int, board_size: int = 11) -> Agent:
    p = s.as_dict()
    if s.kind == "random":
        return RandomAgent(seed=seed, name=s.label)
    if s.kind == "rule":
        return RuleBasedAgent(seed=seed, noise=p.get("noise", 0.05),
                              defence_weight=p.get("defence_weight", 1.0), name=s.label)
    if s.kind == "minimax":
        return MinimaxAgent(time_budget=p.get("time_budget", 1.0), beam=p.get("beam", 8),
                            max_depth=p.get("max_depth", 12), seed=seed, name=s.label)
    if s.kind == "rollout":
        return RolloutMCTSAgent(simulations=p.get("simulations", 5000),
                                time_budget=p.get("time_budget"),
                                c_uct=p.get("c_uct", 1.0), seed=seed, name=s.label)
    if s.kind == "az":
        return AlphaZeroAgent(_evaluator(p["ckpt"], board_size),
                              simulations=p.get("simulations", 400),
                              c_puct=p.get("c_puct", 1.6),
                              batch_size=p.get("batch_size", 16),
                              seed=seed, name=s.label)
    if s.kind == "policy":
        return PolicyOnlyAgent(_evaluator(p["ckpt"], board_size), seed=seed, name=s.label)
    raise ValueError(f"unknown agent kind {s.kind!r}")


def default_field(ckpt: str) -> list[AgentSpec]:
    """The tournament field: the trained agent against each classical family."""
    return [
        spec("random", "random"),
        spec("rule", "rule-based (two-distance+bridges)", noise=0.05),
        spec("minimax", "brute-force alpha-beta (1.0s/move)", time_budget=1.0, beam=8),
        spec("rollout", "classic MCTS rollouts (1.0s/move)",
             simulations=200_000, time_budget=1.0),
        spec("policy", "AlphaZero policy only (no search)", ckpt=ckpt),
        spec("az", "AlphaZero (400 sims)", ckpt=ckpt, simulations=400),
    ]
