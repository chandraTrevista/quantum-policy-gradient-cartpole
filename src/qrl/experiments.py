"""Named experiment configurations and a resumable multi-seed runner."""

from __future__ import annotations

import json
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import torch

from qrl.reinforce import TrainConfig, train

# Learning rates below were selected on pilot seeds (100-102) that are disjoint
# from the evaluation seeds (0-9); see ``scripts/select_learning_rates.py`` and
# ``results/pilot/selection.json``.
QUANTUM_LR = 0.03
MLP_SMALL_LR = 0.03
MLP_LARGE_LR = 0.003

MAIN_CONFIGS: dict[str, TrainConfig] = {
    # The full quantum policy and its ablations.
    "quantum": TrainConfig("quantum", "quantum", learning_rate=QUANTUM_LR),
    "quantum_fixed_input_scaling": TrainConfig(
        "quantum_fixed_input_scaling",
        "quantum",
        learning_rate=QUANTUM_LR,
        trainable_input_scaling=False,
    ),
    "quantum_fixed_output_scale": TrainConfig(
        "quantum_fixed_output_scale",
        "quantum",
        learning_rate=QUANTUM_LR,
        trainable_output_scale=False,
    ),
    "quantum_1_layer": TrainConfig(
        "quantum_1_layer", "quantum", learning_rate=QUANTUM_LR, n_layers=1
    ),
    "quantum_3_layers": TrainConfig(
        "quantum_3_layers", "quantum", learning_rate=QUANTUM_LR, n_layers=3
    ),
    # Classical baselines: one with a matched parameter budget, one much larger.
    "mlp_matched": TrainConfig(
        "mlp_matched", "mlp", learning_rate=MLP_SMALL_LR, hidden_sizes=(10,)
    ),
    "mlp_large": TrainConfig("mlp_large", "mlp", learning_rate=MLP_LARGE_LR, hidden_sizes=(64, 64)),
}


def result_path(directory: Path, config: TrainConfig) -> Path:
    return directory / f"{config.name}_seed{config.seed}.json"


def _run_one(job: tuple[TrainConfig, str]) -> str:
    config, directory = job
    torch.set_num_threads(1)  # one core per run; parallelism is across runs
    result = train(config)
    path = result_path(Path(directory), config)
    path.write_text(json.dumps(result.to_json()))
    torch.save(result.state_dict, path.with_suffix(".pt"))
    return f"{path.name}: {result.wall_time_s:.0f}s"


def run_sweep(
    configs: list[TrainConfig], seeds: list[int], directory: Path, workers: int = 1
) -> None:
    """Train every configuration on every seed, skipping finished runs."""
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in seeds:  # seed-major order: partial sweeps stay balanced
        for config in configs:
            seeded = replace(config, seed=seed)
            if not result_path(directory, seeded).exists():
                jobs.append((seeded, str(directory)))
    print(f"{len(jobs)} runs to do", flush=True)
    if workers == 1:
        for job in jobs:
            print(_run_one(job), flush=True)
        return
    with get_context("spawn").Pool(workers) as pool:
        for message in pool.imap_unordered(_run_one, jobs):
            print(message, flush=True)


def load_returns(directory: Path, name: str) -> dict[int, list[list[float]]]:
    """Map ``seed -> episode_returns`` for every finished run of ``name``."""
    runs = {}
    for path in sorted(directory.glob(f"{name}_seed*.json")):
        payload = json.loads(path.read_text())
        if payload["config"]["name"] == name:
            runs[payload["config"]["seed"]] = payload["episode_returns"]
    return runs
