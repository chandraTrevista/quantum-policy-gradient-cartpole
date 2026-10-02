"""How many measurement shots does a trained quantum policy need per decision?

Loads the checkpoints of the full quantum policy and re-evaluates each one
with the circuit expectation estimated from a finite number of shots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qrl.evaluate import evaluate_quantum_policy
from qrl.policies import QuantumPolicy

SHOT_BUDGETS: list[int | None] = [1, 4, 16, 64, 256, 1024, None]  # None = exact
EVAL_SEED_OFFSET = 10_000  # evaluation environments never share a seed with training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("results/main"))
    parser.add_argument("--config", default="quantum")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results/shots.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    results: dict[str, dict[str, list[float]]] = {}
    for checkpoint in sorted(args.runs.glob(f"{args.config}_seed*.pt")):
        run = json.loads(checkpoint.with_suffix(".json").read_text())["config"]
        policy = QuantumPolicy(n_layers=run["n_layers"])
        policy.load_state_dict(torch.load(checkpoint))
        per_budget = {}
        for shots in SHOT_BUDGETS:
            returns = evaluate_quantum_policy(
                policy, args.episodes, seed=EVAL_SEED_OFFSET + run["seed"], shots=shots
            )
            per_budget["exact" if shots is None else str(shots)] = returns.tolist()
            print(f"seed {run['seed']} shots {shots}: mean return {returns.mean():.1f}", flush=True)
        results[str(run["seed"])] = per_budget
    args.out.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
