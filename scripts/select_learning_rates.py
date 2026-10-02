"""Choose one learning rate per policy class on held-out pilot seeds.

Every policy class gets the same treatment: a small grid, three pilot seeds
(100-102, never used for the reported results), and the same selection
criterion -- the mean on-policy return over the whole training run, i.e. the
area under the learning curve.  The winning values are written to
``results/pilot/selection.json`` and hard-coded in ``qrl/experiments.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from qrl.experiments import load_returns, run_sweep
from qrl.reinforce import TrainConfig

PILOT_SEEDS = [100, 101, 102]
GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
BASES = {
    "quantum": TrainConfig("quantum", "quantum"),
    "mlp_matched": TrainConfig("mlp_matched", "mlp", hidden_sizes=(10,)),
    "mlp_large": TrainConfig("mlp_large", "mlp", hidden_sizes=(64, 64)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/pilot"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    configs = [
        replace(base, name=f"{name}_lr{lr}", learning_rate=lr)
        for name, base in BASES.items()
        for lr in GRID
    ]
    run_sweep(configs, PILOT_SEEDS, args.out, workers=args.workers)

    selection = {}
    for name in BASES:
        scores = {}
        for lr in GRID:
            runs = load_returns(args.out, f"{name}_lr{lr}")
            scores[str(lr)] = float(np.mean([np.mean(r) for r in runs.values()]))
        best = max(scores, key=scores.get)
        selection[name] = {"mean_training_return": scores, "selected": float(best)}
        print(name, scores, "->", best)
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
