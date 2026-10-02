"""Train every configuration in ``qrl.experiments.MAIN_CONFIGS`` on several seeds.

Example
-------
    python scripts/run_experiments.py --workers 4
    python scripts/run_experiments.py --configs quantum mlp_matched --seeds 0 1 2

Finished runs are skipped, so an interrupted sweep can simply be restarted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qrl.experiments import MAIN_CONFIGS, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=list(MAIN_CONFIGS), choices=MAIN_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--out", type=Path, default=Path("results/main"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    configs = [MAIN_CONFIGS[name] for name in args.configs]
    run_sweep(configs, args.seeds, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
