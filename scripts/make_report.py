"""Turn the raw run files into the summary table and figures used in the README.

Reads ``results/main/*.json`` (and ``results/shots.json`` if present) and
writes ``results/summary.json``, ``results/summary.md`` and ``figures/*.png``.
Nothing here trains a model, so it is safe to re-run at any time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qrl import stats
from qrl.experiments import MAIN_CONFIGS, load_returns

# Colour follows the entity: the full quantum policy is blue in every figure.
# Each three-colour set was checked for colour-vision-deficiency separation.
STYLE = {
    "quantum": ("Quantum circuit (69 parameters)", "#2a78d6"),
    "mlp_matched": ("MLP 1x10 (72 parameters)", "#eb6834"),
    "mlp_large": ("MLP 2x64 (4,610 parameters)", "#1baf7a"),
    "quantum_fixed_input_scaling": ("Input scaling frozen at 1", "#e34948"),
    "quantum_fixed_output_scale": ("Output scale frozen at 1", "#4a3aa7"),
    "quantum_3_layers": ("3 layers", "#e87ba4"),
    "quantum_1_layer": ("1 layer", "#008300"),
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def iqm_over_seeds(curves: np.ndarray) -> np.ndarray:
    """IQM along axis 0 of a ``(n_seeds, ...)`` array."""
    ordered = np.sort(curves, axis=0)
    cut = curves.shape[0] // 4
    return ordered[cut : curves.shape[0] - cut].mean(axis=0)


def curve_with_band(curves: np.ndarray, n_resamples: int = 2000, seed: int = 0):
    """Pointwise IQM and 95% bootstrap band for ``(n_seeds, n_updates)`` curves."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(curves), size=(n_resamples, len(curves)))
    resampled = np.stack([iqm_over_seeds(curves[row]) for row in indices])
    low, high = np.quantile(resampled, [0.025, 0.975], axis=0)
    return iqm_over_seeds(curves), low, high


def random_policy_return(n_episodes: int = 1000, seed: int = 0) -> float:
    """Mean return of uniformly random actions, as a floor for the plots."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    total = 0.0
    for _ in range(n_episodes):
        env.reset()
        done = False
        while not done:
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(2)))
            total += reward
            done = terminated or truncated
    env.close()
    return total / n_episodes


def summarise(runs: dict[int, list[list[float]]]) -> dict:
    """Across-seed summary of one configuration."""
    seeds = sorted(runs)
    final = np.array([stats.final_return(runs[s]) for s in seeds])
    area = np.array([stats.mean_training_return(runs[s]) for s in seeds])
    solve = np.array([stats.episodes_to_solve(runs[s]) for s in seeds])
    solved = solve[~np.isnan(solve)]
    return {
        "n_seeds": len(seeds),
        "final_return_per_seed": final.tolist(),
        "final_return_iqm": stats.interquartile_mean(final),
        "final_return_ci": stats.bootstrap_ci(final),
        "final_return_mean": float(final.mean()),
        "final_return_std": float(final.std(ddof=1)),
        "training_return_iqm": stats.interquartile_mean(area),
        "training_return_ci": stats.bootstrap_ci(area),
        "n_solved": int(len(solved)),
        "median_episodes_to_solve": float(np.median(solved)) if len(solved) else None,
    }


def plot_curves(ax, runs_by_name: dict, names: list[str], floor: float, episodes_per_update: int):
    for name in names:
        label, colour = STYLE[name]
        curves = np.stack([stats.learning_curve(r) for r in runs_by_name[name].values()])
        centre, low, high = curve_with_band(curves)
        episodes = (np.arange(curves.shape[1]) + 1) * episodes_per_update
        ax.fill_between(episodes, low, high, color=colour, alpha=0.15, linewidth=0)
        ax.plot(episodes, centre, color=colour, linewidth=2, label=label, solid_capstyle="round")
    ax.axhline(floor, color=INK_SECONDARY, linewidth=1)
    ax.annotate(
        "random actions",
        (ax.get_xlim()[1], floor),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax.set_ylim(0, 520)
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Episode return")


def title_and_legend(ax, title: str, labels: list[str] | None = None) -> None:
    """Put the legend in a row above the plot area, under the title."""
    handles, default_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels or default_labels,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.0),
        ncol=len(handles),
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.4,
        borderaxespad=0.2,
    )
    ax.set_title(title, pad=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("results/main"))
    parser.add_argument("--shots", type=Path, default=Path("results/shots.json"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.figures.mkdir(exist_ok=True)

    runs_by_name = {name: load_returns(args.runs, name) for name in MAIN_CONFIGS}
    runs_by_name = {name: runs for name, runs in runs_by_name.items() if runs}
    batch = MAIN_CONFIGS["quantum"].episodes_per_update
    floor = random_policy_return()

    # ---- summary table ----------------------------------------------------
    summary = {name: summarise(runs) for name, runs in runs_by_name.items()}
    for name in summary:
        first = json.loads(next(args.runs.glob(f"{name}_seed*.json")).read_text())
        summary[name]["n_parameters"] = first["n_parameters"]
    summary["random_policy_return"] = floor

    # Quantum minus matched MLP: bootstrap interval for the difference in IQM.
    if {"quantum", "mlp_matched"} <= runs_by_name.keys():
        summary["quantum_minus_mlp_matched"] = {}
        for label, metric in [
            ("final_return", stats.final_return),
            ("training_return", stats.mean_training_return),
        ]:
            quantum = np.array([metric(r) for r in runs_by_name["quantum"].values()])
            classical = np.array([metric(r) for r in runs_by_name["mlp_matched"].values()])
            rng = np.random.default_rng(0)
            differences = [
                stats.interquartile_mean(rng.choice(quantum, len(quantum)))
                - stats.interquartile_mean(rng.choice(classical, len(classical)))
                for _ in range(10_000)
            ]
            summary["quantum_minus_mlp_matched"][label] = {
                "iqm_difference": stats.interquartile_mean(quantum)
                - stats.interquartile_mean(classical),
                "ci": [float(q) for q in np.quantile(differences, [0.025, 0.975])],
            }

    rows = [
        "| Policy | Parameters | Final return, IQM [95% CI] | Training return, IQM [95% CI] "
        "| Runs solved | Median episodes to solve |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in runs_by_name:
        s = summary[name]
        median = s["median_episodes_to_solve"]
        to_solve = "-" if median is None else f"{median:.0f}"
        final_low, final_high = s["final_return_ci"]
        area_low, area_high = s["training_return_ci"]
        rows.append(
            f"| `{name}` | {s['n_parameters']:,} "
            f"| {s['final_return_iqm']:.0f} [{final_low:.0f}, {final_high:.0f}] "
            f"| {s['training_return_iqm']:.0f} [{area_low:.0f}, {area_high:.0f}] "
            f"| {s['n_solved']}/{s['n_seeds']} | {to_solve} |"
        )

    # ---- figure 1: quantum vs classical -------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    plot_curves(ax, runs_by_name, ["quantum", "mlp_matched", "mlp_large"], floor, batch)
    title_and_legend(ax, "Quantum circuit policy vs. classical networks on CartPole-v1")
    fig.tight_layout()
    fig.savefig(args.figures / "learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: ablations ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    plot_curves(
        axes[0],
        runs_by_name,
        ["quantum", "quantum_fixed_output_scale", "quantum_fixed_input_scaling"],
        floor,
        batch,
    )
    title_and_legend(
        axes[0],
        "Which trainable scalings matter?",
        ["Full policy", "Output scale frozen at 1", "Input scaling frozen at 1"],
    )
    plot_curves(
        axes[1], runs_by_name, ["quantum", "quantum_3_layers", "quantum_1_layer"], floor, batch
    )
    axes[1].set_ylabel("")
    title_and_legend(
        axes[1], "How many re-uploading layers?", ["5 layers (full policy)", "3 layers", "1 layer"]
    )
    fig.tight_layout()
    fig.savefig(args.figures / "ablations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 3: finite-shot evaluation ------------------------------------
    if args.shots.exists():
        shots = json.loads(args.shots.read_text())
        budgets = list(next(iter(shots.values())))
        per_seed = np.array([[np.mean(run[b]) for b in budgets] for run in shots.values()])
        centre = [stats.interquartile_mean(per_seed[:, i]) for i in range(len(budgets))]
        bands = [stats.bootstrap_ci(per_seed[:, i]) for i in range(len(budgets))]
        summary["shots"] = {
            b: {"iqm": centre[i], "ci": bands[i], "per_seed": per_seed[:, i].tolist()}
            for i, b in enumerate(budgets)
        }
        rows += ["", "| Shots per decision | Evaluation return, IQM [95% CI] |", "|---:|---:|"]
        rows += [
            f"| {b} | {centre[i]:.0f} [{bands[i][0]:.0f}, {bands[i][1]:.0f}] |"
            for i, b in enumerate(budgets)
        ]

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        colour = STYLE["quantum"][1]
        x = np.arange(len(budgets))
        lower = [c - b[0] for c, b in zip(centre, bands, strict=True)]
        upper = [b[1] - c for c, b in zip(centre, bands, strict=True)]
        ax.errorbar(
            x,
            centre,
            yerr=[lower, upper],
            color=colour,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            capsize=3,
        )
        ax.axhline(floor, color=INK_SECONDARY, linewidth=1)
        ax.annotate(
            "random actions",
            (x[-1], floor),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
        ax.set_xticks(x, budgets)
        ax.set_ylim(0, 520)
        ax.set_xlabel("Measurement shots per decision")
        ax.set_ylabel("Evaluation return")
        ax.set_title("Trained quantum policy evaluated with finite measurement shots")
        fig.tight_layout()
        fig.savefig(args.figures / "shot_noise.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    Path("results/summary.json").write_text(json.dumps(summary, indent=2))
    Path("results/summary.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nrandom policy: {floor:.1f}")


if __name__ == "__main__":
    main()
