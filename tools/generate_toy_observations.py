"""Generate a compact three-gene version of the paper's Hill toy system.

Hill function（希尔函数）is used here only in the data-generating process:
the generated observations keep the website's six-feature contract
``[S_A, S_B, S_C, U_A, U_B, U_C]`` so the local RegVelo fitting code can be
reused without changing the teaching-page dimensions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GENES = ["A", "B", "C"]
FEATURE_ORDER = ["S_A", "S_B", "S_C", "U_A", "U_B", "U_C"]


def hill_rates(spliced: np.ndarray, k: np.ndarray, n: np.ndarray,
               h: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Return transcription rates from signed Hill interactions.

    Positive ``k[target, regulator]`` uses the activation probability
    ``s^n / (h^n + s^n)``.  Negative ``k`` uses the complementary repression
    probability ``h^n / (h^n + s^n)``.  This is the same construction as the
    paper toy simulator, with a small three-gene network.
    """
    state = np.maximum(spliced[None, :], 0.0)
    power = np.maximum(n, 1.0)
    x_n = state**power
    h_n = np.maximum(h, 1e-6) ** power
    p_act = x_n / (h_n + x_n)
    p_rep = h_n / (h_n + x_n)
    sign = (k >= 0).astype(np.float64)
    contribution = np.abs(k) * (p_act * sign + p_rep * (1.0 - sign))
    return baseline + contribution.sum(axis=1)


def simulate(seed: int, cells: int, steps: int) -> dict:
    rng = np.random.default_rng(seed)

    # Rows are targets, columns are regulators.  The signed edges mirror the
    # activation/repression convention in the original toy simulation.
    k = np.array([
        [0.0, 0.80, 0.0],
        [0.95, 0.0, -0.65],
        [0.0, 0.75, 0.0],
    ], dtype=np.float64)
    n = np.where(k != 0.0, 2.0, 0.0)
    h = np.where(k != 0.0, 0.70, 1.0)
    beta = np.array([0.72, 0.58, 0.66], dtype=np.float64)
    gamma = np.array([0.34, 0.29, 0.38], dtype=np.float64)
    baseline = np.array([0.18, 0.16, 0.20], dtype=np.float64)

    cells_out = []
    for index in range(cells):
        # Different starting states and observation times create a compact
        # cross-section of one shared dynamical system.
        spliced = np.array([0.20, 0.28, 0.22], dtype=np.float64)
        spliced += rng.normal(0.0, 0.035, size=3)
        spliced = np.maximum(spliced, 0.02)
        unspliced = np.maximum(spliced * 0.55 + rng.normal(0.0, 0.02, size=3), 0.01)
        target_time = 0.35 + 5.4 * index / max(1, cells - 1)
        dt = target_time / max(1, steps)
        for _ in range(max(1, steps)):
            alpha = hill_rates(spliced, k, n, h, baseline)
            du = alpha - beta * unspliced
            ds = beta * unspliced - gamma * spliced
            unspliced = np.maximum(unspliced + dt * du, 0.0)
            spliced = np.maximum(spliced + dt * ds, 0.0)
        observed_s = np.maximum(spliced + rng.normal(0.0, 0.012, size=3), 0.0)
        observed_u = np.maximum(unspliced + rng.normal(0.0, 0.012, size=3), 0.0)
        cells_out.append({
            "id": f"C{index + 1:02d}",
            "group": index % 3,
            "raw": np.concatenate([observed_s, observed_u]).tolist(),
            "true_time": float(target_time),
        })

    return {
        "format": "regvelo-toy-hill-v1",
        "generator": "paper toy Hill dynamics, compact three-gene reduction",
        "seed": seed,
        "featureOrder": FEATURE_ORDER,
        "genes": GENES,
        "cells": cells_out,
        "truth": {
            "K": k.tolist(),
            "n": n.tolist(),
            "h": h.tolist(),
            "beta": beta.tolist(),
            "gamma": gamma.tolist(),
            "alpha_b": baseline.tolist(),
            "networkConvention": "rows=target, columns=regulator",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compact paper-toy Hill observations.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=24)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.cells < 12:
        parser.error("--cells must be at least 12 for the shared train/validation split")
    if args.steps < 1:
        parser.error("--steps must be positive")
    payload = simulate(args.seed, args.cells, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({args.cells} cells, Hill toy dynamics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
