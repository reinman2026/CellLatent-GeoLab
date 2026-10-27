"""Generate a six-gene paper-style Hill toy dataset for local RegVelo fits.

The reference simulation uses a Hill response (希尔调控函数) and a noisy
two-pool RNA system.  The local environment has no ``torchsde`` package, so
this script keeps the reference drift and uses an explicit Euler--Maruyama
step.  True K/n/h/beta/gamma values and the support skeleton are saved with
the observations for auditability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".runtime/paper-toy/observations.json"
GENES = [f"G{i}" for i in range(1, 7)]


def draw_poisson_times(n_obs: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    increments = [-0.1 * np.log(rng.uniform(0.0, 1.0)) for _ in range(n_obs - 1)]
    return torch.tensor(np.insert(np.cumsum(increments), 0, 0.0), dtype=torch.float32)


def get_parameters(n_obs: int, seed: int):
    mu = torch.tensor([5.0, 0.5, 0.125], dtype=torch.float32).log()
    corr = torch.tensor(
        [[1.0, 0.2, 0.2], [0.2, 1.0, 0.8], [0.2, 0.8, 1.0]], dtype=torch.float32
    )
    scale = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32)[:, None]
    covariance = scale * scale.T * corr
    torch.manual_seed(seed)
    alpha, beta, gamma = MultivariateNormal(mu, covariance_matrix=covariance).sample((6,)).exp().T
    mean_alpha = alpha.mean()
    # Columns are regulator, target, signed K, Hill coefficient, threshold.
    coef_m = torch.tensor(
        [
            [0, 1, -mean_alpha, 2, 2], [1, 0, -mean_alpha, 2, 2],
            [0, 2, mean_alpha, 2, 4], [0, 3, mean_alpha, 2, 4],
            [2, 3, -mean_alpha, 2, 2], [3, 2, -mean_alpha, 2, 2],
            [1, 4, mean_alpha, 2, 4], [1, 5, mean_alpha, 2, 4],
            [4, 5, -mean_alpha, 2, 2], [5, 4, -mean_alpha, 2, 2],
        ], dtype=torch.float32
    )
    K = torch.zeros((6, 6), dtype=torch.float32)
    hill_n = torch.zeros((6, 6), dtype=torch.float32)
    hill_h = torch.zeros((6, 6), dtype=torch.float32)
    K[coef_m[:, 1].long(), coef_m[:, 0].long()] = coef_m[:, 2]
    hill_n[coef_m[:, 1].long(), coef_m[:, 0].long()] = coef_m[:, 3]
    hill_h[coef_m[:, 1].long(), coef_m[:, 0].long()] = coef_m[:, 4]
    return K, hill_n, hill_h, beta, gamma, draw_poisson_times(n_obs, seed)


def hill_drift(state: torch.Tensor, K: torch.Tensor, hill_n: torch.Tensor,
               hill_h: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    u, s = state[:6], state[6:]
    sign = (K > 0).to(state.dtype)
    s_matrix = torch.clamp(s, min=0).repeat(6, 1)
    x_n = torch.pow(s_matrix, hill_n)
    h_n = torch.pow(hill_h, hill_n)
    denominator = h_n + x_n
    p_act = torch.where(denominator > 0, x_n / denominator, torch.zeros_like(x_n))
    p_rep = torch.where(denominator > 0, h_n / denominator, torch.zeros_like(h_n))
    alpha = (torch.abs(K) * (p_act * sign + p_rep * (1 - sign))).sum(1)
    return torch.cat([torch.clamp(alpha, min=0) - beta * u, beta * u - gamma * s])


def simulate(n_obs: int, seed: int) -> dict:
    K, hill_n, hill_h, beta, gamma, times = get_parameters(n_obs, seed)
    torch.manual_seed(seed + 100_003)
    state = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0] + [0.0] * 6, dtype=torch.float32)
    states = [state.clone()]
    for index in range(1, len(times)):
        dt = torch.clamp(times[index] - times[index - 1], min=1e-6)
        state = torch.clamp(state + hill_drift(state, K, hill_n, hill_h, beta, gamma) * dt
                            + 0.1 * torch.sqrt(dt) * torch.randn_like(state), min=0)
        states.append(state.clone())
    trajectories = torch.stack(states)
    unspliced, spliced = trajectories[:, :6], trajectories[:, 6:]
    return {"times": times, "spliced": spliced, "unspliced": unspliced,
            "velocity": unspliced * beta - spliced * gamma, "K": K,
            "hill_n": hill_n, "hill_h": hill_h, "beta": beta, "gamma": gamma}


def build_observations(n_obs: int, seed: int) -> dict:
    result = simulate(n_obs, seed)
    times = result["times"].tolist()
    spliced, unspliced = result["spliced"].tolist(), result["unspliced"].tolist()
    velocity = result["velocity"].tolist()
    order = [f"S_{gene}" for gene in GENES] + [f"U_{gene}" for gene in GENES]
    bounds = np.quantile(np.asarray(times), [1 / 3, 2 / 3])
    cells = []
    for index, time in enumerate(times):
        group = 0 if time <= bounds[0] else 1 if time <= bounds[1] else 2
        cells.append({"id": f"toy_{index + 1:04d}", "group": group, "time": float(time),
                      "raw": [float(v) for v in spliced[index] + unspliced[index]],
                      "true_velocity": [float(v) for v in velocity[index]]})
    K = result["K"]
    return {
        "format": "regvelo-paper-toy-hill-v1", "genes": GENES, "featureOrder": order,
        "cells": cells,
        "generator": {"seed": seed, "nObs": n_obs,
                       "equation": "Hill drift + Euler-Maruyama diffusion; state=[u_1..u_6,s_1..s_6]",
                       "noiseScale": 0.1,
                       "reference": "ref/regvelo_reproducibility_source/scripts/toy_grn/1_simulation.py",
                       "torchsdeAvailable": False},
        "truth": {"K": K.tolist(), "hillN": result["hill_n"].tolist(),
                  "hillH": result["hill_h"].tolist(), "beta": result["beta"].tolist(),
                  "gamma": result["gamma"].tolist(), "time": times},
        "prior": {"kind": "toy-true-support", "skeleton": (K != 0).to(torch.float32).tolist(),
                  "orientation": "target_by_regulator"},
        "training": {"soft_constraint": False, "lam": 1.0, "lam2": 1.0,
                      "simple_dynamics": True,
                      "description": "paper toy fit settings: hard support, Jacobian L1, simple dynamics"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-obs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.n_obs < 24:
        parser.error("--n-obs must be at least 24")
    data = build_observations(args.n_obs, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "genes": len(data["genes"]),
                      "cells": len(data["cells"]), "seed": args.seed,
                      "priorEdges": int(np.asarray(data["prior"]["skeleton"]).sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
