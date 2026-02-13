"""
Entry point: runs the simulation across load regimes and mechanisms, writes outputs.

Writes:
  - outputs/metrics.csv
  - outputs/wait_samples.npz
  - outputs/summary_table.csv
  - outputs/report.md
  - outputs/*.png
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from simulator import SimConfig, Simulator, build_mechanism, expected_service_time_lognormal
from analysis import run_analysis


REGIMES = [
    ("low", 0.5),
    ("medium", 0.8),
    ("high", 0.95),
]

MECHANISMS = [
    "FCFS",
    "FixedPricePriorityLane",
    "CongestionPricePriorityLane",
    "PriorityAuction",
]


def compute_replication_metrics(
    jobs: List[Dict],
    revenue_total: float,
    warmup_frac: float,
) -> Dict[str, float]:
    n = len(jobs)
    warm_cut = int(np.floor(warmup_frac * n))
    js = [j for j in jobs if int(j["arrival_idx"]) >= warm_cut]
    if not js:
        raise RuntimeError("warmup cut removed all jobs; reduce warmup_frac or increase n_arrivals")

    served = [j for j in js if j["completion_time"] is not None]
    abandoned = [j for j in js if j["abandon_time"] is not None and j["completion_time"] is None]

    abandonment_rate = float(len(abandoned) / len(js))

    def _arr(x):
        return np.asarray(x, dtype=float)

    if served:
        waits = _arr([float(j["start_service_time"]) - float(j["arrival_time"]) for j in served])
        sojourn = _arr([float(j["completion_time"]) - float(j["arrival_time"]) for j in served])
        mean_wait = float(np.mean(waits))
        mean_sojourn = float(np.mean(sojourn))
        p90 = float(np.quantile(waits, 0.90))
        p95 = float(np.quantile(waits, 0.95))

        welfare_served = _arr(
            [
                float(j["v"]) - float(j["c"]) * (float(j["completion_time"]) - float(j["arrival_time"]))
                for j in served
            ]
        )
        delay_cost_served = _arr(
            [float(j["c"]) * (float(j["completion_time"]) - float(j["arrival_time"])) for j in served]
        )
        welfare_per_served = float(np.mean(welfare_served))
        delay_cost_per_served = float(np.mean(delay_cost_served))
    else:
        waits = _arr([])
        mean_wait = np.nan
        mean_sojourn = np.nan
        p90 = np.nan
        p95 = np.nan
        welfare_per_served = np.nan
        delay_cost_per_served = np.nan

    # Primary welfare/delay-cost per arrival: abandoned jobs contribute 0 by default
    welfare_total = 0.0
    delay_cost_total = 0.0
    for j in js:
        if j["completion_time"] is None:
            continue
        flow = float(j["completion_time"]) - float(j["arrival_time"])
        welfare_total += float(j["v"]) - float(j["c"]) * flow
        delay_cost_total += float(j["c"]) * flow
    welfare_per_arrival = float(welfare_total / len(js))
    delay_cost_per_arrival = float(delay_cost_total / len(js))

    revenue_per_arrival = float(revenue_total / len(js))

    return {
        "n_jobs_eval": float(len(js)),
        "served_rate": float(len(served) / len(js)),
        "abandonment_rate": abandonment_rate,
        "mean_wait_served": float(mean_wait),
        "mean_sojourn_served": float(mean_sojourn),
        "p90_wait_served": float(p90),
        "p95_wait_served": float(p95),
        "welfare_per_arrival": welfare_per_arrival,
        "delay_cost_per_arrival": delay_cost_per_arrival,
        "welfare_per_served": float(welfare_per_served),
        "delay_cost_per_served": float(delay_cost_per_served),
        "revenue_per_arrival": revenue_per_arrival,
    }


def downsample_concat(existing: np.ndarray, new: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if existing.size == 0:
        combined = new
    else:
        combined = np.concatenate([existing, new])
    if combined.size <= cap:
        return combined
    idx = rng.choice(combined.size, size=cap, replace=False)
    return combined[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--replications", type=int, default=30)
    ap.add_argument("--outputs-dir", type=str, default="outputs")
    ap.add_argument("--n-arrivals", type=int, default=6000)
    ap.add_argument("--warmup-frac", type=float, default=0.2)
    ap.add_argument("--patience-mean", type=float, default=20.0)
    ap.add_argument("--wtp-corr", type=str, default="strong", choices=["strong", "weak"])

    # Mechanism params
    ap.add_argument("--p-fixed", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--q0", type=int, default=5)
    ap.add_argument("--auction-scale", type=str, default="E_S", choices=["E_S", "service_time"])

    args = ap.parse_args()

    os.makedirs(args.outputs_dir, exist_ok=True)

    base_cfg = SimConfig(
        n_arrivals=int(args.n_arrivals),
        warmup_frac=float(args.warmup_frac),
        patience_mean=float(args.patience_mean),
        w_corr=str(args.wtp_corr),
    )
    base_cfg.mech.p_fixed = float(args.p_fixed)
    base_cfg.mech.p0 = float(args.p0)
    base_cfg.mech.alpha = float(args.alpha)
    base_cfg.mech.q0 = int(args.q0)
    base_cfg.mech.auction_scale = str(args.auction_scale)

    E_S = expected_service_time_lognormal(base_cfg.service_mu, base_cfg.service_sigma)

    rows: List[Dict] = []
    wait_samples: Dict[str, np.ndarray] = {f"{r}__{m}": np.array([], dtype=float) for r, _ in REGIMES for m in MECHANISMS}

    # Dedicated RNG for downsampling only (keeps sim RNGs independent)
    sampler_rng = np.random.default_rng(int(args.seed) + 99991)

    for regime_name, rho in REGIMES:
        lam = float(rho / E_S)
        for mech_name in MECHANISMS:
            for rep in range(int(args.replications)):
                rep_seed = int(args.seed) + 100000 * (REGIMES.index((regime_name, rho)) + 1) + 1000 * (MECHANISMS.index(mech_name) + 1) + rep
                rng = np.random.default_rng(rep_seed)

                # fresh mechanism per replication (so internal queues/heaps don't carry over)
                mech = build_mechanism(mech_name, base_cfg)
                sim = Simulator(base_cfg, rng=rng, mechanism=mech)
                jobs, revenue_total = sim.run(lambda_rate=lam)

                metrics = compute_replication_metrics(jobs, revenue_total, warmup_frac=base_cfg.warmup_frac)

                row = {
                    "regime": regime_name,
                    "rho_target": float(rho),
                    "lambda": float(lam),
                    "mechanism": mech_name,
                    "replication": int(rep),
                    "seed": int(rep_seed),
                    "E_service_time": float(sim.E_S),
                    "n_arrivals": int(base_cfg.n_arrivals),
                    "warmup_frac": float(base_cfg.warmup_frac),
                    "patience_mean": float(base_cfg.patience_mean) if base_cfg.patience_mean is not None else np.nan,
                    "wtp_corr": str(base_cfg.w_corr),
                    "p_fixed": float(base_cfg.mech.p_fixed),
                    "p0": float(base_cfg.mech.p0),
                    "alpha": float(base_cfg.mech.alpha),
                    "q0": int(base_cfg.mech.q0),
                    "auction_scale": str(base_cfg.mech.auction_scale),
                }
                row.update(metrics)
                rows.append(row)

                # waiting-time samples for CDF plots (served jobs after warmup)
                warm_cut = int(np.floor(base_cfg.warmup_frac * len(jobs)))
                served = [
                    j
                    for j in jobs
                    if int(j["arrival_idx"]) >= warm_cut and j["completion_time"] is not None
                ]
                waits = np.asarray(
                    [float(j["start_service_time"]) - float(j["arrival_time"]) for j in served],
                    dtype=float,
                )
                key = f"{regime_name}__{mech_name}"
                wait_samples[key] = downsample_concat(wait_samples[key], waits, cap=20000, rng=sampler_rng)

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(args.outputs_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    wait_npz_path = os.path.join(args.outputs_dir, "wait_samples.npz")
    np.savez_compressed(wait_npz_path, **wait_samples)

    params_for_report = {
        "seed": int(args.seed),
        "replications": int(args.replications),
        "n_arrivals": int(base_cfg.n_arrivals),
        "warmup_frac": float(base_cfg.warmup_frac),
        "service_mu": float(base_cfg.service_mu),
        "service_sigma": float(base_cfg.service_sigma),
        "patience_mean": float(base_cfg.patience_mean) if base_cfg.patience_mean is not None else None,
        "v_mu": float(base_cfg.v_mu),
        "v_sigma": float(base_cfg.v_sigma),
        "c_mu": float(base_cfg.c_mu),
        "c_sigma": float(base_cfg.c_sigma),
        "wtp_corr": str(base_cfg.w_corr),
        "p_fixed": float(base_cfg.mech.p_fixed),
        "p0": float(base_cfg.mech.p0),
        "alpha": float(base_cfg.mech.alpha),
        "q0": int(base_cfg.mech.q0),
        "auction_scale": str(base_cfg.mech.auction_scale),
        "E_service_time": float(E_S),
        "regimes": str(REGIMES),
        "mechanisms": str(MECHANISMS),
    }

    out = run_analysis(args.outputs_dir, params=params_for_report)
    print("Wrote outputs:")
    for k, v in out.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

