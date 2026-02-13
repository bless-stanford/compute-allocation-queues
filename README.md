# CS360 — queue allocation simulation (ECON 284 paper)

This repo contains a small, reproducible discrete-event simulation of a single-server “video embedding” pipeline with stochastic arrivals, heavy-tailed service times, heterogeneous users, and optional abandonment.

## How to run

1. Create an environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the experiment (writes everything to `./outputs/`):

```bash
MPLBACKEND=Agg python run_experiment.py --seed 123 --replications 30
```

## Outputs
- `outputs/metrics.csv`: one row per (regime, mechanism, replication)
- `outputs/summary_table.csv`: aggregated means + 95% CIs
- `outputs/report.md`: narrative summary (includes paste-ready paragraph)
- `outputs/*.png`: plots

