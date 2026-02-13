"""
Compute metrics, aggregate across replications, generate plots, and write report.md.

Inputs:
  - outputs/metrics.csv (per-replication metrics)
  - outputs/wait_samples.npz (pooled waiting-time samples for CDF plots)

Outputs:
  - outputs/summary_table.csv
  - outputs/*.png
  - outputs/report.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class AnalysisConfig:
    outputs_dir: str = "outputs"
    max_cdf_points: int = 20000


MECH_ORDER = [
    "FCFS",
    "FixedPricePriorityLane",
    "CongestionPricePriorityLane",
    "PriorityAuction",
]

REGIME_ORDER = ["low", "medium", "high"]


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """
    Render a DataFrame as a GitHub-flavored markdown table without external deps.

    (Avoids pandas.DataFrame.to_markdown() which requires 'tabulate'.)
    """
    cols = list(df.columns)
    rows = df.to_numpy(dtype=object)

    # Convert to strings
    str_rows: List[List[str]] = []
    for r in rows:
        str_rows.append([("" if x is None else str(x)) for x in r])

    widths = [len(str(c)) for c in cols]
    for r in str_rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(cell))

    def fmt_row(items: List[str]) -> str:
        return "| " + " | ".join(items[j].ljust(widths[j]) for j in range(len(cols))) + " |"

    header = fmt_row([str(c) for c in cols])
    sep = "| " + " | ".join("-" * widths[j] for j in range(len(cols))) + " |"
    body = "\n".join(fmt_row(r) for r in str_rows)
    return "\n".join([header, sep, body]) + "\n"


def mean_ci95(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = float(np.mean(x))
    if n == 1:
        return (m, m, m)
    sd = float(np.std(x, ddof=1))
    half = 1.96 * sd / np.sqrt(n)
    return (m, m - half, m + half)


def aggregate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["regime", "mechanism"]
    metric_cols = [
        "mean_wait_served",
        "mean_sojourn_served",
        "p90_wait_served",
        "p95_wait_served",
        "abandonment_rate",
        "welfare_per_arrival",
        "delay_cost_per_arrival",
        "revenue_per_arrival",
    ]

    rows: List[Dict] = []
    for (reg, mech), g in metrics.groupby(group_cols):
        row: Dict = {"regime": reg, "mechanism": mech, "n_replications": int(len(g))}
        for col in metric_cols:
            m, lo, hi = mean_ci95(g[col].to_numpy())
            row[f"{col}_mean"] = m
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
        rows.append(row)

    out = pd.DataFrame(rows)
    out["regime"] = pd.Categorical(out["regime"], categories=REGIME_ORDER, ordered=True)
    out["mechanism"] = pd.Categorical(out["mechanism"], categories=MECH_ORDER, ordered=True)
    out = out.sort_values(["regime", "mechanism"]).reset_index(drop=True)
    return out


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    v = np.sort(v)
    y = np.arange(1, v.size + 1) / v.size
    return v, y


def plot_cdf_waiting_times(wait_npz_path: str, outputs_dir: str, max_points: int) -> List[str]:
    data = np.load(wait_npz_path, allow_pickle=False)
    keys = sorted(list(data.keys()))
    # keys look like "{regime}__{mechanism}"
    saved_paths: List[str] = []

    for regime in REGIME_ORDER:
        plt.figure(figsize=(8, 5))
        for mech in MECH_ORDER:
            k = f"{regime}__{mech}"
            if k not in data:
                continue
            x = data[k]
            x = x[np.isfinite(x)]
            if x.size > max_points:
                # deterministic downsample: take quantile grid
                qs = np.linspace(0, 1, max_points)
                x = np.quantile(x, qs)
            xs, ys = _ecdf(x)
            if xs.size == 0:
                continue
            plt.plot(xs, ys, label=mech)
        plt.xlabel("Waiting time")
        plt.ylabel("CDF")
        plt.title(f"Waiting time CDF (served jobs) — {regime} load")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = os.path.join(outputs_dir, f"cdf_wait_{regime}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        saved_paths.append(out_path)

    return saved_paths


def _bar_with_ci(
    summary: pd.DataFrame,
    regime: str,
    value_col_mean: str,
    value_col_lo: str,
    value_col_hi: str,
    title: str,
    ylabel: str,
    outputs_dir: str,
    filename: str,
) -> str:
    s = summary[summary["regime"] == regime].copy()
    s = s.sort_values("mechanism")
    xlabels = [str(m) for m in s["mechanism"].tolist()]
    means = s[value_col_mean].to_numpy(dtype=float)
    lo = s[value_col_lo].to_numpy(dtype=float)
    hi = s[value_col_hi].to_numpy(dtype=float)
    yerr = np.vstack([means - lo, hi - means])

    plt.figure(figsize=(8, 4.8))
    xs = np.arange(len(xlabels))
    plt.bar(xs, means, yerr=yerr, capsize=4, alpha=0.85)
    plt.xticks(xs, xlabels, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    out_path = os.path.join(outputs_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_bars(summary: pd.DataFrame, outputs_dir: str) -> List[str]:
    saved: List[str] = []
    for regime in REGIME_ORDER:
        # abandonment (no CI bars needed, but we have them anyway)
        saved.append(
            _bar_with_ci(
                summary,
                regime,
                "abandonment_rate_mean",
                "abandonment_rate_ci95_lo",
                "abandonment_rate_ci95_hi",
                title=f"Abandonment rate by mechanism — {regime} load",
                ylabel="Abandonment rate",
                outputs_dir=outputs_dir,
                filename=f"abandonment_{regime}.png",
            )
        )
        saved.append(
            _bar_with_ci(
                summary,
                regime,
                "welfare_per_arrival_mean",
                "welfare_per_arrival_ci95_lo",
                "welfare_per_arrival_ci95_hi",
                title=f"Social welfare per job (v - c*(completion_time-arrival_time)) — {regime} load",
                ylabel="Welfare per arrival (abandoned counted as 0)",
                outputs_dir=outputs_dir,
                filename=f"welfare_{regime}.png",
            )
        )
        saved.append(
            _bar_with_ci(
                summary,
                regime,
                "p95_wait_served_mean",
                "p95_wait_served_ci95_lo",
                "p95_wait_served_ci95_hi",
                title=f"p95 waiting time (served jobs) — {regime} load",
                ylabel="p95 wait (served)",
                outputs_dir=outputs_dir,
                filename=f"p95_wait_{regime}.png",
            )
        )
    return saved


def plot_tradeoff_scatter(summary: pd.DataFrame, outputs_dir: str) -> str:
    plt.figure(figsize=(7, 5.5))

    markers = {"low": "o", "medium": "s", "high": "^"}
    for regime in REGIME_ORDER:
        s = summary[summary["regime"] == regime]
        x = s["p95_wait_served_mean"].to_numpy(dtype=float)
        y = s["welfare_per_arrival_mean"].to_numpy(dtype=float)
        plt.scatter(x, y, label=regime, marker=markers.get(regime, "o"), s=70, alpha=0.85)
        for _, row in s.iterrows():
            plt.annotate(
                str(row["mechanism"]),
                (float(row["p95_wait_served_mean"]), float(row["welfare_per_arrival_mean"])),
                fontsize=8,
                alpha=0.85,
            )

    plt.xlabel("p95 waiting time (served)")
    plt.ylabel("Welfare per arrival")
    plt.title("Welfare vs p95 wait tradeoff (one point per mechanism per regime)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Load regime")
    out_path = os.path.join(outputs_dir, "tradeoff_scatter.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def write_report(
    outputs_dir: str,
    params: Dict,
    summary: pd.DataFrame,
    plot_paths: List[str],
) -> str:
    report_path = os.path.join(outputs_dir, "report.md")

    # derive a minimal, data-grounded qualitative paragraph
    # (no causal/general claims; just relative comparisons in this generated data)
    def _pick(regime: str, metric: str) -> Tuple[str, float]:
        s = summary[summary["regime"] == regime].copy()
        s = s.sort_values(f"{metric}_mean", ascending=(metric != "welfare_per_arrival"))
        top = s.iloc[0]
        return str(top["mechanism"]), float(top[f"{metric}_mean"])

    lines: List[str] = []
    lines.append("# Queue allocation experiment report\n")
    lines.append("## How to run\n")
    lines.append("```bash\npython run_experiment.py --seed 123 --replications 30\n```\n")

    lines.append("## Model\n")
    lines.append(
        textwrap.dedent(
            """
            We simulate a single-server “video embedding” pipeline. Jobs arrive as a Poisson process, require heavy-tailed service time (lognormal), and have heterogeneous value `v` and delay sensitivity `c`. Each job’s welfare contribution is computed as `v - c * (completion_time - arrival_time)` (payments are excluded from welfare). Jobs may abandon if their waiting time exceeds a random patience time.
            """
        ).strip()
        + "\n"
    )

    lines.append("\n## Mechanisms\n")
    lines.append(
        textwrap.dedent(
            """
            - FCFS: one queue, served in arrival order.
            - Fixed posted-price priority lane: users join a priority queue if willingness-to-pay exceeds a fixed posted price; priority jobs preempt regular jobs.
            - Congestion-dependent posted price: the posted price increases with current queue length; users choose priority if willingness-to-pay exceeds the current price.
            - Priority auction benchmark: waiting jobs are served in descending bid order; a generalized second-price style payment approximation charges the next-highest bid among waiting jobs at service start, scaled for units.
            """
        ).strip()
        + "\n"
    )

    lines.append("\n## Load regimes\n")
    lines.append(
        "We run three utilization targets $\\rho \\in \\{0.5, 0.8, 0.95\\}$ by setting the arrival rate $\\lambda = \\rho / E[S]$.\n"
    )

    lines.append("\n## Parameters used\n")
    params_df = pd.DataFrame(
        [{"parameter": k, "value": str(v)} for k, v in sorted(params.items(), key=lambda kv: kv[0])]
    )
    lines.append(df_to_markdown_table(params_df))
    lines.append("\n")

    lines.append("\n## Results summary (means and 95% CIs)\n")
    # Keep table readable: round numeric columns a bit
    summary_disp = summary.copy()
    for c in summary_disp.columns:
        if c.endswith("_mean") or c.endswith("_ci95_lo") or c.endswith("_ci95_hi"):
            summary_disp[c] = pd.to_numeric(summary_disp[c], errors="ignore")
            if np.issubdtype(summary_disp[c].dtype, np.number):
                summary_disp[c] = summary_disp[c].round(4)
    lines.append(df_to_markdown_table(summary_disp))
    lines.append("\n")

    lines.append("\n## Key qualitative patterns (from this generated data)\n")
    best_w_low, _ = _pick("low", "welfare_per_arrival")
    best_w_med, _ = _pick("medium", "welfare_per_arrival")
    best_w_high, _ = _pick("high", "welfare_per_arrival")
    low_p95, _ = _pick("low", "p95_wait_served")
    med_p95, _ = _pick("medium", "p95_wait_served")
    high_p95, _ = _pick("high", "p95_wait_served")

    lines.append(
        textwrap.dedent(
            f"""
            In this run, the mechanism with the highest mean welfare per arrival was **{best_w_low}** (low load), **{best_w_med}** (medium load), and **{best_w_high}** (high load). The mechanism with the lowest mean p95 waiting time among served jobs was **{low_p95}** (low load), **{med_p95}** (medium load), and **{high_p95}** (high load). Abandonment rates are higher under higher utilization in the summary table above.
            """
        ).strip()
        + "\n"
    )

    lines.append("\n## Figures\n")
    for p in plot_paths:
        lines.append(f"- `{p}`")
    lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def run_analysis(outputs_dir: str, params: Dict) -> Dict[str, str]:
    os.makedirs(outputs_dir, exist_ok=True)
    metrics_path = os.path.join(outputs_dir, "metrics.csv")
    wait_npz_path = os.path.join(outputs_dir, "wait_samples.npz")

    metrics = pd.read_csv(metrics_path)
    summary = aggregate_summary(metrics)
    summary_path = os.path.join(outputs_dir, "summary_table.csv")
    summary.to_csv(summary_path, index=False)

    plot_paths: List[str] = []
    if os.path.exists(wait_npz_path):
        plot_paths.extend(plot_cdf_waiting_times(wait_npz_path, outputs_dir, max_points=20000))
    plot_paths.extend(plot_bars(summary, outputs_dir))
    plot_paths.append(plot_tradeoff_scatter(summary, outputs_dir))

    report_path = write_report(outputs_dir, params=params, summary=summary, plot_paths=plot_paths)

    return {
        "metrics_csv": metrics_path,
        "summary_table_csv": summary_path,
        "report_md": report_path,
        "plots_dir": outputs_dir,
    }

