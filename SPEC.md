# Queue allocation experiment spec

This project simulates a single-server “video embedding” style pipeline under stochastic arrivals and heterogeneous users, and compares queue allocation mechanisms under different load regimes.

## Model
- **Arrivals**: Poisson process with rate \(\lambda\).
- **Service times**: heavy-tailed; default is **lognormal** (configurable).
- **User heterogeneity**:
  - value \(v\)
  - delay sensitivity \(c\)
  - willingness-to-pay \(w\) for priority, positively correlated with \(c\) by default (toggle for weak correlation).
- **Abandonment**: enabled by default. Each job draws a patience time \(T \sim \text{Exponential}(\text{mean}=\text{patience\_mean})\) and abandons if it waits longer than \(T\).
- **Welfare** (payments excluded): per served job, \(v - c \cdot (\text{completion\_time} - \text{arrival\_time})\). Per-arrival welfare treats abandoned jobs as 0 and reports abandonment separately.

## Mechanisms compared
1. **FCFS**: single queue, first-come-first-served.
2. **Fixed posted-price priority lane**: two FCFS queues (priority/regular). On arrival, join priority if \(w \ge P_\text{fixed}\). Priority served before regular.
3. **Congestion-dependent posted price**: posted price at arrival depends on total waiting jobs \(q\):
   - \(P(q) = P_0 + \alpha \cdot \max(0, q - q_0)\)
   - join priority if \(w \ge P(q)\)
4. **Priority auction benchmark**: order waiting jobs by bid \(b=w\) (highest first). Payment approximation: when a job starts service, charge the next-highest bid among waiting jobs, scaled by expected service time \(E[S]\) (default).

## Experiment design
- Three load regimes targeting utilization \(\rho \in \{0.5, 0.8, 0.95\}\), where \(\rho = \lambda E[S]\).
- For each regime: run \(N\) replications (default \(N=30\)) with deterministic seeds.

## Metrics
Per (regime, mechanism, replication):
- mean waiting time (served jobs)
- mean time-in-system / completion time (served jobs)
- p90 waiting time (served jobs)
- p95 waiting time (served jobs)
- abandonment rate
- welfare per arrival
- weighted delay cost per arrival
- platform revenue per arrival

## Outputs
Written under `./outputs/`:
- `metrics.csv`: one row per replication per mechanism per regime
- `summary_table.csv`: aggregated means + 95% CIs
- `report.md`: narrative writeup + paste-ready paragraph grounded in produced numbers
- `*.png`: plots (CDFs, bars, and tradeoff scatter)

