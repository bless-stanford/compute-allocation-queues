"""
Discrete-event simulation of a single-server video embedding pipeline.

Model:
- Arrivals: Poisson process with rate lambda
- Service times: lognormal (configurable via mu/sigma of underlying normal)
- Users/jobs: heterogeneous (v, c, w), with abandonment (patience) enabled by default

Only uses stdlib + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq
import math

import numpy as np

from mechanisms import Mechanism, MechanismConfig, make_mechanisms


@dataclass
class SimConfig:
    # Workload / horizon
    n_arrivals: int = 6000
    warmup_frac: float = 0.2  # used by analysis; simulator records arrival_idx

    # Service time lognormal parameters (underlying normal)
    service_mu: float = -0.2
    service_sigma: float = 0.9

    # Abandonment: patience ~ Exp(mean=patience_mean). If None, no abandonment.
    patience_mean: Optional[float] = 20.0

    # User primitives
    v_mu: float = 1.2
    v_sigma: float = 0.6
    c_mu: float = -0.2
    c_sigma: float = 0.7

    # WTP correlation
    w_corr: str = "strong"  # "strong" or "weak"
    w_k: float = 1.0
    w_noise_sigma: float = 0.6

    # Mechanism config
    mech: MechanismConfig = MechanismConfig()


class Simulator:
    """
    A single replication simulator.

    Usage:
        sim = Simulator(cfg, rng, mechanism)
        jobs, revenue = sim.run(lambda_rate)
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator, mechanism: Mechanism):
        self.cfg = cfg
        self.rng = rng
        self.mech = mechanism

        # event heap: (time, seq, type, job_id)
        self._events: List[Tuple[float, int, str, int]] = []
        self._seq = 0

        self.jobs: Dict[int, Dict] = {}
        self.active_waiting: set[int] = set()  # waiting + not abandoned + not started

        self.server_busy: bool = False
        self.now: float = 0.0
        self.revenue_total: float = 0.0

        self.E_S: float = float(self.expected_service_time())

    def expected_service_time(self) -> float:
        mu = self.cfg.service_mu
        s = self.cfg.service_sigma
        return math.exp(mu + 0.5 * s * s)

    def _push_event(self, t: float, typ: str, job_id: int) -> None:
        self._seq += 1
        heapq.heappush(self._events, (float(t), self._seq, typ, int(job_id)))

    def _pop_event(self) -> Tuple[float, int, str, int]:
        return heapq.heappop(self._events)

    def _draw_service_time(self) -> float:
        return float(self.rng.lognormal(mean=self.cfg.service_mu, sigma=self.cfg.service_sigma))

    def _draw_patience(self) -> Optional[float]:
        if self.cfg.patience_mean is None:
            return None
        mean = float(self.cfg.patience_mean)
        if mean <= 0:
            return 0.0
        # exponential with mean
        return float(self.rng.exponential(scale=mean))

    def _draw_user_primitives(self) -> Tuple[float, float, float]:
        v = float(self.rng.lognormal(mean=self.cfg.v_mu, sigma=self.cfg.v_sigma))
        c = float(self.rng.lognormal(mean=self.cfg.c_mu, sigma=self.cfg.c_sigma))

        if self.cfg.w_corr == "weak":
            k = 0.3 * self.cfg.w_k
            noise_sigma = 1.2 * self.cfg.w_noise_sigma
        else:
            k = self.cfg.w_k
            noise_sigma = self.cfg.w_noise_sigma

        noise = float(abs(self.rng.normal(loc=0.0, scale=noise_sigma)))
        w = float(max(0.0, k * c + noise))
        return v, c, w

    def _maybe_start_service(self) -> None:
        if self.server_busy:
            return
        if not self.active_waiting:
            return

        state = {"active_waiting": self.active_waiting, "E_S": self.E_S}
        jid = self.mech.select_next(self.now, state)
        if jid is None:
            return

        # remove from waiting set
        self.active_waiting.discard(jid)

        job = self.jobs[jid]
        job["start_service_time"] = self.now
        job["waiting_time"] = float(self.now - job["arrival_time"])

        pay = float(self.mech.payment_at_service_start(jid, job, self.now, state))
        job["paid"] = pay
        self.revenue_total += pay

        self.server_busy = True
        self._push_event(self.now + float(job["service_time"]), "SERVICE_END", jid)

    def run(self, lambda_rate: float) -> Tuple[List[Dict], float]:
        """
        Run the replication until all generated jobs are either completed or abandoned.
        Returns (job_records, revenue_total).
        """
        lam = float(lambda_rate)
        if lam <= 0:
            raise ValueError("lambda_rate must be > 0")

        # Generate arrivals as a Poisson process (exponential interarrivals)
        t = 0.0
        for i in range(self.cfg.n_arrivals):
            dt = float(self.rng.exponential(scale=1.0 / lam))
            t += dt
            jid = i

            service_time = self._draw_service_time()
            patience = self._draw_patience()
            v, c, w = self._draw_user_primitives()

            job = {
                "job_id": jid,
                "arrival_idx": i,
                "arrival_time": t,
                "service_time": service_time,
                "patience_time": patience,
                "v": v,
                "c": c,
                "w": w,
                "bid": w,
                "queue_type": None,
                "posted_price": 0.0,
                "posted_price_faced": None,
                "start_service_time": None,
                "completion_time": None,
                "abandon_time": None,
                "paid": 0.0,
                "waiting_time": None,
            }
            self.jobs[jid] = job

            # schedule arrival event; job fields get processed at arrival
            self._push_event(t, "ARRIVAL", jid)

            # schedule abandonment event if enabled
            if patience is not None:
                self._push_event(t + patience, "ABANDON", jid)

        completed_or_abandoned = 0

        while self._events:
            ev_t, ev_seq, ev_typ, jid = self._pop_event()
            self.now = ev_t
            job = self.jobs[jid]

            if ev_typ == "ARRIVAL":
                # Route + enqueue in mechanism
                state = {"active_waiting": self.active_waiting, "E_S": self.E_S}
                self.mech.on_arrival(job, self.now, state)
                self.active_waiting.add(jid)
                self.mech.enqueue(jid, job, state)
                self._maybe_start_service()

            elif ev_typ == "ABANDON":
                # If job hasn't started service or finished, it abandons.
                if job["start_service_time"] is None and job["completion_time"] is None and job["abandon_time"] is None:
                    if jid in self.active_waiting:
                        self.active_waiting.discard(jid)
                    job["abandon_time"] = self.now
                    completed_or_abandoned += 1

            elif ev_typ == "SERVICE_END":
                if job["completion_time"] is not None:
                    continue  # defensive
                job["completion_time"] = self.now
                self.server_busy = False
                completed_or_abandoned += 1
                self._maybe_start_service()

            else:
                raise RuntimeError(f"unknown event type {ev_typ}")

            if completed_or_abandoned >= self.cfg.n_arrivals:
                break

        return list(self.jobs.values()), float(self.revenue_total)


def expected_service_time_lognormal(mu: float, sigma: float) -> float:
    """Analytic E[S] for lognormal with underlying N(mu, sigma^2)."""
    return float(math.exp(mu + 0.5 * sigma * sigma))


def build_mechanism(mechanism_name: str, cfg: SimConfig) -> Mechanism:
    mechs = make_mechanisms(cfg.mech)
    if mechanism_name not in mechs:
        raise ValueError(f"Unknown mechanism {mechanism_name}. Options: {sorted(mechs.keys())}")
    return mechs[mechanism_name]

