"""
Queue allocation mechanisms for the single-server video embedding simulation.

Only uses stdlib + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple
from collections import deque
import heapq


@dataclass
class MechanismConfig:
    # Fixed posted-price lane
    p_fixed: float = 1.0

    # Congestion-dependent posted price
    p0: float = 0.5
    alpha: float = 0.2
    q0: int = 5

    # Auction payment scaling
    auction_scale: str = "E_S"  # "E_S" or "service_time"


class Mechanism:
    """
    Mechanism interface.

    The simulator owns the authoritative waiting structures; mechanisms own
    *how* jobs are routed/selected and *how* payments are computed.
    """

    name: str = "base"

    def __init__(self, cfg: MechanismConfig):
        self.cfg = cfg

    def on_arrival(self, job: Dict, now: float, state: Dict) -> None:
        """Route job / set job fields like queue_type and posted_price."""
        raise NotImplementedError

    def enqueue(self, job_id: int, job: Dict, state: Dict) -> None:
        """Add job_id to the mechanism's waiting structure(s)."""
        raise NotImplementedError

    def select_next(self, now: float, state: Dict) -> Optional[int]:
        """Return next job_id to start service, or None."""
        raise NotImplementedError

    def payment_at_service_start(
        self, job_id: int, job: Dict, now: float, state: Dict
    ) -> float:
        """Compute platform revenue collected when job starts service."""
        return 0.0

    def queue_size(self, state: Dict) -> int:
        """Total waiting size (used by congestion pricing)."""
        raise NotImplementedError


class FCFS(Mechanism):
    name = "FCFS"

    def __init__(self, cfg: MechanismConfig):
        super().__init__(cfg)
        self.q: Deque[int] = deque()

    def on_arrival(self, job: Dict, now: float, state: Dict) -> None:
        job["queue_type"] = "single"

    def enqueue(self, job_id: int, job: Dict, state: Dict) -> None:
        self.q.append(job_id)

    def select_next(self, now: float, state: Dict) -> Optional[int]:
        active_waiting = state["active_waiting"]
        while self.q:
            jid = self.q.popleft()
            if jid in active_waiting:
                return jid
        return None

    def queue_size(self, state: Dict) -> int:
        return len(state["active_waiting"])


class FixedPricePriorityLane(Mechanism):
    name = "FixedPricePriorityLane"

    def __init__(self, cfg: MechanismConfig):
        super().__init__(cfg)
        self.q_pri: Deque[int] = deque()
        self.q_reg: Deque[int] = deque()

    def on_arrival(self, job: Dict, now: float, state: Dict) -> None:
        choose_pri = job["w"] >= self.cfg.p_fixed
        job["queue_type"] = "priority" if choose_pri else "regular"
        job["posted_price"] = float(self.cfg.p_fixed) if choose_pri else 0.0

    def enqueue(self, job_id: int, job: Dict, state: Dict) -> None:
        if job["queue_type"] == "priority":
            self.q_pri.append(job_id)
        else:
            self.q_reg.append(job_id)

    def select_next(self, now: float, state: Dict) -> Optional[int]:
        active_waiting = state["active_waiting"]
        for q in (self.q_pri, self.q_reg):
            while q:
                jid = q.popleft()
                if jid in active_waiting:
                    return jid
        return None

    def payment_at_service_start(
        self, job_id: int, job: Dict, now: float, state: Dict
    ) -> float:
        return float(job.get("posted_price", 0.0))

    def queue_size(self, state: Dict) -> int:
        return len(state["active_waiting"])


class CongestionPricePriorityLane(Mechanism):
    name = "CongestionPricePriorityLane"

    def __init__(self, cfg: MechanismConfig):
        super().__init__(cfg)
        self.q_pri: Deque[int] = deque()
        self.q_reg: Deque[int] = deque()

    def _price(self, q_total: int) -> float:
        return float(self.cfg.p0 + self.cfg.alpha * max(0, q_total - self.cfg.q0))

    def on_arrival(self, job: Dict, now: float, state: Dict) -> None:
        q_total = self.queue_size(state)
        p = self._price(q_total)
        choose_pri = job["w"] >= p
        job["queue_type"] = "priority" if choose_pri else "regular"
        job["posted_price"] = p if choose_pri else 0.0
        job["posted_price_faced"] = p

    def enqueue(self, job_id: int, job: Dict, state: Dict) -> None:
        if job["queue_type"] == "priority":
            self.q_pri.append(job_id)
        else:
            self.q_reg.append(job_id)

    def select_next(self, now: float, state: Dict) -> Optional[int]:
        active_waiting = state["active_waiting"]
        for q in (self.q_pri, self.q_reg):
            while q:
                jid = q.popleft()
                if jid in active_waiting:
                    return jid
        return None

    def payment_at_service_start(
        self, job_id: int, job: Dict, now: float, state: Dict
    ) -> float:
        return float(job.get("posted_price", 0.0))

    def queue_size(self, state: Dict) -> int:
        return len(state["active_waiting"])


class PriorityAuction(Mechanism):
    name = "PriorityAuction"

    def __init__(self, cfg: MechanismConfig):
        super().__init__(cfg)
        # max-bid via min-heap on (-bid, seq)
        self.heap: list[Tuple[float, int, int]] = []
        self._seq = 0

    def on_arrival(self, job: Dict, now: float, state: Dict) -> None:
        job["queue_type"] = "auction"
        job["bid"] = float(job["w"])

    def enqueue(self, job_id: int, job: Dict, state: Dict) -> None:
        self._seq += 1
        heapq.heappush(self.heap, (-float(job["bid"]), self._seq, job_id))

    def _peek_two_active(self, state: Dict) -> Tuple[Optional[int], Optional[int]]:
        """
        Return (best_job_id, second_best_job_id) among currently-active waiting jobs,
        without permanently removing them from the heap (uses temporary pops).
        """
        active_waiting = state["active_waiting"]
        popped: list[Tuple[float, int, int]] = []
        best = None
        second = None

        while self.heap and best is None:
            item = heapq.heappop(self.heap)
            bid_neg, seq, jid = item
            if jid in active_waiting:
                best = jid
                popped.append(item)
            # else discard inactive

        while self.heap and second is None:
            item = heapq.heappop(self.heap)
            bid_neg, seq, jid = item
            if jid in active_waiting:
                second = jid
                popped.append(item)
            # else discard inactive

        for item in popped:
            heapq.heappush(self.heap, item)

        return best, second

    def select_next(self, now: float, state: Dict) -> Optional[int]:
        active_waiting = state["active_waiting"]
        while self.heap:
            bid_neg, seq, jid = heapq.heappop(self.heap)
            if jid in active_waiting:
                return jid
        return None

    def payment_at_service_start(
        self, job_id: int, job: Dict, now: float, state: Dict
    ) -> float:
        # Payment approx: next-highest bid among waiting jobs at service start
        # (excluding the job being served). If none, 0.
        active_waiting = state["active_waiting"]
        if job_id in active_waiting:
            # Simulator should have removed it already, but be robust.
            active_waiting = set(active_waiting)
            active_waiting.discard(job_id)

        # Find the highest active bid in the remaining heap (lazy)
        popped: list[Tuple[float, int, int]] = []
        second_bid = 0.0
        while self.heap:
            item = heapq.heappop(self.heap)
            bid_neg, seq, jid = item
            if jid in active_waiting:
                second_bid = -bid_neg
                popped.append(item)
                break
            # else discard inactive
        for item in popped:
            heapq.heappush(self.heap, item)

        if second_bid <= 0:
            return 0.0

        if self.cfg.auction_scale == "service_time":
            scale = float(job["service_time"])
        else:
            scale = float(state["E_S"])
        return float(second_bid * scale)

    def queue_size(self, state: Dict) -> int:
        return len(state["active_waiting"])


def make_mechanisms(cfg: MechanismConfig) -> Dict[str, Mechanism]:
    return {
        FCFS.name: FCFS(cfg),
        FixedPricePriorityLane.name: FixedPricePriorityLane(cfg),
        CongestionPricePriorityLane.name: CongestionPricePriorityLane(cfg),
        PriorityAuction.name: PriorityAuction(cfg),
    }

