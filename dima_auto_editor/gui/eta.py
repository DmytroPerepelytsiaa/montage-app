from __future__ import annotations

import time
from typing import Callable


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class EtaEstimator:
    MIN_FRACTION = 0.02
    MIN_ELAPSED_S = 1.0

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._started = clock()

    def reset(self) -> None:
        self._started = self._clock()

    def elapsed(self) -> float:
        return self._clock() - self._started

    def remaining(self, overall_fraction: float) -> float | None:
        fraction = min(1.0, max(0.0, overall_fraction))
        elapsed = self.elapsed()
        if fraction < self.MIN_FRACTION or elapsed < self.MIN_ELAPSED_S:
            return None
        return elapsed * (1.0 - fraction) / fraction
