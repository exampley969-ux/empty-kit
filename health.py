"""
Vision Radar - Monitoring: component health
Explicit, visible failure states - "Nothing should silently fail".

Spec reference: PDF Section 26 (Error handling: CAMERA OFFLINE, VISION
ENGINE ERROR, DATA STORAGE UNAVAILABLE, TRACKING DEGRADED), Section 17-A
(system status panel), Section 32 ("stale data isn't presented as live").
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

COMPONENTS = ("camera", "vision", "tracking", "events", "database")


class ComponentStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"      # never reported yet


# The exact strings Section 26 specifies for each failure.
SPEC_MESSAGES = {
    ("camera", ComponentStatus.ERROR): "CAMERA OFFLINE",
    ("vision", ComponentStatus.ERROR): "VISION ENGINE ERROR",
    ("database", ComponentStatus.ERROR): "DATA STORAGE UNAVAILABLE",
    ("tracking", ComponentStatus.DEGRADED): "TRACKING DEGRADED",
}


@dataclass
class ComponentHealth:
    component: str
    status: ComponentStatus
    message: Optional[str]
    detail: Optional[str]
    updated_at: Optional[float]
    stale: bool


class HealthRegistry:
    def __init__(self, stale_after_seconds: float = 10.0, clock: Callable[[], float] = time.time):
        self.stale_after_seconds = stale_after_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._state: Dict[str, tuple] = {}   # component -> (status, detail, updated_at)

    def set(self, component: str, status: ComponentStatus, detail: Optional[str] = None) -> None:
        if component not in COMPONENTS:
            raise ValueError(f"Unknown component '{component}' (expected one of {COMPONENTS})")
        with self._lock:
            self._state[component] = (status, detail, self._clock())

    def get(self, component: str) -> ComponentHealth:
        with self._lock:
            entry = self._state.get(component)
        if entry is None:
            return ComponentHealth(component, ComponentStatus.UNKNOWN, None, None, None, stale=True)
        status, detail, updated_at = entry
        stale = (self._clock() - updated_at) > self.stale_after_seconds
        return ComponentHealth(
            component=component, status=status,
            message=SPEC_MESSAGES.get((component, status)),
            detail=detail, updated_at=updated_at, stale=stale,
        )

    def snapshot(self) -> Dict[str, ComponentHealth]:
        return {c: self.get(c) for c in COMPONENTS}

    def overall(self) -> ComponentStatus:
        """ERROR if any component errored, else DEGRADED if any degraded or
        stale/unknown, else OK. A component that stopped reporting counts as
        not-OK: silence must never read as health."""
        statuses = []
        for h in self.snapshot().values():
            if h.status in (ComponentStatus.ERROR, ComponentStatus.DEGRADED):
                statuses.append(h.status)
            elif h.stale or h.status == ComponentStatus.UNKNOWN:
                statuses.append(ComponentStatus.DEGRADED)
            else:
                statuses.append(ComponentStatus.OK)
        if ComponentStatus.ERROR in statuses:
            return ComponentStatus.ERROR
        if ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED
        return ComponentStatus.OK
