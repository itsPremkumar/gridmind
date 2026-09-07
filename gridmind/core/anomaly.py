"""Anomaly detection for grid monitoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class AnomalyEvent:
    """A detected anomaly."""
    node_id: str
    timestamp: float
    expected_load: float
    actual_load: float
    deviation: float
    severity: str  # low, medium, high, critical


class AnomalyDetector:
    """Detect anomalies in grid load patterns."""

    def __init__(self, threshold_std: float = 2.0):
        self._history: dict[str, list[float]] = {}
        self._threshold_std = threshold_std

    def record(self, node_id: str, load_mw: float) -> None:
        """Record a load observation."""
        if node_id not in self._history:
            self._history[node_id] = []
        self._history[node_id].append(load_mw)

    def check(self, node_id: str, load_mw: float) -> AnomalyEvent | None:
        """Check if a load reading is anomalous."""
        history = self._history.get(node_id, [])
        if len(history) < 5:
            return None

        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        deviation = abs(load_mw - mean) / std_dev
        if deviation <= self._threshold_std:
            return None

        severity = "low"
        if deviation > 4.0:
            severity = "critical"
        elif deviation > 3.0:
            severity = "high"
        elif deviation > 2.5:
            severity = "medium"

        return AnomalyEvent(
            node_id=node_id,
            timestamp=0.0,
            expected_load=mean,
            actual_load=load_mw,
            deviation=deviation,
            severity=severity,
        )

    def get_history(self, node_id: str) -> list[float]:
        """Get history for a node."""
        return self._history.get(node_id, [])

    def get_all_anomalies(self) -> list[AnomalyEvent]:
        """Get all detected anomalies."""
        anomalies = []
        for node_id in self._history:
            history = self._history[node_id]
            if len(history) > 5:
                latest = history[-1]
                event = self.check(node_id, latest)
                if event:
                    anomalies.append(event)
        return anomalies
