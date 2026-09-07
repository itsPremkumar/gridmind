"""Anomaly detector for grid metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class AnomalyEvent:
    """A detected anomaly."""
    metric: str
    value: float
    expected_range: tuple[float, float]
    severity: str  # low | medium | high
    timestamp: float

    @property
    def is_significant(self) -> bool:
        return self.severity in ("medium", "high")


class AnomalyDetector:
    """Detect anomalies in grid metrics using z-score analysis."""

    def __init__(self, threshold: float = 2.0, window_size: int = 50):
        self.threshold = threshold
        self.window_size = window_size
        self.history: dict[str, list[float]] = {}
        self.events: list[AnomalyEvent] = []

    def update(self, metric: str, value: float, timestamp: float = 0.0) -> AnomalyEvent | None:
        """Update detector with a new metric value. Returns anomaly if detected."""
        if metric not in self.history:
            self.history[metric] = []

        history = self.history[metric]
        history.append(value)

        # Keep only recent history
        if len(history) > self.window_size:
            history.pop(0)

        # Need minimum data
        if len(history) < 5:
            return None

        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / (len(history) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev == 0:
            return None

        z_score = abs(value - mean) / std_dev
        if z_score > self.threshold:
            severity = "high" if z_score > self.threshold * 2 else "medium"
            event = AnomalyEvent(
                metric=metric,
                value=value,
                expected_range=(mean - self.threshold * std_dev, mean + self.threshold * std_dev),
                severity=severity,
                timestamp=timestamp,
            )
            self.events.append(event)
            return event

        return None

    def record(self, metric: str, value: float) -> AnomalyEvent | None:
        """Record a metric value and check for anomalies."""
        if metric not in self.history:
            self.history[metric] = []
        self.history[metric].append(value)
        if len(self.history[metric]) > self.window_size:
            self.history[metric].pop(0)
        # Check for anomaly
        return self.check(metric, value)

    def check(self, metric: str, value: float) -> AnomalyEvent | None:
        """Check if a value is an anomaly (without recording it)."""
        if metric not in self.history:
            return None
        history = self.history[metric]
        if len(history) < 5:
            return None
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / (len(history) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        if std_dev == 0:
            # All history values are identical - flag only significant deviations
            if abs(value - mean) > mean * 0.5:
                event = AnomalyEvent(
                    metric=metric,
                    value=value,
                    expected_range=(mean, mean),
                    severity="high",
                    timestamp=0.0,
                )
                self.events.append(event)
                return event
            return None
        z_score = abs(value - mean) / std_dev
        if z_score > self.threshold:
            severity = "high" if z_score > self.threshold * 2 else "medium"
            event = AnomalyEvent(
                metric=metric,
                value=value,
                expected_range=(mean - self.threshold * std_dev, mean + self.threshold * std_dev),
                severity=severity,
                timestamp=0.0,
            )
            self.events.append(event)
            return event
        return None

    def get_history(self, metric: str) -> list[float]:
        """Get history for a metric."""
        return self.history.get(metric, [])

    def get_all_anomalies(self, metric: str | None = None) -> list[AnomalyEvent]:
        """Get all anomaly events."""
        results = self.events
        if metric:
            results = [e for e in results if e.metric == metric]
        return results

    def get_events(self, metric: str | None = None, severity: str | None = None) -> list[AnomalyEvent]:
        """Get anomaly events with optional filtering."""
        results = self.events
        if metric:
            results = [e for e in results if e.metric == metric]
        if severity:
            results = [e for e in results if e.severity == severity]
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "metrics_tracked": len(self.history),
            "total_events": len(self.events),
            "high_severity": len([e for e in self.events if e.severity == "high"]),
            "medium_severity": len([e for e in self.events if e.severity == "medium"]),
        }
