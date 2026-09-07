"""Load predictor using time-series analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class Prediction:
    """A load prediction for a future time step."""
    timestamp: float
    predicted_load_mw: float
    confidence: float
    lower_bound: float
    upper_bound: float


class LoadPredictor:
    """Predict future grid load using exponential smoothing."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha  # Level smoothing
        self.beta = beta    # Trend smoothing
        self.level = 0.0
        self.trend = 0.0
        self.history: list[float] = []
        self._initialized = False

    def update(self, load_mw: float) -> None:
        """Update predictor with a new observation."""
        self.history.append(load_mw)
        if not self._initialized:
            self.level = load_mw
            self.trend = 0.0
            self._initialized = True
        else:
            prev_level = self.level
            self.level = self.alpha * load_mw + (1 - self.alpha) * (prev_level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend

    def predict(self, steps_ahead: int = 1) -> Prediction:
        """Predict load for a future time step."""
        if not self._initialized:
            return Prediction(timestamp=0, predicted_load_mw=0, confidence=0, lower_bound=0, upper_bound=0)

        forecast = self.level + steps_ahead * self.trend
        # Confidence decreases with forecast horizon
        confidence = max(0.1, 0.95 - (steps_ahead * 0.05))
        # Uncertainty grows with horizon
        std_dev = self._compute_std_dev() * math.sqrt(steps_ahead)
        return Prediction(
            timestamp=steps_ahead,
            predicted_load_mw=forecast,
            confidence=confidence,
            lower_bound=max(0, forecast - 1.96 * std_dev),
            upper_bound=forecast + 1.96 * std_dev,
        )

    def predict_series(self, n_steps: int) -> list[Prediction]:
        """Predict a series of future loads."""
        return [self.predict(steps_ahead=i + 1) for i in range(n_steps)]

    def _compute_std_dev(self) -> float:
        """Compute standard deviation of historical residuals."""
        if len(self.history) < 2:
            return 0.0
        mean = sum(self.history) / len(self.history)
        variance = sum((x - mean) ** 2 for x in self.history) / (len(self.history) - 1)
        return math.sqrt(variance)

    def detect_trend(self) -> str:
        """Detect overall trend direction."""
        if not self._initialized:
            return "unknown"
        if self.trend > 0.01:
            return "increasing"
        elif self.trend < -0.01:
            return "decreasing"
        return "stable"
