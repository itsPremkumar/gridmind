"""Grid optimizer using AI-driven load balancing."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from .grid import GridManager, GridNode


@dataclass
class OptimizationResult:
    """Result of a grid optimization pass."""
    iterations: int
    initial_cost: float
    final_cost: float
    load_transfers: list[dict[str, Any]]
    converged: bool


class GridOptimizer:
    """AI-powered grid load balancing optimizer."""

    def __init__(self, grid: GridManager):
        self.grid = grid
        self.learning_rate = 0.01
        self.momentum = 0.9

    def compute_cost(self) -> float:
        """Compute total grid cost (sum of squared utilization deviations)."""
        total = 0.0
        for node in self.grid.nodes.values():
            if node.capacity_mw > 0:
                utilization = node.utilization
                # Penalize both under-utilization and over-utilization
                total += (utilization - 0.6) ** 2
        return total

    def suggest_load_transfer(self, overloaded: GridNode, target: GridNode) -> float | None:
        """Suggest a load transfer from overloaded to target node."""
        if overloaded.utilization < 0.8:
            return None
        if target.utilization > 0.7:
            return None

        excess = overloaded.current_load_mw - (overloaded.capacity_mw * 0.6)
        available = (target.capacity_mw * 0.7) - target.current_load_mw
        transfer = min(excess, available)
        return max(transfer, 0.0)

    def optimize(self, max_iterations: int = 100, tolerance: float = 1e-4) -> OptimizationResult:
        """Run optimization to balance grid load."""
        initial_cost = self.compute_cost()
        transfers = []

        for i in range(max_iterations):
            overloaded = self.grid.get_overloaded_nodes(threshold=0.8)
            if not overloaded:
                break

            for node in overloaded:
                neighbors = self.grid.get_neighbors(node.node_id)
                for neighbor in neighbors:
                    transfer = self.suggest_load_transfer(node, neighbor)
                    if transfer and transfer > 0:
                        node.current_load_mw -= transfer
                        neighbor.current_load_mw += transfer
                        transfers.append({
                            "from": node.node_id,
                            "to": neighbor.node_id,
                            "amount_mw": transfer,
                        })

            current_cost = self.compute_cost()
            if abs(current_cost - initial_cost) < tolerance:
                break

        final_cost = self.compute_cost()
        return OptimizationResult(
            iterations=i + 1,
            initial_cost=initial_cost,
            final_cost=final_cost,
            load_transfers=transfers,
            converged=final_cost < initial_cost,
        )

    def predict_optimal_capacity(self, node_id: str, horizon_hours: int = 24) -> float:
        """Predict optimal capacity for a node based on historical patterns."""
        node = self.grid.get_node(node_id)
        if not node:
            return 0.0
        # Simple prediction: current load * safety margin
        return node.current_load_mw * 1.2 if node.current_load_mw > 0 else node.capacity_mw * 0.5
