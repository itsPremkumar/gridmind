"""Grid management core."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GridNode:
    """A node in the power grid."""
    node_id: str
    name: str
    capacity_mw: float
    current_load_mw: float = 0.0
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        if self.capacity_mw == 0:
            return 0.0
        return self.current_load_mw / self.capacity_mw


@dataclass
class GridEdge:
    """A connection between two grid nodes."""
    edge_id: str
    source_id: str
    target_id: str
    capacity_mw: float
    current_flow_mw: float = 0.0
    status: str = "active"


class GridManager:
    """Manage power grid topology and state."""

    def __init__(self):
        self.nodes: dict[str, GridNode] = {}
        self.edges: dict[str, GridEdge] = {}

    def add_node(self, name: str, capacity_mw: float, **kwargs) -> GridNode:
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        node = GridNode(node_id=node_id, name=name, capacity_mw=capacity_mw, **kwargs)
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, capacity_mw: float) -> GridEdge | None:
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        edge = GridEdge(edge_id=edge_id, source_id=source_id, target_id=target_id, capacity_mw=capacity_mw)
        self.edges[edge_id] = edge
        return edge

    def get_node(self, node_id: str) -> GridNode | None:
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> GridEdge | None:
        return self.edges.get(edge_id)

    def get_neighbors(self, node_id: str) -> list[GridNode]:
        neighbors = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                target = self.nodes.get(edge.target_id)
                if target:
                    neighbors.append(target)
            elif edge.target_id == node_id:
                source = self.nodes.get(edge.source_id)
                if source:
                    neighbors.append(source)
        return neighbors

    def total_capacity(self) -> float:
        return sum(n.capacity_mw for n in self.nodes.values())

    def total_load(self) -> float:
        return sum(n.current_load_mw for n in self.nodes.values())

    def get_overloaded_nodes(self, threshold: float = 0.8) -> list[GridNode]:
        return [n for n in self.nodes.values() if n.utilization >= threshold]

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_capacity_mw": self.total_capacity(),
            "total_load_mw": self.total_load(),
            "overloaded_count": len(self.get_overloaded_nodes()),
        }
