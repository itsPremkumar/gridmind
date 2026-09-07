"""GridMind REST API server."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gridmind.core.grid import GridManager, GridNode, GridEdge
from gridmind.core.optimizer import GridOptimizer
from gridmind.core.predictor import LoadPredictor
from gridmind.core.anomaly import AnomalyDetector

app = FastAPI(title="GridMind API", version="1.0.0")

# Global state
grid = GridManager()
optimizer = GridOptimizer(grid)
detector = AnomalyDetector()


class NodeInput(BaseModel):
    name: str
    capacity_mw: float
    current_load_mw: float = 0.0


class EdgeInput(BaseModel):
    source_id: str
    target_id: str
    capacity_mw: float


@app.get("/")
async def root():
    return {"message": "GridMind AI Platform API", "version": "1.0.0"}


@app.get("/stats")
async def stats():
    return grid.get_stats()


@app.post("/nodes")
async def add_node(input_data: NodeInput):
    node = grid.add_node(input_data.name, input_data.capacity_mw, current_load_mw=input_data.current_load_mw)
    return {"node_id": node.node_id, "name": node.name}


@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    node = grid.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return {
        "node_id": node.node_id,
        "name": node.name,
        "capacity_mw": node.capacity_mw,
        "current_load_mw": node.current_load_mw,
        "utilization": node.utilization,
    }


@app.post("/edges")
async def add_edge(input_data: EdgeInput):
    edge = grid.add_edge(input_data.source_id, input_data.target_id, input_data.capacity_mw)
    if not edge:
        raise HTTPException(status_code=400, detail="Invalid node IDs")
    return {"edge_id": edge.edge_id}


@app.post("/optimize")
async def optimize():
    result = optimizer.optimize()
    return {
        "iterations": result.iterations,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "converged": result.converged,
        "transfers": len(result.load_transfers),
    }


@app.get("/overloaded")
async def get_overloaded():
    nodes = grid.get_overloaded_nodes()
    return [{"node_id": n.node_id, "name": n.name, "utilization": n.utilization} for n in nodes]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
