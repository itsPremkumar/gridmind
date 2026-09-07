# GridMind AI Platform — AI-Powered Grid Management

An AI-powered grid management platform for optimizing power distribution, predicting load, and detecting anomalies.

## Features

- **Grid Topology** — Node/edge graph with capacity and load tracking
- **AI Optimizer** — Load balancing via gradient-free optimization
- **Load Predictor** — Exponential smoothing with confidence intervals
- **Anomaly Detection** — Statistical outlier detection on grid metrics
- **REST API** — FastAPI-based control plane

## Quick Start

```python
from gridmind.core.grid import GridManager
from gridmind.core.optimizer import GridOptimizer
from gridmind.core.predictor import LoadPredictor

grid = GridManager()
n1 = grid.add_node("Substation A", capacity_mw=100)
n2 = grid.add_node("Substation B", capacity_mw=80)
grid.add_edge(n1.node_id, n2.node_id, capacity_mw=50)

n1.current_load_mw = 85
n2.current_load_mw = 40

optimizer = GridOptimizer(grid)
result = optimizer.optimize()
print(f"Cost reduced: {result.initial_cost:.2f} → {result.final_cost:.2f}")

predictor = LoadPredictor()
for load in [45, 48, 52, 49, 55]:
    predictor.update(load)
pred = predictor.predict(steps_ahead=3)
print(f"Predicted load: {pred.predicted_load_mw:.1f} MW (confidence: {pred.confidence:.0%})")
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT
