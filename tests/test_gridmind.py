"""Tests for GridMind AI Platform."""

import pytest

from gridmind.core.grid import GridManager, GridNode, GridEdge
from gridmind.core.optimizer import GridOptimizer, OptimizationResult
from gridmind.core.predictor import LoadPredictor, Prediction
from gridmind.core.anomaly import AnomalyDetector, AnomalyEvent


class TestGridNode:
    def test_create(self):
        node = GridNode(node_id="n1", name="Generator A", capacity_mw=100.0)
        assert node.node_id == "n1"
        assert node.name == "Generator A"
        assert node.capacity_mw == 100.0
        assert node.status == "active"

    def test_utilization(self):
        node = GridNode(node_id="n1", name="Gen", capacity_mw=100.0, current_load_mw=50.0)
        assert node.utilization == 0.5

    def test_utilization_zero_capacity(self):
        node = GridNode(node_id="n1", name="Gen", capacity_mw=0.0)
        assert node.utilization == 0.0


class TestGridEdge:
    def test_create(self):
        edge = GridEdge(edge_id="e1", source_id="n1", target_id="n2", capacity_mw=50.0)
        assert edge.edge_id == "e1"
        assert edge.source_id == "n1"
        assert edge.target_id == "n2"


class TestGridManager:
    def test_create(self):
        gm = GridManager()
        assert gm is not None

    def test_add_node(self):
        gm = GridManager()
        node = gm.add_node("Generator", 100.0)
        assert node is not None
        assert len(gm.nodes) == 1

    def test_add_edge(self):
        gm = GridManager()
        n1 = gm.add_node("A", 100.0)
        n2 = gm.add_node("B", 50.0)
        edge = gm.add_edge(n1.node_id, n2.node_id, 30.0)
        assert edge is not None
        assert len(gm.edges) == 1

    def test_add_edge_invalid_node(self):
        gm = GridManager()
        edge = gm.add_edge("invalid1", "invalid2", 30.0)
        assert edge is None

    def test_get_node(self):
        gm = GridManager()
        node = gm.add_node("Test", 100.0)
        found = gm.get_node(node.node_id)
        assert found is not None
        assert found.name == "Test"

    def test_get_neighbors(self):
        gm = GridManager()
        n1 = gm.add_node("A", 100.0)
        n2 = gm.add_node("B", 50.0)
        n3 = gm.add_node("C", 75.0)
        gm.add_edge(n1.node_id, n2.node_id, 30.0)
        gm.add_edge(n1.node_id, n3.node_id, 40.0)
        neighbors = gm.get_neighbors(n1.node_id)
        assert len(neighbors) == 2

    def test_total_capacity(self):
        gm = GridManager()
        gm.add_node("A", 100.0)
        gm.add_node("B", 50.0)
        assert gm.total_capacity() == 150.0

    def test_total_load(self):
        gm = GridManager()
        gm.add_node("A", 100.0, current_load_mw=30.0)
        gm.add_node("B", 50.0, current_load_mw=20.0)
        assert gm.total_load() == 50.0

    def test_get_overloaded_nodes(self):
        gm = GridManager()
        gm.add_node("A", 100.0, current_load_mw=90.0)
        gm.add_node("B", 50.0, current_load_mw=10.0)
        overloaded = gm.get_overloaded_nodes(threshold=0.8)
        assert len(overloaded) == 1

    def test_get_stats(self):
        gm = GridManager()
        gm.add_node("A", 100.0)
        gm.add_node("B", 50.0)
        stats = gm.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_capacity_mw"] == 150.0


class TestGridOptimizer:
    def test_create(self):
        gm = GridManager()
        optimizer = GridOptimizer(gm)
        assert optimizer is not None

    def test_compute_cost(self):
        gm = GridManager()
        gm.add_node("A", 100.0, current_load_mw=90.0)
        optimizer = GridOptimizer(gm)
        cost = optimizer.compute_cost()
        assert cost >= 0.0

    def test_suggest_load_transfer(self):
        gm = GridManager()
        n1 = gm.add_node("Overloaded", 100.0, current_load_mw=95.0)
        n2 = gm.add_node("Underloaded", 100.0, current_load_mw=20.0)
        optimizer = GridOptimizer(gm)
        transfer = optimizer.suggest_load_transfer(n1, n2)
        assert transfer is not None
        assert transfer > 0

    def test_optimize(self):
        gm = GridManager()
        n1 = gm.add_node("A", 100.0, current_load_mw=95.0)
        n2 = gm.add_node("B", 100.0, current_load_mw=20.0)
        gm.add_edge(n1.node_id, n2.node_id, 50.0)
        optimizer = GridOptimizer(gm)
        result = optimizer.optimize()
        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0

    def test_predict_optimal_capacity(self):
        gm = GridManager()
        n1 = gm.add_node("A", 100.0, current_load_mw=50.0)
        optimizer = GridOptimizer(gm)
        capacity = optimizer.predict_optimal_capacity(n1.node_id)
        assert capacity > 0


class TestLoadPredictor:
    def test_create(self):
        predictor = LoadPredictor()
        assert predictor is not None

    def test_update(self):
        predictor = LoadPredictor()
        predictor.update(50.0)
        assert len(predictor.history) == 1
        assert predictor._initialized is True

    def test_predict(self):
        predictor = LoadPredictor()
        for i in range(10):
            predictor.update(50.0 + i * 2)
        prediction = predictor.predict(1)
        assert isinstance(prediction, Prediction)
        assert prediction.predicted_load_mw > 0
        assert prediction.confidence > 0

    def test_predict_series(self):
        predictor = LoadPredictor()
        for i in range(5):
            predictor.update(50.0)
        series = predictor.predict_series(5)
        assert len(series) == 5

    def test_detect_trend_increasing(self):
        predictor = LoadPredictor()
        for i in range(10):
            predictor.update(float(i * 10))
        assert predictor.detect_trend() == "increasing"

    def test_detect_trend_stable(self):
        predictor = LoadPredictor()
        for i in range(10):
            predictor.update(50.0)
        assert predictor.detect_trend() == "stable"


class TestAnomalyDetector:
    def test_create(self):
        detector = AnomalyDetector()
        assert detector is not None

    def test_record(self):
        detector = AnomalyDetector()
        detector.record("n1", 50.0)
        assert len(detector.get_history("n1")) == 1

    def test_check_no_anomaly(self):
        detector = AnomalyDetector()
        for i in range(10):
            detector.record("n1", 50.0)
        result = detector.check("n1", 52.0)
        assert result is None

    def test_check_anomaly(self):
        detector = AnomalyDetector()
        for i in range(10):
            detector.record("n1", 50.0)
        result = detector.check("n1", 100.0)
        assert result is not None
        assert isinstance(result, AnomalyEvent)
        assert result.severity in ("low", "medium", "high", "critical")

    def test_check_insufficient_history(self):
        detector = AnomalyDetector()
        for i in range(3):
            detector.record("n1", 50.0)
        result = detector.check("n1", 100.0)
        assert result is None

    def test_get_all_anomalies(self):
        detector = AnomalyDetector()
        for i in range(10):
            detector.record("n1", 50.0)
        detector.record("n1", 200.0)
        anomalies = detector.get_all_anomalies()
        assert len(anomalies) >= 1
